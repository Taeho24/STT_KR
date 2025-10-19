import os
import uuid

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponseServerError, Http404, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.contrib.auth.models import User # Django 기본 User 모델 임포트
from django.db.models import ObjectDoesNotExist
from celery.result import AsyncResult
from kombu.exceptions import OperationalError as KombuOperationalError

from .models import UserTask, TaskInfo

from .tasks import process_and_generate_srt_task

def get_or_create_anonymous_user(request):
    """세션 ID를 기반으로 익명 사용자 객체를 가져오거나 새로 생성합니다."""
    # 세션에서 저장된 사용자 ID를 확인
    anon_user_id = request.session.get('anon_user_id')
    
    if anon_user_id:
        try:
            # DB에서 기존 사용자 조회
            return User.objects.get(id=anon_user_id)
        except User.DoesNotExist:
            pass # DB에 없으면 새로 생성

    # 사용자 새로 생성 (username을 UUID 기반으로 생성)
    new_username = f"anon_{uuid.uuid4().hex[:10]}"
    # is_staff=False, is_superuser=False로 생성
    user = User.objects.create_user(username=new_username) 

    # 새로 생성된 ID를 세션에 저장
    request.session['anon_user_id'] = user.id
    return user

def get_task_status(request, task_id):
    """
    Client-side Polling API: Celery 작업의 현재 상태를 반환합니다.
    - 작업 진행 중: {"status": "processing"} (JsonResponse)
    - 작업 성공: SRT 텍스트 (HttpResponse, text/plain)
    - 작업 실패/오류: 500 Internal Server Error
    """
    try:
        # Celery 결과 객체 조회
        result = AsyncResult(task_id)
        celery_status = result.status
        
        # DB에서 UserTask와 TaskInfo 조회 (Celery 상태 기반으로 DB 업데이트 및 결과 가져오기)
        user_task = get_object_or_404(UserTask, task_id=task_id)
        task_result = TaskInfo.objects.filter(task=user_task).first()

        # DB 상태 업데이트
        if user_task.status != celery_status:
            user_task.status = celery_status
            user_task.save()
        
        # 작업 완료 여부 확인
        if result.ready():
            if result.successful():
                # 작업 성공 시: Celery에서 최종 결과(SRT)를 가져옵니다.
                srt_text = result.get()
                
                # DB에 결과가 없으면 저장 (안정성 확보)
                if not task_result or task_result.subtitle != srt_text:
                    if task_result:
                        task_result.subtitle = srt_text
                        task_result.save()
                    else:
                        # TaskInfo 객체가 없으면 새로 생성 (에러 방지)
                        TaskInfo.objects.create(task=user_task, subtitle=srt_text, config={})
                
                # 클라이언트에게 SRT 텍스트를 text/plain으로 반환
                return HttpResponse(srt_text, content_type="text/plain; charset=utf-8")

            elif celery_status in ['FAILURE', 'REVOKED']:
                # 작업 실패 또는 취소 시: 서버 에러 반환 (클라이언트 JS가 500을 처리하도록)
                # 클라이언트가 500을 실패로 처리하도록 유도
                print(f"작업 ID {task_id} 실패/취소 상태: {celery_status}")
                return HttpResponseServerError(f"작업 상태: {celery_status}", status=500)
        
        # 작업 진행 중 (PENDING, STARTED 등)
        # 클라이언트에게 진행 중 상태를 JSON으로 반환
        return JsonResponse({"status": "processing", "current_status": celery_status}, status=200)

    except ObjectDoesNotExist:
        # DB에 작업 ID가 없는 경우
        raise Http404("해당 작업 ID를 찾을 수 없습니다.")
    
    except KombuOperationalError:
        # Celery (Redis) 연결 실패 시
        print(f"경고: Celery 브로커 연결 실패. DB 상태 사용.")
        # DB 상태가 PENDING/STARTED이면 'processing'으로 응답
        if user_task.status in ['PENDING', 'STARTED']:
            return JsonResponse({"status": "processing", "current_status": user_task.status}, status=200)
        
        # DB 상태가 실패면 500 응답
        return HttpResponseServerError("자막 생성 실패: 서버 통신 오류", status=500)
        
    except Exception as e:
        print(f"CRITICAL ERROR in get_task_status: {e}")
        return HttpResponseServerError(f"자막 생성 중 심각한 오류 발생: {e}", status=500)

@csrf_exempt
def generate_caption(request):
    if request.method != "POST" or not request.FILES.get("audio"):
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        audio_file = request.FILES["audio"]

        # 파일명 충돌을 피하기 위해 고유한 파일명 사용
        id = uuid.uuid4()
        unique_filename = f"{id}_{audio_file.name}"
        audio_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'tmp', 'asset', unique_filename)

        # django 서버는 windows, celery는 리눅스 환경에서 실행했을 경우의 경로 교정 (django, celery, redis 모두 리눅스 환경에서의 실행을 권장)
        wsl_path = audio_path.replace('C:', '/mnt/c').replace('\\', '/')

        # 파일을 서버에 저장
        with open(audio_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # 고유명사 값 수집
        proper_nouns = request.POST.get("proper_nouns", [])

        subtitle_settings = {
            "font": {
                "default_size": int(request.POST.get("default_font_size", "24")),
                "min_size": int(request.POST.get("min_font_size", "20")),
                "max_size": int(request.POST.get("max_font_size", "28"))
            },
            "hex_colors": {
                "emotion_colors": {
                "neutral": request.POST.get("중립", "#FFFFFF"),
                "happy": request.POST.get("행복", "#00FF00"),
                "sad": request.POST.get("슬픔", "#0000FF"),
                "angry": request.POST.get("분노", "#FF0000"),
                "fear": request.POST.get("공포", "#800080"),
                "surprise": request.POST.get("놀람", "#00FFFF"),
                "disgust": request.POST.get("혐오", "#008080"),
            },
                "default_color": request.POST.get("중립", "#FFFFFF"),
                "highlight_color": request.POST.get("highlight_color", "#FFFF00")
            }
        }

        #=================================================================================================================================
        # 비동기 방식
        #=================================================================================================================================
        # 오디오 처리
        task = process_and_generate_srt_task.delay(audio_path=wsl_path, subtitle_settings=subtitle_settings, proper_nouns=proper_nouns)

        # 작업 ID와 사용자 정보를 DB에 저장하는 로직 추가
        user = get_or_create_anonymous_user(request)
        
        user_task = UserTask.objects.create(
            user=user,
            task_id=task.id
        )

        TaskInfo.objects.create(
            task=user_task,
            subtitle="",          # 빈 문자열(공백)로 초기화
            config=subtitle_settings # 수집한 설정값 저장
        )

        # 클라이언트에게 즉시 202 응답과 작업 ID를 전달
        return JsonResponse({"task_id": task.id}, status=202)
        #=================================================================================================================================

        #=================================================================================================================================
        # 비동기 서버 설정이 되지 않았을 경우 테스트용 기존 방식
        #=================================================================================================================================
        # generator = SubtitleGenerator(audio_path=audio_path)
        # generator.process_video()
        # srt_text = generator.generate_srt_subtitle()
        # 
        # return HttpResponse(srt_text, content_type="text/plain")
        #=================================================================================================================================
    except Exception as e:
        print("자막 생성 중 오류:", e)
        return HttpResponseServerError("자막 생성 실패")

def get_task_list(request, user_id):
    # 사용자가 존재하는지 확인
    user = User.objects.filter(id=user_id).first()

    if not user:
        return render(request, 'taskID.html', {'user_id': user_id, 'tasks': []})
    
    task_list = []
    user_tasks = UserTask.objects.filter(user=user).order_by('-id')
    
    for user_task in user_tasks:
        current_status = user_task.status  # DB의 저장된 상태를 기본값으로 사용
        
        try:
            # Celery (Redis) 연결을 시도하고 최신 상태를 가져옵니다.
            result = AsyncResult(user_task.task_id)
            celery_status = result.status
            
            # DB 상태와 Celery 상태가 다르면 DB를 업데이트합니다.
            if user_task.status != celery_status:
                user_task.status = celery_status
                user_task.save()
            
            current_status = celery_status
            
        except KombuOperationalError:
            # Redis 또는 Celery 연결 실패 시
            print(f"경고: Celery 브로커 연결 실패. Task ID {user_task.task_id}의 DB 상태 사용.")
            # current_status는 DB 상태 그대로 유지됨
            
        except Exception as e:
            # 다른 Celery 관련 오류 처리 (예: Timeouts)
            print(f"경고: Celery 조회 중 오류 ({e}). Task ID {user_task.task_id}의 DB 상태 사용.")
            
        
        task_list.append({
            "task_id": user_task.task_id,
            "status": current_status, 
            "timestamp_id": user_task.id, 
        })
            
    context = {
        'user_id': user_id,
        'tasks': task_list,
    }
    
    return render(request, 'taskID.html', context)

def task_detail(request, task_id):
    try:
        # DB에서 기본 정보 조회 (TaskInfo 모델이 UserTask와 OneToOneField로 연결되어 있다고 가정)
        user_task = get_object_or_404(UserTask, task_id=task_id)
        
        # TaskInfo에서 자막, Config 등의 상세 정보를 가져옵니다.
        # TaskInfo가 UserTask의 PK를 공유하므로, task_id를 통해 접근합니다.
        task_result = TaskInfo.objects.filter(task=user_task).first()
        
        # Celery 상태를 조회하고 DB를 업데이트 (get_caption_status의 로직 일부 재사용)
        current_status = user_task.status
        srt_text = task_result.subtitle if task_result else None
        
        try:
            result = AsyncResult(task_id)
            celery_status = result.status
            
            if user_task.status != celery_status:
                user_task.status = celery_status
                user_task.save()
            
            # Celery가 최종 결과를 가지고 있지만 DB에 없을 경우, DB에 저장 (안정성 확보)
            if celery_status == 'SUCCESS' and (not task_result or not task_result.subtitle):
                srt_text = result.get()
                
                if task_result:
                    task_result.subtitle = srt_text
                    task_result.save()
                else:
                    # TaskInfo 객체가 없으면 새로 생성 (에러 방지)
                    TaskInfo.objects.create(task=user_task, subtitle=srt_text, config={})
            
            current_status = celery_status

        except KombuOperationalError:
            print(f"경고: Celery 연결 실패. DB 상태 사용.")
        except Exception as e:
            print(f"경고: Celery 조회 중 오류. DB 상태 사용. 오류: {e}")

        # Context 구성
        context = {
            'task_id': task_id,
            'user_id': user_task.user_id,
            'status': current_status,
            'created_time': user_task.id, # timestamp_id 대신 실제 생성 시간 필드가 있다면 사용 권장
            'subtitle': srt_text if srt_text else '',
            'config_data': task_result.config if task_result and task_result.config else {},
            'is_finished': (current_status == 'SUCCESS' or current_status == 'FAILURE' or current_status == 'REVOKED'),
        }

        # HTML 템플릿 렌더링
        return render(request, 'taskDetail.html', context)

    except ObjectDoesNotExist:
        # UserTask를 찾을 수 없는 경우
        raise Http404("해당 작업 ID를 찾을 수 없습니다.")
    except Exception as e:
        return HttpResponseServerError(f"작업 상세 정보를 불러오는 데 실패했습니다: {e}")

@csrf_exempt
def delete_task(request, task_id):
    if request.method == 'POST':
        try:
            user_task = get_object_or_404(UserTask, task_id=task_id)
            
            # Celery에서 작업 취소 (진행 중인 경우)
            result = AsyncResult(task_id)
            if not result.ready():
                 result.revoke(terminate=True) # 워커에서 즉시 종료

            # DB에서 UserTask와 TaskInfo (연결된 모든 데이터) 삭제
            user_task.delete() 
            
            return JsonResponse({'message': 'Task deleted successfully'}, status=200)
            
        except ObjectDoesNotExist:
            return JsonResponse({'error': 'Task not found'}, status=404)
        except Exception as e:
            print(f"Task deletion error: {e}")
            return HttpResponseServerError(f"삭제 중 오류 발생: {e}")
    
    return JsonResponse({'error': 'Method not allowed'}, status=405) # POST 외의 요청은 거부

def index(request):
    # 작업 ID와 사용자 정보를 DB에 저장하는 로직 추가
    user = get_or_create_anonymous_user(request)

    context = {
        'user_id': user.id  # 사용자 ID를 Context에 담아 전달
    }

    return render(request, 'index.html', context)