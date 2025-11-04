import json

from django.db.models import ObjectDoesNotExist
from django.shortcuts import get_object_or_404

from ..models import UserTask, TaskInfo

class DBManager:
    def __init__(self, task_id):
        self.task_id = task_id

    def get_user_task(self):
        try:
            user_task = get_object_or_404(UserTask, task_id=self.task_id)
            
            return user_task
            
        except ObjectDoesNotExist:
            print(f"오류: Task ID user_task에 대한 설정 데이터를 DB에서 찾을 수 없습니다.")
            # 찾지 못하면 기본 설정 또는 빈 딕셔너리를 반환하여 오류 방지
            return {}
    
    def load_user_id(self):
        try:
            user_task = get_object_or_404(UserTask, task_id=self.task_id)
            
            return user_task.user_id
            
        except ObjectDoesNotExist:
            print(f"오류: Task ID user_task에 대한 설정 데이터를 DB에서 찾을 수 없습니다.")
            # 찾지 못하면 기본 설정 또는 빈 딕셔너리를 반환하여 오류 방지
            return -1
    
    def load_index(self):
        try:
            user_task = get_object_or_404(UserTask, task_id=self.task_id)
            
            return user_task.id
            
        except ObjectDoesNotExist:
            print(f"오류: Task ID user_task에 대한 설정 데이터를 DB에서 찾을 수 없습니다.")
            # 찾지 못하면 기본 설정 또는 빈 딕셔너리를 반환하여 오류 방지
            return -1

    def get_task(self):
        try:
            user_task = get_object_or_404(UserTask, task_id=self.task_id)
            task_info = TaskInfo.objects.get(task=user_task)
            
            return task_info
            
        except ObjectDoesNotExist:
            print(f"오류: Task ID task_id에 대한 설정 데이터를 DB에서 찾을 수 없습니다.")
            # 찾지 못하면 기본 설정 또는 빈 딕셔너리를 반환하여 오류 방지
            return {}
    
    def update_task_status(self, status):
        try:
            # UserTask 객체를 불러오기
            user_task = self.get_user_task()
            
            # UserTask 모델의 status 필드에 데이터를 JSON 형태로 저장
            user_task.status = status
            user_task.save()

        except Exception as e:
            print(f"오류: DB에 status 저장 중 예기치 않은 오류 발생: {e}")
    
    def load_task_status(self):
        try:
            # UserTask 객체를 불러오기
            user_task = self.get_user_task()
            
            # UserTask 모델의 status 데이터 불러오기
            return user_task.status

        except Exception as e:
            print(f"오류: DB에 status 로드 중 예기치 않은 오류 발생: {e}")
    
    def update_config(self, config:dict):
        try:
            # UserTask를 조회하여 TaskInfo 객체를 불러오기
            task_info = self.get_task()
            
            # TaskInfo 모델의 config 필드에 데이터를 JSON 형태로 저장
            task_info.config = config
            task_info.save()

        except Exception as e:
            print(f"오류: DB에 segments 저장 중 예기치 않은 오류 발생: {e}")
    
    def load_config(self):
        try:
            # Task ID를 사용하여 TaskInfo 레코드를 조회하고 config 데이터를 반환
            task_info = self.get_task()
            
            # config는 JSONField이므로 바로 딕셔너리 형태로 반환됩니다.
            return task_info.config
        
        except Exception as e:
            print(f"오류: DB에 config 로드 중 예기치 않은 오류 발생: {e}")
        
    def update_segment(self, segment:dict):
        try:
            # UserTask를 조회하여 TaskInfo 객체를 불러오기
            task_info = self.get_task()
            
            # TaskInfo 모델의 segment 필드에 데이터를 JSON 형태로 저장
            task_info.segment = segment
            task_info.save()

        except Exception as e:
            print(f"오류: DB에 segments 저장 중 예기치 않은 오류 발생: {e}")
    
    def load_segment(self):
        try:
            task_info = self.get_task()

            # TaskInfo의 segment 필드에서 데이터를 로드
            if task_info.segment:
                return task_info.segment
            else:
                print(f"경고: Task ID {self.task_id}의 DB에 segment 데이터가 비어 있습니다.")
                return []
            
        except Exception as e:
            print(f"오류: DB에서 segments 로드 중 오류 발생: {e}")
            return []
    
    def update_srt_subtitle(self, srt_subtitle):
        try:
            # UserTask를 조회하여 TaskInfo 객체를 불러오기
            task_info = self.get_task()
            
            # TaskInfo 모델의 srt_subtitle 필드에 데이터를 JSON 형태로 저장
            task_info.srt_subtitle = srt_subtitle
            task_info.save()

        except Exception as e:
            print(f"오류: DB에 srt_subtitle 저장 중 예기치 않은 오류 발생: {e}")
    
    def load_srt_subtitle(self):
        try:
            task_info = self.get_task()

            # TaskInfo의 srt_subtitle 필드에서 데이터를 로드
            if task_info.srt_subtitle != "":
                return task_info.srt_subtitle
            else:
                print(f"경고: Task ID task_id의 DB에 srt_subtitle 데이터가 비어 있습니다.")
                return ""
            
        except Exception as e:
            print(f"오류: DB에서 segments 로드 중 오류 발생: {e}")
            return ""
    
    def update_file_name(self, file_name):
        try:
            # UserTask를 조회하여 TaskInfo 객체를 불러오기
            task_info = self.get_task()
            
            # TaskInfo 모델의 file_name 필드에 데이터를 JSON 형태로 저장
            task_info.file_name = file_name
            task_info.save()

        except Exception as e:
            print(f"오류: DB에 file_name 저장 중 예기치 않은 오류 발생: {e}")
    
    def load_file_name(self):
        try:
            task_info = self.get_task()

            # TaskInfo의 file_name 필드에서 데이터를 로드
            if task_info.file_name != "":
                return task_info.file_name
            else:
                print(f"경고: Task ID task_id의 DB에 file_name 데이터가 비어 있습니다.")
                return ""
            
        except Exception as e:
            print(f"오류: DB에서 segments 로드 중 오류 발생: {e}")
            return ""
        
    def update_speaker_name(self, new_names: dict):
        """
        new_names: {"SPEAKER_00": "NEW_NAME", }
        """
        segments = self.load_segment()

        for segment in segments:
            current_segment_name = segment['speaker']
            if current_segment_name in new_names:
                # 딕셔너리의 새 이름으로 즉시 교체
                segment['speaker'] = new_names[current_segment_name]
            for word_data in segment.get('words', []):
                # 딕셔너리 조회로 스피커 이름 존재 여부 확인
                current_name = word_data.get('speaker')
                if current_name in new_names:
                    # 딕셔너리의 새 이름으로 즉시 교체
                    word_data['speaker'] = new_names[current_name]
        
        return segments
        
    def get_speaker_name(self):
        segments = self.load_segment() # segments 데이터 로드.

        # 고유한 화자 이름을 저장할 set 생성
        unique_speakers = set()

        for segment in segments:
            for s in segment.get('words', []): 
                # 'speaker' 키가 있는지 확인하여 set에 추가
                if 'speaker' in s:
                    unique_speakers.add(s['speaker'])
        
        # set에 있는 각 화자 이름을 key로, "" (빈 문자열)을 value로 갖는 딕셔너리를 생성
        speaker_dict = {name: "" for name in unique_speakers}
        
        return speaker_dict