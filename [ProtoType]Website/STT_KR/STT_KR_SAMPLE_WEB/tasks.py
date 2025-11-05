import os

from STT_KR.celery import app
from .utils.subtitle_generator import SubtitleGenerator 
from .utils.db_manager import DBManager

@app.task(bind=True)
def process_and_generate_srt_task(self, audio_path, proper_nouns: list, file_format: str = "srt", model: str = "large-v2"):
    """
    음성 인식부터 SRT 자막 생성까지의 전체 작업을 수행하는 Celery 태스크

    [사용 예시]
    task = process_and_generate_srt_task.delay(audio_path)
    """
    try:
        task_id = self.request.id

        sg = SubtitleGenerator(audio_path=audio_path, task_id=task_id, model=model)
        
        sg.process_video(file_format=file_format)

        if len(proper_nouns) > 0:
            sg.modify_proper_nouns(proper_nouns)
        
        srt_subtitle = sg.generate_srt_subtitle()

        db_manager = DBManager(task_id)

        db_manager.update_srt_subtitle(srt_subtitle)
        
        return srt_subtitle
    except FileNotFoundError as e:
        # 오류 발생 시 오디오 파일 삭제
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"입력 오디오 파일 삭제 완료: {audio_path}")
            else:
                print(f"경고: 삭제하려던 오디오 파일이 이미 존재하지 않습니다: {audio_path}")
        except Exception as e:
            # 권한 문제 등으로 삭제가 실패할 경우 경고만 출력하고 계속 진행
            print(f"오디오 파일 삭제 실패: {e}")

        print(f"Error: {e}")
        # 파일이 없을 경우 태스크를 실패로 기록합니다.
        raise e
    except Exception as e:
        # 오류 발생 시 오디오 파일 삭제
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"입력 오디오 파일 삭제 완료: {audio_path}")
            else:
                print(f"경고: 삭제하려던 오디오 파일이 이미 존재하지 않습니다: {audio_path}")
        except Exception as e:
            # 권한 문제 등으로 삭제가 실패할 경우 경고만 출력하고 계속 진행
            print(f"오디오 파일 삭제 실패: {e}")

        print(f"An unexpected error occurred: {e}")
        raise e