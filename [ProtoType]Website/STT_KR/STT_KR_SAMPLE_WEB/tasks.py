import os
import json

from STT_KR.celery import app
from .utils.subtitle_generator import SubtitleGenerator 
from .utils.subtitle_config import set_subtitle_settings

@app.task()
def process_and_generate_srt_task(audio_path, proper_nouns: list, subtitle_settings: json, file_format: str = "srt"):
    """
    음성 인식부터 SRT 자막 생성까지의 전체 작업을 수행하는 Celery 태스크

    [사용 예시]
    task = process_and_generate_srt_task.delay(audio_path)
    """
    id = os.path.splitext(os.path.basename(audio_path))[0]

    try:
        sg = SubtitleGenerator(audio_path=audio_path)
        
        sg.process_video(file_format=file_format)

        set_subtitle_settings(id, subtitle_settings)

        if len(proper_nouns) > 0:
            sg.modify_proper_nouns(proper_nouns)
        
        srt_text = sg.generate_srt_subtitle()
        
        return srt_text
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        # 파일이 없을 경우 태스크를 실패로 기록합니다.
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e