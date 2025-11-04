import os
import json

from django.conf import settings

config_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'tmp', 'config')
# 설정 디렉토리 보장
os.makedirs(config_path, exist_ok=True)

def set_subtitle_settings(id, subtitle_settings):
    """
    [예시]
    subtitle_settings = {
        "font": {
            "default_size": 24,
            "min_size": 16,
            "max_size": 36
        },
        "hex_colors": {
            "emotion_colors": {
                "neutral": "#FFFFFF",
                "happy": "#00FF00",
                "sad": "#0000FF",
                "angry": "#FF0000",
                "fear": "#800080",
                "surprise": "#00FFFF",
                "disgust": "#008080"
            },
            "default_color": "#FFFFFF",
            "highlight_color": "#FFFF00"
        }
    }
    """
    json_path = os.path.join(config_path, f"{id}_config.json")

    # 딕셔너리를 JSON 문자열로 변환 (들여쓰기 적용)
    json_string = json.dumps(subtitle_settings, ensure_ascii=False)

    # JSON 문자열을 파일로 저장
    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(json_string)

    print(f"JSON 파일이 '{json_path}'에 성공적으로 저장되었습니다.")

def load_subtitle_settings(id):
    """
    JSON 파일에서 자막 설정값을 불러옵니다.
    """
    json_path = os.path.join(config_path, f"{id}_config.json")

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)
        return settings
    except FileNotFoundError:
        print(f"오류: 설정 파일 '{json_path}'을(를) 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        print(f"오류: '{json_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return None