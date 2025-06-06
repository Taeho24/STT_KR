import json
import os
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
ROOT_DIR = Path(__file__).parent

# 필요한 디렉토리 구조 설정
DIRS = {
    'assets': ROOT_DIR / 'assets',      # 입력 비디오 저장
    'result': ROOT_DIR / 'result',      # 결과물 저장
    'private': ROOT_DIR / 'private',    # 토큰 등 민감한 파일 저장
    'cache': ROOT_DIR / '.cache'        # 캐시 파일 저장
}

# 디렉토리 생성
for dir_path in DIRS.values():
    dir_path.mkdir(exist_ok=True)

# 설정값
CONFIG = {
    'paths': DIRS,
    'emotions': {
        'mapping': {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'fear': 'fear',
            'surprise': 'surprise',
            'disgust': 'disgust',
            'neutral': 'neutral'
        },
        'weights': {'audio': 0.7, 'text': 0.3},
        'emotion_weights': {
            'neutral': 0.9,
            'happy': 1.1,
            'sad': 1.0,
            'angry': 1.1,
            'fear': 0.9,
            'surprise': 1.0,
            'disgust': 0.8
        }
    },
    'colors': {
        "emotion_colors": {
            "neutral": "#FFFFFF",    # 노란색
            "happy": "#00FF00",      # 초록색
            "sad": "#0000FF",        # 빨간색
            "angry": "#FF0000",      # 빨간색
            "fear": "#800080",       # 보라색
            "surprise": "#00FFFF",   # 민트색
            "disgust": "#008080",    # 청록색
        },
        "default_color": "#FFFFFF",  # 흰색
        "highlight_color": "#FFFF00",   # 노란색
    }
}

def get(section, key, default=None):
    """설정값 가져오기"""
    return CONFIG.get(section, {}).get(key, default)

class SubtitleConfig:
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self._load_default_config()
        self.load_config()
    
    def _load_default_config(self):
        """기본 설정값 정의"""
        self.config = {
            "font": {
                "default_font": "Arial",
                "default_size": 24,
                "min_size": 16,
                "max_size": 36,
                "available_fonts": [
                    "Arial", "Verdana", "Georgia", "Tahoma", "Trebuchet MS",
                    "Times New Roman", "Courier New", "Comic Sans MS"
                ],
                "size_multipliers": {
                    "whisper": 0.7,
                    "normal": 1.0,
                    "shout": 1.5
                }
            },
            "model_weights": {
                "multimodal": {
                    "audio": 0.6,
                    "text": 0.4
                }
            },
            "colors": {
                "emotion_colors": {
                    "neutral": "#FFFFFF",    # 노란색
                    "happy": "#00FF00",      # 초록색
                    "sad": "#0000FF",        # 빨간색
                    "angry": "#FF0000",      # 빨간색
                    "fear": "#800080",       # 보라색
                    "surprise": "#00FFFF",   # 민트색
                    "disgust": "#008080",    # 청록색
                },
                "default_color": "#FFFFFF",  # 흰색
                "highlight_color": "#FFFF00",   # 노란색
            },
            "spacing": {
                "speech_rate": {
                    "slow": 5,      # 10 -> 5로 수정
                    "normal": 0,     
                    "fast": -2      # -5 -> -2로 수정
                }
            },
            "outline": {
                "pitch_levels": {
                    "low": 4,
                    "normal": 2,
                    "high": 1
                }
            },
            "volume": {
                "sizes": {
                    "soft": 30,
                    "normal": 40,
                    "loud": 50
                }
            },
            "emotions": {
                "mapping": {
                    "sadness": "sad",
                    "anger": "angry",
                    "joy": "happy",
                    "fear": "fear",
                    "surprise": "surprise",
                    "disgust": "disgust",
                    "neutral": "neutral"
                },
                "weights": {
                    "audio": 0.6,
                    "text": 0.4
                },
                "emotion_weights": {
                    "neutral": 0.9,    # 중립은 약간 낮게
                    "happy": 1.2,      # 긍정 감정은 강화
                    "sad": 1.0,        # 기본값
                    "angry": 1.1,      # 강한 감정도 약간 강화
                    "fear": 0.9,       # 공포는 약간 낮게
                    "surprise": 1.0,   # 기본값
                    "disgust": 0.8     # 혐오는 가장 낮게
                }
            }
        }

        def validate_color_code(code):
            """ASS 색상 코드 유효성 검사"""
            if not code.startswith('&H'):
                return False
            try:
                # AABBGGRR 형식 검증
                int(code[2:], 16)
                return len(code) == 8
            except ValueError:
                return False

        # 감정 색상 유효성 검사
        for emotion, color in self.config['colors']['emotion_colors'].items():
            if not validate_color_code(color):
                print(f"Warning: Invalid color code for {emotion}: {color}")
                # 기본 색상으로 대체
                self.config['colors']['emotion_colors'][emotion] = '&HFFFFFF'

    def load_config(self):
        """설정 파일 로드"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # 기존 설정과 병합
                self._merge_config(loaded_config)
                print(f"설정을 로드했습니다: {self.config_path}")
        except Exception as e:
            print(f"설정 로드 실패, 기본값을 사용합니다: {e}")

    def save_config(self):
        """현재 설정을 파일로 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            print(f"설정을 저장했습니다: {self.config_path}")
        except Exception as e:
            print(f"설정 저장 실패: {e}")

    def _merge_config(self, new_config):
        """새로운 설정을 기존 설정과 병합"""
        def merge_dicts(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    merge_dicts(d1[k], v)
                else:
                    d1[k] = v
        merge_dicts(self.config, new_config)

    def get(self, *keys, default=None):
        """설정값 가져오기"""
        value = self.config
        for key in keys:
            try:
                value = value[key]
            except (KeyError, TypeError):
                return default
        return value

    def set(self, value, *keys):
        """설정값 변경"""
        if not keys:
            return False
        
        target = self.config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        
        target[keys[-1]] = value
        return True

# 전역 설정 인스턴스
config = SubtitleConfig()
