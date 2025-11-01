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
        # 제외할 감정 레이블(선택). 예: ['disgust']
        'exclude': [],
    'weights': {'audio': 0.65, 'text': 0.35},  # 텍스트:오디오 = 3.5:6.5
    'audio_temperature': 0.55,  # 약간 완화(민감도 보수화)
        'av_fusion': {
            'enabled': True,
            # 텍스트 강한 주장일수록 AV 호환성으로 보너스(상대 가중)
            'text_gain': 0.4,
            # 결핍 클래스의 의사 오디오 점수(작게): AV 호환성에 비례해 가산
            'audio_gain': 0.15
        },
        'emotion_weights': {
            'neutral': 0.90,  # 중립 비중을 약간 더 올림
            'happy': 1.1,
            'sad': 1.2,
            'angry': 1.1,
            'fear': 0.9,
            'surprise': 1.0,
            'disgust': 0.8
        },
        'ensemble': {
            'audio_confidence_threshold': 0.35,
            'text_confidence_threshold': 0.60,
            'dominance_margin': 0.10,
            'audio_confidence_boost': 1.8,
            'audio_confidence_decay': 0.55,
            'text_confidence_boost': 1.2,
            'text_confidence_decay': 0.75,
            'neutral_suppression': 0.75,  # 중립 억제 완화(값↑ → 억제량↓)
            'neutral_floor': 0.07
        },
        # 스파이크성 happy/angry 억제를 위한 가드와 시간 스무딩
        'neutral_guard': {
            'enabled': True,
            # happy/angry 허용 기준을 소폭 완화(정상적인 긍정/분노 억제 방지)
            'min_audio_conf': 0.58,     # 오디오 최고값 최소 확신도 (0.62 -> 0.58)
            'min_audio_margin': 0.15,   # 오디오 1-2위 마진 (0.18 -> 0.15)
            'min_text_support': 0.18,   # 텍스트 동일 감정 최소 지지 (0.22 -> 0.18)
            'damp_factor': 0.35,        # 미충족 시 happy/angry 점수 축소 배수
            'target_classes': ['happy', 'angry']
        },
        'temporal_smoothing': {
            'enabled': True,
            'window': 3,               # 좌우 1개씩 검사(총 3)
            'min_conf': 0.65,          # 신뢰도 기준을 완화(0.70 -> 0.65)
            'target_classes': ['happy', 'angry']
        }
    },
    'models': {
        'audio': 'xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned',
        'text': 'j-hartmann/emotion-english-distilroberta-base',
        # 한국어 전용 텍스트 감정 모델 체크포인트가 있을 경우 여기에 지정(없으면 None 유지)
        'text_ko': None,
        'audio_candidates': [
            'xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned'
        ],
        'text_candidates': [
            'j-hartmann/emotion-english-distilroberta-base'
        ],
        'best_pair': {
            'audio': 'xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned',
            'text': 'j-hartmann/emotion-english-distilroberta-base',
            'metrics': {
                'accuracy': 0.581,
                'macro_f1': 0.479,
                'neutral_rate': 0.484
            }
        }
    },
    'colors': {
        'emotion_colors': {
            'neutral': '&H00FFFF',
            'happy': '&H00FF00',
            'sad': '&HE16941',  # 더 밝은 파란색(로열 블루 계열, BGR)
            'angry': '&H0000FF',
            'fear': '&H800080',
            'surprise': '&H00A5FF',
            'disgust': '&H008080'
        },
        'default_color': '&HFFFFFF'
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
            "analysis": {
                # 빠른 실행 모드: 발화속도 계산을 단순화하여 CPU 사용량 감소
                "fast_mode": False
            },
            "voice": {
                # 위치 특화 휴리스틱은 기본 비활성화(일반화 유지)
                "use_peak_position_heuristics": False
            },
            "font": {
                "default_font": "Arial",
                "default_size": 40,
                "min_size": 20,
                "max_size": 60,
                "available_fonts": [
                    "Arial", "Verdana", "Georgia", "Tahoma", "Trebuchet MS",
                    "Times New Roman", "Courier New", "Comic Sans MS"
                ],
                "voice_fonts": {
                    "whisper": "Arial",           # 속삭임: 기본 폰트
                    "normal": "Arial",            # 일반: 기본 폰트
                    "shout": "Arial Black"        # 외침: 두꺼운 폰트
                },
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
                    "neutral": "&H00FFFF",    # 노란색
                    "happy": "&H00FF00",      # 초록색
                    "sad": "&HE16941",        # 더 밝은 파란색(로열 블루, BGR)
                    "angry": "&H0000FF",      # 빨간색
                    "fear": "&H800080",       # 보라색
                    "surprise": "&H00A5FF",   # 주황색
                    "disgust": "&H008080",    # 청록색
                },
                "default_color": "&HFFFFFF",  # 흰색
                "speaker_colors": {
                    "Unknown": "&HFFFFFF",
                    "SPEAKER_1": "&HFFFFFF",
                    "SPEAKER_2": "&HFFFFFF",
                    "SPEAKER_3": "&HFFFFFF",
                    "SPEAKER_4": "&HFFFFFF"
                }
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
                    "audio": 0.65,
                    "text": 0.35
                },
                "audio_temperature": 0.55,
                "av_fusion": {
                    "enabled": True,
                    "text_gain": 0.4,
                    "audio_gain": 0.15
                },
                "emotion_weights": {
                    "neutral": 0.90,   # 중립 비중을 약간 더 높임
                    "happy": 1.2,      # 긍정 감정은 강화
                    "sad": 1.2,        # 슬픔은 추가 보정으로 강화
                    "angry": 1.1,      # 강한 감정도 약간 강화
                    "fear": 0.9,       # 공포는 약간 낮게
                    "surprise": 1.0,   # 기본값
                    "disgust": 0.8     # 혐오는 가장 낮게
                },
                "ensemble": {
                    "audio_confidence_threshold": 0.35,
                    "text_confidence_threshold": 0.60,
                    "dominance_margin": 0.10,
                    "audio_confidence_boost": 1.8,
                    "audio_confidence_decay": 0.55,
                    "text_confidence_boost": 1.2,
                    "text_confidence_decay": 0.75,
                    "neutral_suppression": 0.75,
                    "neutral_floor": 0.07
                },
                "neutral_guard": {
                    "enabled": True,
                    "min_audio_conf": 0.62,
                    "min_audio_margin": 0.18,
                    "min_text_support": 0.22,
                    "damp_factor": 0.35,
                    "target_classes": ["happy", "angry"]
                },
                "temporal_smoothing": {
                    "enabled": True,
                    "window": 3,
                    "min_conf": 0.70,
                    "target_classes": ["happy", "angry"]
                }
            }
        }
        self.config["models"] = {
            "audio": "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
            "text": "j-hartmann/emotion-english-distilroberta-base",
            "audio_candidates": [
                "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
            ],
            "text_candidates": [
                "j-hartmann/emotion-english-distilroberta-base"
            ],
            "best_pair": {
                "audio": "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
                "text": "j-hartmann/emotion-english-distilroberta-base",
                "metrics": {
                    "accuracy": 0.581,
                    "macro_f1": 0.479,
                    "neutral_rate": 0.484
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
