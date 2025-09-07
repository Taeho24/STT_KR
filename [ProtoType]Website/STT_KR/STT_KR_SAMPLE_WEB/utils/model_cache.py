import whisperx
from google import genai

class ModelCache:
    _instance = None
    whisper_model = None
    align_model = None
    diarize_model = None
    emotion_classifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_models(cls, device, compute_type, auth_token, gemini_api_key):
        print("WhisperX 모델 로드...")
        cls.whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)

        print("화자 분리 모델 로드...")
        try:
            cls.diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.0", use_auth_token=auth_token, device=device)
        except Exception as e:
            print(f"화자 분리 모델 로드 실패: {e}")

        """
        감정 분류 모델 로드 부분 추가
        print("감정 분류 모델을 로드합니다...")
        cls.emotion_classifier = EmotionClassifier(device=device, batch_size=16)
        """
        
        print("모든 모델 로딩 완료")

        cls.client = genai.Client(api_key=gemini_api_key)