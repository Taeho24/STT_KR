import whisperx
from google import genai

class ModelCache:
    _instance = None
    whisper_model = None
    align_model = None
    diarize_model = None
    emotion_classifier = None
    client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_models(cls, device, compute_type, auth_token, gemini_api_key):
        try:
            print("WhisperX 모델 로드...")
            # 모델 종류: large-v3, large-v2, medium
            cls.whisper_model = whisperx.load_model("large-v2", device=device, compute_type=compute_type, )

            print("en 모델 로드...")
            cls.model_en, cls.metadata_en = whisperx.load_align_model(language_code="en", device=device)

            print("ko 모델 로드...")
            cls.model_ko, cls.metadata_ko = whisperx.load_align_model(language_code="ko", device=device)

            print("화자 분리 모델 로드...")
            try:
                cls.diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.0", use_auth_token=auth_token, device=device)
            except Exception as e:
                print(f"화자 분리 모델 로드 실패: {e}")
            
            print("모든 모델 로드 완료")
        except Exception as e:
            print(f"ERROR: WhisperX 모델 로드 실패 - {e}")
            raise e

        # gemini client 등록
        if gemini_api_key:
            cls.client = genai.Client(api_key=gemini_api_key)
            print("gemini client 등록 완료")
        else:
            print("gemini api key가 유효하지 않습니다.")