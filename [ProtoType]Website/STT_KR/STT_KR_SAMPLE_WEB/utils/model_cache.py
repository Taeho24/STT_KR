import importlib

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
            # 지연 로딩: whisperx 모듈은 여기서 import
            whisperx = importlib.import_module("whisperx")
            print("WhisperX 모델 로드...")
            # 모델 로드: 큰 모델부터 시도하되 실패 시 보수적으로 자동 폴백
            load_order = ["large-v2", "medium", "small"]
            last_err = None
            for model_name in load_order:
                try:
                    print(f"시도: Whisper 모델 '{model_name}' 로드")
                    cls.whisper_model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
                    print(f"성공: Whisper 모델 '{model_name}' 로드 완료")
                    break
                except Exception as e:
                    last_err = e
                    print(f"경고: 모델 '{model_name}' 로드 실패 → 다음 후보 시도: {e}")
                    cls.whisper_model = None
            if cls.whisper_model is None:
                raise last_err or RuntimeError("Whisper 모델 로드 실패")

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

        # gemini client 등록(필요 시에만 import)
        try:
            if gemini_api_key:
                genai = importlib.import_module("google.genai")
                cls.client = genai.Client(api_key=gemini_api_key)
                print("gemini client 등록 완료")
            else:
                print("gemini api key가 유효하지 않습니다.")
        except Exception as e:
            print(f"gemini client 초기화 실패: {e}")