import whisperx
from google import genai

import torch
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

from torch.torch_version import TorchVersion

from pyannote.audio.core.model import Introspection
from pyannote.audio.core.task import Specifications
from pyannote.audio.core.task import Problem
from pyannote.audio.core.task import Resolution

from builtins import list, dict, tuple, int, float, bool, str, set # 기본 타입
from typing import Any, Tuple, Dict, Union, Optional, List, Set, FrozenSet # typing 모듈
from collections import defaultdict, OrderedDict, deque # collections 모듈

import pytorch_lightning.trainer.states # PyTorch Lightning 상태 관리

# PyTorch Lightning 및 기타 라이브러리에서 흔히 사용되는 클래스 임포트
try:
    from torchmetrics.metric import Metric # torchmetrics 사용하는 경우
except ImportError:
    # torchmetrics가 없다면 무시
    Metric = None

# 안전 목록에 추가할 클래스 리스트 (List comprehension으로 정리)
safe_globals_list = [
    # Omegaconf 관련 (필수)
    ListConfig, 
    DictConfig, 
    ContainerMetadata,
    AnyNode,
    Metadata,
    
    # Python 표준 builtins (이전 오류 포함)
    list, dict, tuple, int, float, bool, str, set,
    
    # typing 모듈 (이전 오류 포함)
    Any, Tuple, Dict, Union, Optional, List, Set, FrozenSet,
    
    # collections 모듈 (이전 오류 포함)
    defaultdict, OrderedDict, deque,
    
    # PyTorch Lightning 내부 클래스 (필요할 가능성이 높음)
    getattr(pytorch_lightning.trainer.states, 'RunningStage', None),
    getattr(pytorch_lightning.trainer.states, 'TrainerState', None),
    
    # PyTorch 내부 클래스
    TorchVersion,

    # pyannote.audio 관련
    Introspection,
    Specifications,
    Problem,
    Resolution,

    # torchmetrics 클래스 (Metric 저장 시 필요)
    Metric,
]

# None 값을 제거하고 실제 로드된 클래스만 필터링하여 안전 목록에 추가
safe_globals_to_add = [g for g in safe_globals_list if g is not None]

# 안전 목록 등록
torch.serialization.add_safe_globals(safe_globals_to_add)

print("INFO: PyTorch 2.6 보안 정책 우회를 위해 모델 체크포인트 클래스들을 안전 목록에 추가했습니다.")

class ModelCache:
    _instance = None
    _initialized = False

    medium_model = None
    large_v2_model = None
    large_v3_model = None
    model_en = None
    model_ko = None
    metadata_en = None
    metadata_ko = None
    align_model = None
    diarize_model = None
    client = None

    device=None
    compute_type=None
    auth_token=None
    gemini_api_key=None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, device=None, compute_type=None, auth_token=None, gemini_api_key=None):
        # 이미 초기화된 경우, 아무것도 하지 않고 즉시 반환
        if ModelCache._initialized:
            return

        # 인스턴스 변수에 설정값 저장
        self.device = device
        self.compute_type = compute_type
        self.auth_token = auth_token
        self.gemini_api_key = gemini_api_key
        
        ModelCache._initialized = True

    @classmethod
    def load_models(cls):
        instance = cls._instance
        if instance is None:
            print("ERROR: ModelCache instance not found.")
            return
        
        try:
            print("en 모델 로드...")
            cls.model_en, cls.metadata_en = whisperx.load_align_model(language_code="en", device=cls.device)

            print("ko 모델 로드...")
            cls.model_ko, cls.metadata_ko = whisperx.load_align_model(language_code="ko", device=cls.device)

            print("화자 분리 모델 로드...")
            try:
                cls.diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.0", use_auth_token=instance.auth_token, device=instance.device)
            except Exception as e:
                print(f"화자 분리 모델 로드 실패: {e}")
            
            print("모든 모델 로드 완료")
        except Exception as e:
            print(f"ERROR: WhisperX 모델 로드 실패 - {e}")
            raise e

        # gemini client 등록
        if instance.gemini_api_key:
            cls.client = genai.Client(api_key=instance.gemini_api_key)
            print("gemini client 등록 완료")
        else:
            print("gemini api key가 유효하지 않습니다.")
    
    @classmethod
    def load_whisperx_model(cls, model):
        print("WhisperX 모델 로드...")
        instance = cls._instance

        # 모델 종류: large-v3, large-v2, medium
        if model == "medium":
            cls.medium_model = whisperx.load_model("medium", device=instance.device, compute_type=instance.compute_type, )
            return cls.medium_model
        elif model == "large-v2":
            cls.large_v2_model = whisperx.load_model("large-v2", device=instance.device, compute_type=instance.compute_type, )
            return cls.large_v2_model
        elif model == "large-v3":
            cls.large_v3_model = whisperx.load_model("large-v3", device=instance.device, compute_type=instance.compute_type, )
            return cls.large_v3_model
        else:
            print("유효하지 않은 모델입니다.")
            return None