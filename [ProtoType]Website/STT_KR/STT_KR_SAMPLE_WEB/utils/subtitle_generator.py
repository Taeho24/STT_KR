import os
import whisperx
import torch

# 상대 경로 모듈 가져오기
from .audio_analyzer import AudioAnalyzer
from .emotion_classifier import EmotionClassifier  # 감정 분류기 임포트
from .srt_subtitle_generator import SRTSubtitleGenerator
from .utils import read_auth_token, split_segment_by_max_words
from django.conf import settings

class SubtitleGenerator:
    def __init__(
            self, audio_path,
            max_words=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32",
            batch_size=16):
        # 경로 설정
        self.audio_path = audio_path
        self.hf_token_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'hf_token.txt')
        self.output_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'results')
        self.srt_output_path = os.path.join(self.output_path, f"subtitle.srt")
        self.ass_output_path = os.path.join(self.output_path, f"subtitle.ass")

        os.makedirs(self.output_path, exist_ok=True)

        self.auth_token = read_auth_token(self.hf_token_path)

        self.max_words = max_words
        self.device = device
        self.compute_type = compute_type
        # CPU 모드에서는 더 작은 배치 크기 사용
        if self.device == "cpu":
            self.batch_size = min(batch_size, 4)
        else:
            self.batch_size = batch_size

        # 파일 존재 여부 확인 추가
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"입력 오디오 파일을 찾을 수 없습니다: {self.audio_path}\n"
                                f"현재 작업 디렉토리: {os.getcwd()}\n"
                                f"입력된 경로: {self.audio_path}")

    def _process_extra_video(self):
        """추가 오디오 처리 (음성 인식 및 세그먼트 분할)"""
        print(f"\n추가 오디오 처리 시작: {self.audio_path}")
        
        vad_options = {"use_vad": True}
        model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type, vad_options=vad_options)
        
        print("오디오 로딩 중...")
        audio = whisperx.load_audio(self.audio_path)
        
        print("음성 인식(STT) 수행 중...")
        result = model.transcribe(audio, batch_size=16)
        language_code = result["language"]
        print(f"감지된 언어: {language_code}")
        
        print("음성 정렬 수행 중...")
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        
        segments = split_segment_by_max_words(result["segments"], 10)
        segments = [s for s in segments if (s["end"] - s["start"]) > 0.7]
        print(f"분할된 세그먼트 수: {len(segments)}")
        
        audio_analyzer = AudioAnalyzer(sample_rate=16000)
        segments = audio_analyzer.analyze_audio_features(segments, audio)
        
        return segments

    def _process_video(self):
        # ✅ 미리 감정 분류 모델 로드 시작
        print("MIT/AST 및 RoBERTa 감정 분류 모델 로드 중...")
        
        emotion_classifier = EmotionClassifier(
            device=self.device,
            batch_size=self.batch_size,
            cache_dir=os.path.join(self.output_path, ".cache")
        )
        print("감정 분류 모델 로드 완료")

        # WhisperX 모델 로드 부분 수정
        print("WhisperX 모델 로드 중...")
        model = whisperx.load_model(
            "large-v2",
            self.device,
            compute_type=self.compute_type
        )

        print(f"오디오 추출 중: {self.audio_path}")
        audio = whisperx.load_audio(self.audio_path)

        print("음성 인식(STT) 수행 중...")
        result = model.transcribe(
            audio,
            batch_size=self.batch_size
        )

        # language_code 추출 추가
        language_code = result["language"]

        print("음성 정렬 수행 중...")
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        print("화자 분리 수행 중...")
        try:
            diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.0", use_auth_token=self.auth_token, device=self.device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            print(f"화자 분리 중 오류 발생: {str(e)}")
            print("화자 분리 없이 계속 진행합니다.")
            
        # 화자 분리 실패시 기본값 설정
        for segment in result["segments"]:
            if "speaker" not in segment:
                segment["speaker"] = "Unknown"
            for word in segment.get("words", []):
                if "speaker" not in word:
                    word["speaker"] = segment["speaker"]

        segments = split_segment_by_max_words(result["segments"], self.max_words)
        # 최소 지속 시간을 0.7초에서 0.2초로 변경
        # 너무 짧은 노이즈만 제거 (200ms 미만)
        segments = [s for s in segments if (s["end"] - s["start"]) > 0.2]
        print(f"분할된 세그먼트 수: {len(segments)}")

        print("오디오 특성 분석 중...")
        audio_analyzer = AudioAnalyzer(sample_rate=16000)
        segments = audio_analyzer.analyze_voice_type(segments, audio)

        # 감정 분류기 초기화 (중복 제거)
        print("감정 분류 모델 로드 중...")
        emotion_classifier = EmotionClassifier(
            device=self.device,
            batch_size=self.batch_size,
            cache_dir=os.path.join(self.output_path, ".cache")
        )
        print("감정 분류 모델 로드 완료")

        # 감정 분석 배치 처리
        print("감정 분석 중...")
        segments = emotion_classifier.classify_emotions(segments, full_audio=audio)

        # 감정 분석 통계 출력
        emotion_stats = {}
        for segment in segments:
            emotion = segment.get('emotion', 'unknown')
            emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

        print("\n===== 감정 분석 통계 =====")
        total_segments = len(segments)
        for emotion, count in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_segments) * 100
            print(f" - {emotion}: {count}개 세그먼트 ({percentage:.1f}%)")
        print("=======================\n")

        return segments

    def generate_srt_subtitle(self):
        srt_subtitle_generator = SRTSubtitleGenerator()

        segments = self._process_video()
        srt_subtitle_generator.segments_to_srt(segments, self.srt_output_path)

        with open(self.srt_output_path, 'r', encoding='utf-8') as f:
            return f.read()