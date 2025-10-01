import os
import whisperx
import torch
import json

# 상대 경로 모듈 가져오기
from .audio_analyzer import AudioAnalyzer
from .emotion_classifier import EmotionClassifier  # 감정 분류기 임포트
from .srt_subtitle_generator import SRTSubtitleGenerator
from .utils import split_segment_by_max_words
from .subtitle_config import load_subtitle_settings
from django.conf import settings
from .model_cache import ModelCache

class SubtitleGenerator:
    def __init__(
            self, audio_path,
            max_words=10, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="float16" if torch.cuda.is_available() else "float32",
            batch_size=4):
        # 경로 설정
        self.audio_path = audio_path
        self.hf_token_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'hf_token.txt')
        self.output_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'tmp', 'result')
        
        self.id = os.path.splitext(os.path.basename(audio_path))[0]
        self.json_path = os.path.join(self.output_path, f"{self.id}_segments.json")
        self.srt_output_path = os.path.join(self.output_path, f"{self.id}_subtitle.srt")
        self.ass_output_path = os.path.join(self.output_path, f"{self.id}_subtitle.ass")

        os.makedirs(self.output_path, exist_ok=True)

        self.max_words = max_words
        self.device = device
        self.compute_type = compute_type
        # CPU 모드에서는 더 작은 배치 크기 사용
        if self.device == "cpu":
            self.batch_size = min(batch_size, 4)
        else:
            self.batch_size = batch_size

        self.model_cache = ModelCache()

        # 파일 존재 여부 확인 추가
        if not os.path.exists(self.audio_path):
            raise FileNotFoundError(f"입력 오디오 파일을 찾을 수 없습니다: {self.audio_path}\n"
                                f"현재 작업 디렉토리: {os.getcwd()}\n"
                                f"입력된 경로: {self.audio_path}")

    def process_video(self, file_format:str = "srt"):
        # WhisperX 모델 불러오기
        model = self.model_cache.whisper_model

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
        if language_code == "en":
            result = whisperx.align(result["segments"], self.model_cache.model_en, self.model_cache.metadata_en, audio, self.device, return_char_alignments=False)
        elif language_code == "ko":
            result = whisperx.align(result["segments"], self.model_cache.model_ko, self.model_cache.metadata_ko, audio, self.device, return_char_alignments=False)
        else:
            model_tmp, metadata_tmp = whisperx.load_align_model(language_code=language_code, device=self.device)
            result = whisperx.align(result["segments"], model_tmp, metadata_tmp, audio, self.device, return_char_alignments=False)
        
        print("화자 분리 수행 중...")
        try:
            diarize_model = self.model_cache.diarize_model
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

        if language_code == 'en':
            # 감정 분류기 초기화 (중복 제거)
            print("감정 분류 모델 로드 중...")
            emotion_classifier = EmotionClassifier(
                device=self.device,
                batch_size=self.batch_size,
                cache_dir=os.path.join(self.output_path, ".cache"),
                file_format=file_format
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
        else:
            print(f"인식된 언어가 'en'이 아닌 '{language_code}'이기 때문에 감정 분석 과정을 생략합니다.")
        
        self._segments_to_json(segments)

        return segments
    
    def _segments_to_json(self, segments:dict):
        with open(self.json_path, "w") as f:
            json.dump(segments, f)
    
    def _load_segments(self):
        with open(self.json_path, "r") as f:
            segments = json.load(f)
        
        return segments
    
    def modify_proper_nouns(self, proper_nouns: list):
        if len(proper_nouns) > 0:
            segments = self._load_segments()
            full_text = " ".join([s['text'] for s in segments])

            # TODO: s['text']로 이루어진 full_text가 아닌 s['words']['word'] 값들을 보고 답변을 생성하도록 수정
            # 현재로서는 key, value가 word 단위로 생성되지 않는 경우가 생김
            response = self.model_cache.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"""
                Ignore all previous instructions. You are starting a new conversation.

                **Role:** You are a text analysis expert specialized in correcting speech-to-text (STT) misrecognitions of proper nouns. Your task is to identify potential misspellings in subtitle text and map them to their correct form.

                **Instructions:**
                1.  **Strictly** analyze the provided `subtitle` text against the `proper_nouns` list.
                2.  Find any substrings in the `subtitle` that are likely misrecognized versions of the `proper_nouns`.
                3.  Your final output must be a single JSON object. The **KEYS** of this object should be the **correct proper nouns**, and the **VALUES** should be the **misrecognized substrings** found in the subtitle text.
                4.  **Crucially, your response MUST be a valid JSON-formatted string. DO NOT use markdown code blocks or any other formatting. Provide the raw JSON object only, with no other text or explanation.**

                **Example Input:**
                - proper_nouns: ["박지훈"]
                - subtitle: "안녕하세요, 박지후를 찾으세요?"

                **Example Output:**
                {{
                "박지훈을": "박지후를", 
                }}

                **Input Data:**
                - proper_nouns: {proper_nouns}
                - subtitle: {full_text}
                """
            )

            try:
                modifications = json.loads(response.text)
                print(f"response 내용: {modifications}")
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"API 응답 처리 오류: {e}")
                print("API 응답이 유효한 JSON 형식이 아니거나 응답 텍스트가 없습니다.")
                print(f"response 내용: {response.text}")
                return

            for segment in segments:
                for modified_text, misrecognized_text in modifications.items():
                    if misrecognized_text in segment['text']:
                        segment['text'] = segment['text'].replace(misrecognized_text, modified_text)
                        for word in segment['words']:
                            word['word'] = word['word'].replace(misrecognized_text, modified_text)

            # 수정된 세그먼트 데이터 저장
            self._segments_to_json(segments)
            
            print("고유명사 교정 완료")
        else:
            print("입력된 고유명사가 없어 고유명사 교정 작업을 생략합니다.")

    def get_speaker_name(self):
        segments = self._load_segments()
        speaker_names = set()

        for segment in segments:
            for s in segment['words']:
                speaker_names.add(s['speaker'])
        
        return list(speaker_names)

    def replace_speaker_name(self, new_names: json):
        """
        new_names: {"SPEAKER_00": "NEW_NAME", }
        """
        segments = self._load_segments()

        for segment in segments:
            for word_data in segment.get('words', []):
                # 딕셔너리 조회로 스피커 이름 존재 여부 확인
                current_name = word_data.get('speaker')
                if current_name in new_names:
                    # 딕셔너리의 새 이름으로 즉시 교체
                    word_data['speaker'] = new_names[current_name]
        
        self._segments_to_json(segments)

        srt_subtitle = self.generate_srt_subtitle()

        return srt_subtitle

    def generate_srt_subtitle(self):
        subtitle_settings = load_subtitle_settings(self.id)

        srt_subtitle_generator = SRTSubtitleGenerator(subtitle_settings=subtitle_settings)

        segments = self._load_segments()

        srt_subtitle_generator.segments_to_srt(segments, self.srt_output_path)

        with open(self.srt_output_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def generate_ass_subtitle(self):
        pass