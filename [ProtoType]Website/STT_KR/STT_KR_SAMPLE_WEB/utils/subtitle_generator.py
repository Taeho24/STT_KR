import whisperx
import torch
import numpy as np
import os
from django.conf import settings

class SubtitleGenerator:
    def __init__(self, audio_path, 
                 highlight_color="yellow",
                 default_font_size=12,
                 min_font_size=10,
                 max_font_size=14,
                 max_words=10,  # 자막 세그먼트 당 최대 단어 개수 (최대 10개 단어 단위로 나누어 자막 생성)
                 device="cuda", # GPU 사용
                 batch_size=16,
                 compute_type="float16"):
        
        self.audio_path = audio_path
        self.output_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'results', 'subtitle.srt')
        self.hf_token_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'hf_token.txt')
        
        self.highlight_color = highlight_color
        self.default_font_size = default_font_size
        self.whispering_font_size = min_font_size
        self.shouting_font_size = max_font_size
        self.max_words = max_words
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type

        self.audio = whisperx.load_audio(self.audio_path)
        self.auth_token = self.read_auth_token()

    # STT_KR_SAMPLE_WEB/static/private/hf_token.txt에서 hugging face 읽기 토큰 불러오기
    def read_auth_token(self):
        try:
            with open(self.hf_token_path, "r") as file:
                return file.read().strip()
        except Exception as e:
            print(f"토큰 파일 오류: {e}")
            return None

    def split_segment_by_max_words(self, segments):
        new_segments = []
        for segment in segments:
            words = segment.get("words", [])
            if len(words) > self.max_words:
                num_chunks = len(words) // self.max_words + (1 if len(words) % self.max_words > 0 else 0)
                words_per_chunk = len(words) // num_chunks
                for i in range(num_chunks):
                    chunk = words[i * words_per_chunk: (i + 1) * words_per_chunk if i != num_chunks - 1 else len(words)]
                    if chunk:
                        new_segments.append({
                            "start": chunk[0]["start"],
                            "end": chunk[-1]["end"],
                            "text": " ".join([w["word"] for w in chunk]),
                            "words": chunk
                        })
            else:
                new_segments.append(segment)
        return new_segments

    def analyze_voice_type(self, segments, sample_rate=16000):
        def compute_rms(audio_segment):
            if isinstance(audio_segment, np.ndarray):
                audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
            return torch.sqrt(torch.mean(audio_segment ** 2)).item()

        for segment in segments:
            for word in segment.get("words", []):
                if "start" not in word or "end" not in word:
                    continue
                start_sample = int(word["start"] * sample_rate)
                end_sample = int(word["end"] * sample_rate)
                audio_segment = self.audio[start_sample:end_sample]
                if len(audio_segment) == 0:
                    word["type"] = -1
                    continue
                rms = compute_rms(audio_segment)
                if rms < 0.02:
                    word["type"] = 0  # 속삭임
                elif rms > 0.07:
                    word["type"] = 2  # 고함
                else:
                    word["type"] = 1  # 일반
        return segments

    def format_timestamp(self, seconds):
        h, m = divmod(int(seconds), 3600)
        m, s = divmod(m, 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def segments_to_srt(self, segments):
        with open(self.output_path, "w", encoding="utf-8") as f:
            index = 1
            for segment in segments:
                words = segment.get("words", [])
                prev_end_time = 0
                for i, word_info in enumerate(words):
                    if "start" not in word_info or "end" not in word_info:
                        continue
                    start = word_info["start"]
                    end = word_info["end"]

                    speaker = word_info.get("speaker", "Unknown")

                    if prev_end_time != 0:
                        start = prev_end_time
                    prev_end_time = end

                    highlighted_sentence = f"<font size={self.default_font_size}px> [{speaker}]\n"
                    for j, w in enumerate(words):
                        word_text = w["word"]
                        word_type = w.get("type", 1)
                        if j == i:
                            if word_type == 0:
                                highlighted_sentence += f'<font color={self.highlight_color} size={self.whispering_font_size}px>{word_text}</font> '
                            elif word_type == 2:
                                highlighted_sentence += f'<font color={self.highlight_color} size={self.shouting_font_size}px>{word_text}</font> '
                            else:
                                highlighted_sentence += f'<font color={self.highlight_color}>{word_text}</font> '
                        else:
                            if word_type == 0:
                                highlighted_sentence += f'<font size={self.whispering_font_size}px>{word_text}</font> '
                            elif word_type == 2:
                                highlighted_sentence += f'<font size={self.shouting_font_size}px>{word_text}</font> '
                            else:
                                highlighted_sentence += word_text + " "
                    highlighted_sentence += "</font>"

                    f.write(f"{index}\n")
                    f.write(f"{self.format_timestamp(start)} --> {self.format_timestamp(end)}\n")
                    f.write(f"{highlighted_sentence.strip()}\n\n")
                    index += 1
        print(f"SRT 파일이 저장되었습니다: {self.output_path}")

    def generate_subtitles(self):
        model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)
        result = model.transcribe(self.audio, batch_size=self.batch_size)

        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, self.audio, self.device, return_char_alignments=False)

        diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.0", use_auth_token=self.auth_token, device=self.device)
        diarize_segments = diarize_model(self.audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        segments = self.split_segment_by_max_words(result["segments"])
        segments = self.analyze_voice_type(segments)
        self.segments_to_srt(segments)

        with open(self.output_path, 'r', encoding='utf-8') as f:
            return f.read()
