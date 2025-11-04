import torch
import numpy as np
import json

class SRTSubtitleGenerator:
    def __init__(self, subtitle_settings: json, max_words=10):  # 자막 세그먼트 당 최대 단어 개수 (최대 10개 단어 단위로 나누어 자막 생성)
        self.max_words = max_words
        self.default_font_size = subtitle_settings['font']['default_size']
        self.min_font_size = subtitle_settings['font']['min_size']
        self.max_font_size = subtitle_settings['font']['max_size']
        self.emotion_colors = subtitle_settings['hex_colors']['emotion_colors']
        self.default_color = subtitle_settings['hex_colors']['default_color']
        self.highlight_color = subtitle_settings['hex_colors']['highlight_color']

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

    def segments_to_srt(self, segments, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
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


                    """
                    프론트엔드에 감정인식/화자분리 결과 적용할 건지 여부의 설정에 대한 인터페이스가 추가될 시 아래와 같이 적용

                    highlighted_sentence = ""
                    if apply_emotion == True:
                        emotion = segment.get('emotion', 'neutral')
                        emotion_color = self.emotion_colors.get(emotion, self.default_color)
                        highlighted_sentence += f"<font color={emotion_color} size={self.default_font_size}px>"
                    else:
                        highlighted_sentence += f"<font color={self.default_color} size={self.default_font_size}px>"
                    
                    if apply_diarization == True:
                        highlighted_sentence += f"[{speaker}]\n"
                    """


                    emotion = segment.get('emotion', 'neutral')
                    emotion_color = self.emotion_colors.get(emotion, self.default_color)
                    highlighted_sentence = f"<font color={emotion_color} size={self.default_font_size}px>[{speaker}]\n"
                    for j, w in enumerate(words):
                        word_text = w["word"]
                        word_type = w.get("type", 1)
                        if j == i:
                            if word_type == 0:
                                highlighted_sentence += f'<font color={self.highlight_color} size={self.min_font_size}px>{word_text}</font> '
                            elif word_type == 2:
                                highlighted_sentence += f'<font color={self.highlight_color} size={self.max_font_size}px>{word_text}</font> '
                            else:
                                highlighted_sentence += f'<font color={self.highlight_color}>{word_text}</font> '
                        else:
                            if word_type == 0:
                                highlighted_sentence += f'<font size={self.min_font_size}px>{word_text}</font> '
                            elif word_type == 2:
                                highlighted_sentence += f'<font size={self.max_font_size}px>{word_text}</font> '
                            else:
                                highlighted_sentence += word_text + " "
                    highlighted_sentence += "</font>"

                    f.write(f"{index}\n")
                    f.write(f"{self.format_timestamp(start)} --> {self.format_timestamp(end)}\n")
                    f.write(f"{highlighted_sentence.strip()}\n\n")
                    index += 1
        print(f"SRT 파일이 저장되었습니다: {output_path}")