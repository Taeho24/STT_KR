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

    def format_timestamp(self, seconds):
        h, m = divmod(int(seconds), 3600)
        m, s = divmod(m, 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def segments_to_srt(self, segments):
        index = 1
        subtitle = ""

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

                emotion = segment.get('emotion', 'neutral')
                emotion_color = self.emotion_colors.get(emotion, self.default_color)
                highlighted_sentence = f"<font color={emotion_color} size={self.default_font_size}px>[{speaker}]\n"
                for j, w in enumerate(words):
                    word_text = w["word"]
                    word_type = w.get("voice_type", 'normal')
                    if j == i:
                        if word_type == 'whisper':
                            highlighted_sentence += f'<font color={self.highlight_color} size={self.min_font_size}px>{word_text}</font> '
                        elif word_type == 'shout':
                            highlighted_sentence += f'<font color={self.highlight_color} size={self.max_font_size}px>{word_text}</font> '
                        else:
                            highlighted_sentence += f'<font color={self.highlight_color}>{word_text}</font> '
                    else:
                        if word_type == 'whisper':
                            highlighted_sentence += f'<font size={self.min_font_size}px>{word_text}</font> '
                        elif word_type == 'shout':
                            highlighted_sentence += f'<font size={self.max_font_size}px>{word_text}</font> '
                        else:
                            highlighted_sentence += word_text + " "
                highlighted_sentence += "</font>"

                subtitle += f"{index}\n"
                subtitle += f"{self.format_timestamp(start)} --> {self.format_timestamp(end)}\n"
                subtitle += f"{highlighted_sentence.strip()}\n\n"
                index += 1

        print(f"SRT 자막이 생성되었습니다")

        return subtitle