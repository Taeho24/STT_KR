import whisperx
import torch
from torchaudio.functional import spectral_centroid
import numpy as np

video_path = "assets/video.mp4"    # 동영상 파일 경로
output_path = "result/subtitle.srt" # 자막 파일 경로
hf_token_path = "private/hf_token.txt"  # Hugging face 읽기 토큰 파일 경로

font_color = "yellow"    # 자막 하이라이트 색상 
default_font_size = 72  # 기본 폰트 크기
whisper_font_size = int(default_font_size * 0.8)
shouting_font_size = int(default_font_size * 1.2)
max_words = 10  # 한 문장 당 최대 단어 수

device = "cuda"  # 사용할 디바이스 설정 (cpu: cpu, cuda: GPU)
batch_size = 16  # 배치 크기 설정 (GPU 메모리 부족 시 줄이기)
compute_type = "float16"  # GPU 메모리 부족 시 "int8"으로 변경 가능

# Hugging face 읽기 토큰 불러오기
def read_auth_token(file_path=hf_token_path):
    try:
        with open(file_path, "r") as file:
            auth_token = file.read().strip()  # 파일에서 읽어온 토큰을 반환
            return auth_token
    except FileNotFoundError:
        print(f"파일 {file_path}을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        return None

# max_words를 초과하는 문장을 분할
def split_segment_by_max_words(segments, max_words):
    # 주어진 segment를 max_words 개수로 분할하여 새로 반환된 new_segments 리스트에 추가
    new_segments = []

    for segment in segments:
        # 만약 문장이 너무 길면 분할된 segments로 처리
        if len(segment.get("words", [])) > max_words:
            words = segment.get("words", [])
            num_chunks = len(words) // max_words + (1 if len(words) % max_words > 0 else 0)
            words_per_chunk = len(words) // num_chunks  # 분할된 각 문장의 단어 개수

            word_chunks = []
            for i in range(num_chunks):
                start_index = int(i * words_per_chunk)
                if i == num_chunks - 1:
                    end_index = len(words)
                else:
                    end_index = int((i + 1) * words_per_chunk)

                chunk = words[start_index:end_index]

                if chunk:  # 비어있는 청크는 제외
                    word_chunks.append(chunk)

            for chunk in word_chunks:
                if not chunk:  # 비어있는 청크는 제외
                    continue
                
                start = chunk[0]["start"]
                end = chunk[-1]["end"]

                new_segments.append({
                    "start": start,
                    "end": end,
                    "text": " ".join([word["word"] for word in chunk]),
                    "words": chunk
                })
        else:
            new_segments.append(segment)
    
    return new_segments

# 임시 코드
# 함수 수정 및 값 조정 필요
# 목소리 타입 분석하여 segments 요소에 추가하여 반환 (segments["type"]: 0 == 속삭임, 1 == 일반, 2 == 소리침)
def analyze_voice_type(segments, audio, sample_rate=16000):
    def compute_rms(audio_segment):
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)

        return torch.sqrt(torch.mean(audio_segment ** 2)).item()

    def compute_centroid(audio_segment):
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)  # (1, n)

        n_fft = 1024
        hop_length = 512
        win_length = n_fft
        pad = 0  # 패딩 제거 또는 최소로 설정

        # 길이가 너무 짧으면 처리 불가하므로 생략
        if audio_segment.size(-1) < win_length:
            return 0

        centroids = spectral_centroid(
            audio_segment,
            sample_rate=sample_rate,
            pad=pad,
            window=torch.hann_window(win_length),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )

        return centroids.mean().item() if centroids.numel() > 0 else 0

    # 단어 단위 분석
    for segment in segments:
        for word in segment.get("words", []):
            start = word["start"]
            end = word["end"]

            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            audio_segment = audio[start_sample:end_sample]

            if len(audio_segment) == 0:
                word["type"] = -1
                continue

            rms = compute_rms(audio_segment)
            centroid = compute_centroid(audio_segment)

            # 조건 기반 분류 (기준값은 실험적으로 조정 필요)
            if rms < 0.01 and centroid < 1500:
                word["type"] = 0  # 속삭임
            elif rms > 0.05 and (centroid > 2500 or centroid < 800):
                word["type"] = 2  # 소리침 (고주파든 저주파든 에너지가 크면 소리침으로 구분)
            else:
                word["type"] = 1  # 일반

    return segments

# segments를 입력받아 srt 자막 파일 생성
def segments_to_srt(segments, output_path=output_path):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_path, "w", encoding="utf-8") as f:
        index = 1
        for segment in segments:
            words = segment.get("words", [])
            prev_end_time = 0

            # 각 단어를 하이라이트
            for i, word_info in enumerate(words):
                start = word_info["start"]
                end = word_info["end"]

                if word_info["type"] == 0:  # 속삭임
                    font_size = whisper_font_size
                elif word_info["type"] == 2:    # 소리침
                    font_size = shouting_font_size
                else:   # 일반반
                    font_size = default_font_size

                # speaker 정보 불러오기
                # "SPEAKER_00" 형태
                # speaker = word_info.get("speaker", "Unknown")
                
                if prev_end_time == 0:
                    prev_end_time = end
                else:
                    start = prev_end_time
                    prev_end_time = end

                highlighted_sentence = f"<font size={font_size}px>"
                for j, w in enumerate(words):
                    if i == j:
                        # 하이라이트 적용
                        highlighted_sentence += f'<font color={font_color}>{w["word"]}</font> '
                    else:
                        # 띄어쓰기 추가
                        highlighted_sentence += w["word"] + " "
                highlighted_sentence += "</font>"

                f.write(f"{index}\n")
                f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
                f.write(f"{highlighted_sentence.strip()}\n\n")

                # speaker 정보 포함된 자막 생성
                # f.write(f"<font size={font_size}>[{speaker}]</font>\n{highlighted_sentence.strip()}\n\n")

                index += 1

    print(f"SRT 파일이 저장되었습니다: {output_path}")

# Whisper 모델 로드 (large-v2 모델 사용)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# 오디오 파일 로드
audio = whisperx.load_audio(video_path)

# Hugging face 토큰 로드
auth_token = read_auth_token(hf_token_path)

# Whisper 모델로 전사 수행 (배치 처리가 가능)
result = model.transcribe(audio, batch_size=batch_size)

# Whisper 결과 정렬 (alignment)
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# 화자 분리 수행 (Diarization)
diarize_model = whisperx.DiarizationPipeline(model_name="pyannote/speaker-diarization-3.0", use_auth_token=auth_token, device=device)

# 화자 분리 실행
diarize_segments = diarize_model(audio)

# 전사된 텍스트에 화자 ID 할당
result = whisperx.assign_word_speakers(diarize_segments, result)

# 단어 개수에 따라 세그먼트 분할
segments = split_segment_by_max_words(result["segments"], max_words)

# 목소리 타입 분석하여 type 요소로 추가하여 반환
segments = analyze_voice_type(segments, audio)

# 자막 파일 생성
segments_to_srt(segments, output_path=output_path)