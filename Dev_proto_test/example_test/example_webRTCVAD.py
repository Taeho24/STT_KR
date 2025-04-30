import webrtcvad
import pyaudio
import numpy as np

# VAD 객체 생성
vad = webrtcvad.Vad(3)  # 0: 가장 적은 민감도, 3: 가장 높은 민감도

# 마이크에서 실시간 음성을 받아오는 함수
def get_audio_chunk():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=320)
    return stream.read(320)  # 음성의 한 조각을 읽음

# 음성 감지 및 출력
def detect_speech():
    while True:
        audio_chunk = get_audio_chunk()
        is_speech = vad.is_speech(audio_chunk, 16000)  # 16000Hz 샘플링 주파수
        if is_speech:
            print("음성이 감지되었습니다.")
        else:
            print("음성이 없습니다.")

if __name__ == "__main__":
    detect_speech()
