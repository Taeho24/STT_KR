import webrtcvad
import pyaudio
import time
import speech_recognition as sr

# WebRTC VAD 객체 생성
vad = webrtcvad.Vad(3)

# 음성 인식기 생성
recognizer = sr.Recognizer()

def get_audio_chunk():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=320)
    return stream.read(320)

def save_subtitle_to_srt(text):
    timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    srt_filename = "output.srt"
    
    with open(srt_filename, "a", encoding="utf-8") as file:
        file.write(f"1\n{timestamp},000 --> {timestamp},500\n{text}\n\n")

def recognize_speech_from_mic():
    mic = sr.Microphone()
    with mic as source:
        print("음성을 기다리는 중...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("음성을 인식 중...")
        text = recognizer.recognize_google(audio, language="ko-KR")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Google API 요청 오류: {e}")
        return None

def detect_speech():
    while True:
        audio_chunk = get_audio_chunk()
        is_speech = vad.is_speech(audio_chunk, 16000)
        if is_speech:
            text = recognize_speech_from_mic()
            if text:
                print("인식된 텍스트:", text)
                save_subtitle_to_srt(text)

if __name__ == "__main__":
    detect_speech()
