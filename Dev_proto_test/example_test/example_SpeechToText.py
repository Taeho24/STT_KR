import speech_recognition as sr

# 음성 인식기 생성
recognizer = sr.Recognizer()

def recognize_speech_from_mic():
    mic = sr.Microphone()
    with mic as source:
        print("음성을 기다리는 중...")
        recognizer.adjust_for_ambient_noise(source)  # 주변 소음 조정
        audio = recognizer.listen(source)

    try:
        print("음성을 인식 중...")
        # Google Web Speech API를 통해 음성을 텍스트로 변환
        text = recognizer.recognize_google(audio, language="ko-KR")
        return text
    except sr.UnknownValueError:
        print("음성을 인식할 수 없습니다.")
        return None
    except sr.RequestError as e:
        print(f"Google API 요청 오류: {e}")
        return None

if __name__ == "__main__":
    while True:
        text = recognize_speech_from_mic()
        if text:
            print("인식된 텍스트:", text)
