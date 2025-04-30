import time

# 자막을 SRT 파일로 저장
def save_subtitle_to_srt(text):
    timestamp = time.strftime("%H:%M:%S", time.gmtime(time.time()))
    srt_filename = "output.srt"
    
    with open(srt_filename, "a", encoding="utf-8") as file:
        file.write(f"1\n{timestamp},000 --> {timestamp},500\n{text}\n\n")

if __name__ == "__main__":
    while True:
        text = recognize_speech_from_mic()
        if text:
            print("자막 생성:", text)
            save_subtitle_to_srt(text)
