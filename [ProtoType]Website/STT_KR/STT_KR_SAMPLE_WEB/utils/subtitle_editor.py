import os

from django.conf import settings
from .config import config

class SubtitleEditor:
    def __init__(self, file_format:str = "srt"):
        self.output_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'results')
        self.file_format = file_format
        
        if file_format == "srt":
            self.subtitle_path = os.path.join(self.output_path, f"subtitle.srt")
        elif file_format == "ass":
            self.subtitle_path = os.path.join(self.output_path, f"subtitle.ass")
        else:
            print("file_format이 적절하지 않습니다.")
            print("가능한 file_format: 'srt', 'ass'")
            print(f"현재 file_format: {file_format}")
            # 임의로 srt 파일 경로 지정
            self.subtitle_path = os.path.join(self.output_path, f"subtitle.srt")

    def update_speaker_name(self, current_name:str, new_name:str):
        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace(f'[{current_name}]', f'[{new_name}]')

        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

    # TODO: hex color 형태로 입력받기 위해 함수 내에서 hex_to_ass 변환과정 추가
    # TODO: config 값 업데이트 과정 추가
    def update_emotion_color(self, new_emotion_colors, new_default_color):
        if self.file_format == 'ass':
            cur_emotion_colors = config.get('ass_colors', 'emotion_colors')
            cur_default_color = config.get('ass_colors', 'default_color')
        else:
            cur_emotion_colors = config.get('hex_colors', 'emotion_colors')
            cur_default_color = config.get('hex_colors', 'default_color')

        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for cur_emotion_color, new_emotion_color in (cur_emotion_colors, new_emotion_colors):
            content = content.replace(cur_emotion_color, new_emotion_color)
        content = content.replace(cur_default_color, new_default_color)

        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content
    
    # TODO: hex color 형태로 입력받기 위해 함수 내에서 hex_to_ass 변환과정 추가
    # TODO: config 값 업데이트 과정 추가
    def update_highlight_color(self, new_highlight_color):
        if self.file_format == 'ass':
            cur_highlight_color = config.get('ass_colors', 'highlight_color')
        else:
            cur_highlight_color = config.get('hex_colors', 'highlight_color')

        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace(cur_highlight_color, new_highlight_color)

        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content
    
    def update_font_size(self, new_min_size:int, new_default_size:int, new_max_size:int):
        if self.validate_font_size(new_min_size, new_default_size, new_max_size):
            # 기존 config 값 불러오기
            cur_min_size = config.get('font', 'min_size')
            cur_default_size = config.get('font', 'default_size')
            cur_max_size = config.get('font', 'max_size')

            # 자막 파일 내용 수정
            with open(self.subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content = content.replace(f'size={cur_min_size}px', f'size={new_min_size}px')\
                            .replace(f'size={cur_default_size}px', f'size={new_default_size}px')\
                            .replace(f'size={cur_max_size}px', f'size={cur_max_size}px')

            with open(self.subtitle_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # config 값 업데이트
            config.set(new_min_size, 'font', 'min_size')
            config.set(new_default_size, 'font', 'default_size')
            config.set(new_max_size, 'font', 'max_size')
                
            return content
    
    def validate_font_size(self, min_size:int, default_size:int, max_size:int):
        if (min_size > default_size):
            print('최소 폰트 크기는 기본 폰트 크기보다 작아야 합니다.')
            return False
        if (max_size < default_size):
            print('최대 폰트 크기는 기본 폰트 크기보다 커야 합니다.')
            return False

        return True