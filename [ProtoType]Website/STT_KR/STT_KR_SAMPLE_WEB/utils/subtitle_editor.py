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

    def update_speaker_name(self, current_name:str, new_name:str) -> str:
        """
        기존 형식: [SPEAKER_00]
        
        [입력 예시] current_name = "SPEAKER_00"
        """

        # 자막 파일 불러오기
        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()

        content = content.replace(f'[{current_name}]', f'[{new_name}]')

        # 자막 파일 덮어쓰기
        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content

    def update_emotion_color(self, new_emotion_colors:dict, new_default_color:str = config.get('hex_colors', 'default_color')) -> str:
        """
        색상 코드는 #RRGGBB 형태로 입력

        [입력 예시]
        new_emotion_colors = {
            "neutral": "#FFFFFF",
            "happy": "#00FF00",
            "sad": "#0000FF",
            "angry": "#FF0000",
            "fear": "#800080",
            "surprise": "#00FFFF",
            "disgust": "#008080",
        }

        new_default_color = "#FFFFFF"
        """

        cur_hex_emotion_colors = config.get('hex_colors', 'emotion_colors')
        cur_hex_default_color = config.get('hex_colors', 'default_color')
        cur_ass_emotion_colors = config.get('ass_colors', 'emotion_colors')
        cur_ass_default_color = config.get('ass_colors', 'default_color')

        # 딕셔너리 키 구성 비교
        if set(new_emotion_colors.keys()) != set(cur_hex_emotion_colors.keys()):
            print(f"""
                [ERROR] new_emotion_colors의 입력 형식이 잘못 되었습니다.
                [입력 예시]
                {{
                    "neutral": "#FFFFFF",
                    "happy": "#00FF00",
                    "sad": "#0000FF",
                    "angry": "#FF0000",
                    "fear": "#800080",
                    "surprise": "#00FFFF",
                    "disgust": "#008080",
                }}
                [입력된 형식]
                {new_emotion_colors}
                """)
            return None
        
        # 색상 코드 형식 확인
        for _, code in new_emotion_colors.items():
            if not config.validate_color_code(code):
                print(f"""
                    [ERROR] new_emotion_colors의 입력 형식이 잘못 되었습니다.
                    [입력 예시]
                    {{
                       "neutral": "#FFFFFF",
                       "happy": "#00FF00",
                       "sad": "#0000FF",
                       "angry": "#FF0000",
                       "fear": "#800080",
                       "surprise": "#00FFFF",
                       "disgust": "#008080",
                    }}
                    [입력된 형식]
                    {new_emotion_colors}
                    """)
                return None

        new_hex_emotion_colors = new_emotion_colors.copy()
        new_hex_default_color = new_default_color
        new_ass_emotion_colors = new_emotion_colors.copy()
        new_ass_default_color = config.hex_to_ass(new_default_color)
        
        # hex color -> ass color 형식 변환(#RRGGBB -> &HBBGGRR)
        for emotion, new_emotion_color in new_emotion_colors.items():
            new_ass_emotion_colors[f'{emotion}'] = config.hex_to_ass(new_emotion_color)

        # 자막 파일 불럭오기
        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 감정 색상 변환
        if self.file_format == 'ass':
            for (_, cur_emotion_color), (_, new_emotion_color) in zip(cur_ass_emotion_colors.items(), new_ass_emotion_colors.items()):
                content = content.replace(cur_emotion_color, new_emotion_color)
            content = content.replace(cur_ass_default_color, new_ass_default_color)
        else:
            for (_, cur_emotion_color), (_, new_emotion_color) in zip(cur_hex_emotion_colors.items(), new_hex_emotion_colors.items()):
                content = content.replace(cur_emotion_color, new_emotion_color)
            content = content.replace(cur_hex_default_color, new_hex_default_color)

        # config 색상 업데이트
        config.set(new_hex_emotion_colors, 'hex_colors', 'emotion_colors')
        config.set(new_hex_default_color, 'hex_colors', 'default_color')
        config.set(new_ass_emotion_colors, 'ass_colors', 'emotion_colors')
        config.set(new_ass_default_color, 'ass_colors', 'default_color')

        # 자막 파일 덮어쓰기
        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content
    
    def update_highlight_color(self, new_highlight_color:str) -> str:
        """
        색상 코드는 #RRGGBB 형태로 입력

        [입력 예시] "#FFFF00"
        """

        cur_hex_highlight_color = config.get('hex_colors', 'highlight_color')
        cur_ass_highlight_color = config.get('ass_colors', 'highlight_color')

        new_hex_highlight_color = new_highlight_color
        new_ass_highlight_color = config.hex_to_ass(new_highlight_color)

        # 색상 코드 형식 확인
        if not config.validate_color_code(cur_hex_highlight_color):
            print(f"""
                [ERROR] new_highlight_color의 입력 형식이 잘못 되었습니다.
                [입력 예시] "#FFFF00"
                [입력된 형식] "{new_highlight_color}"
                """)
            return None

        # 자막 파일 불럭오기
        with open(self.subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 하이라이트 색상 변환
        if self.file_format == 'ass':
            content = content.replace(cur_ass_highlight_color, new_ass_highlight_color)
        else:
            content = content.replace(cur_hex_highlight_color, new_hex_highlight_color)

        # config 색상 업데이트
        config.set(new_hex_highlight_color, 'hex_colors', 'highlight_color')
        config.set(new_ass_highlight_color, 'ass_colors', 'highlight_color')

        # 자막 파일 덮어쓰기
        with open(self.subtitle_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return content
    
    def update_font_size(self, new_min_size:int, new_default_size:int, new_max_size:int) -> str:
        """폰트 크기는 px 단위"""

        if config.validate_font_size(new_min_size, new_default_size, new_max_size):
            # 기존 config 값 불러오기
            cur_min_size = config.get('font', 'min_size')
            cur_default_size = config.get('font', 'default_size')
            cur_max_size = config.get('font', 'max_size')

            # 자막 파일 불러오기
            with open(self.subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()

            content = content.replace(f'size={cur_min_size}px', f'size={new_min_size}px')\
                            .replace(f'size={cur_default_size}px', f'size={new_default_size}px')\
                            .replace(f'size={cur_max_size}px', f'size={cur_max_size}px')

            # 자막 파일 덮어쓰기
            with open(self.subtitle_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # config 값 업데이트
            config.set(new_min_size, 'font', 'min_size')
            config.set(new_default_size, 'font', 'default_size')
            config.set(new_max_size, 'font', 'max_size')
                
            return content
    
    def update_file_format(self, cur_file_format:str, new_file_format:str) -> str:
        """
        사용 가능한 파일 형식: srt, ass

        [입력 예시] "srt", "ass"
        """
        pass