#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import json
from pathlib import Path
import cv2

def read_auth_token(file_path):
    """Hugging Face 인증 토큰 파일 읽기"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
        return token
    except FileNotFoundError:
        print(f"경고: 토큰 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"토큰 파일 읽기 오류: {e}")
        return None

def split_segment_by_max_words(segments, max_words):
    """최대 단어 수에 따라 세그먼트 분할"""
    if not segments:
        return []
    
    new_segments = []
    
    for segment in segments:
        words = segment.get("words", [])
        
        # 단어 유효성 검사
        valid_words = []
        for word in words:
            if not isinstance(word, dict) or 'word' not in word:
                continue
            
            # start와 end가 없는 경우 기본값 설정
            if 'start' not in word or 'end' not in word:
                word['start'] = segment.get('start', 0)
                word['end'] = segment.get('end', segment.get('start', 0) + 0.5)
            
            # 2음절 이하 단어 처리
            if len(word.get('word', '').strip()) <= 2:
                duration = word['end'] - word['start']
                if duration < 0.2:
                    word['end'] = word['start'] + 0.2
            
            valid_words.append(word)
        
        # 유효한 단어가 없으면 원본 세그먼트 그대로 추가
        if not valid_words:
            new_segments.append(segment)
            continue
        
        # 단어 수에 따라 분할
        if len(valid_words) <= max_words:
            segment['words'] = valid_words
            new_segments.append(segment)
            continue
        
        # 단어 수가 최대 단어 수를 초과하는 경우, 분할
        for i in range(0, len(valid_words), max_words):
            chunk = valid_words[i:i + max_words]
            if chunk:  # 비어 있지 않은 청크만 처리
                new_segment = {
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "text": " ".join([w["word"] for w in chunk]),
                    "words": chunk,
                    "speaker": segment.get("speaker", "Unknown")  # 화자 정보 유지
                }
                new_segments.append(new_segment)
    
    return new_segments

def get_video_info(video_path):
    """비디오 파일의 정보 추출"""
    default_info = {"width": 1920, "height": 1080, "fps": 30.0}
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {video_path}")
            return default_info
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration
        }
    except Exception as e:
        print(f"비디오 정보 읽기 오류: {e}")
        return default_info

def ffmpeg_available():
    """FFmpeg가 시스템에 설치되어 있는지 확인"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

def add_subtitle_to_video(video_path, subtitle_path, output_path):
    """FFmpeg를 사용하여 자막을 영상에 하드코딩"""
    if not os.path.exists(subtitle_path):
        print(f"자막 파일이 존재하지 않습니다: {subtitle_path}")
        return False
    
    try:
        # 경로 처리
        video_path = str(Path(video_path).resolve())
        subtitle_path = str(Path(subtitle_path).resolve())
        output_path = str(Path(output_path).resolve())
        
        # 먼저 파일 존재 확인
        print(f"비디오 파일: {os.path.exists(video_path)}")
        print(f"자막 파일: {os.path.exists(subtitle_path)}")

        # 임시 파일 경로 생성
        temp_output = str(Path(output_path).parent / f"temp_{Path(output_path).name}")
        
        # 명령어 구성 - 여러 가지 필터 옵션 시도
        filters = [
            f"subtitles='{subtitle_path}'",  # 기본 subtitles 필터
            f"ass='{subtitle_path}'",        # ass 필터
            f"subtitles=filename='{subtitle_path}'"  # filename 지정
        ]
        
        success = False
        error_messages = []
        
        for filter_str in filters:
            try:
                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vf", filter_str,
                    "-c:v", "libx264",
                    "-preset", "medium",  # veryslow에서 medium으로 변경
                    "-crf", "23",         # 17에서 23으로 변경 (더 빠른 인코딩)
                    "-c:a", "copy",
                    "-y",
                    temp_output
                ]

                print(f"\n시도하는 필터: {filter_str}")
                print("자막 합성 시작...")
                print(f"명령어: {' '.join(command)}")
                
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding='utf-8',
                    errors='replace'
                )
                
                # 실제로 파일이 생성되었는지 확인
                if result.returncode == 0 and os.path.exists(temp_output):
                    # 임시 파일을 최종 파일로 이동
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.rename(temp_output, output_path)
                    print(f"자막이 성공적으로 합성되었습니다: {output_path}")
                    success = True
                    break
                else:
                    error_messages.append(f"필터 '{filter_str}' 실패: {result.stderr}")
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    
            except Exception as e:
                error_messages.append(f"필터 '{filter_str}' 예외 발생: {str(e)}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)

        if not success:
            print("\n모든 자막 합성 시도 실패:")
            for msg in error_messages:
                print(msg)
            return False

        return success

    except Exception as e:
        print(f"자막 합성 중 치명적 오류 발생: {e}")
        return False

def save_segments_to_json(segments, output_path):
    """세그먼트 정보를 JSON 파일로 저장 (디버깅 및 분석용)"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"JSON 저장 오류: {e}")
        return False
