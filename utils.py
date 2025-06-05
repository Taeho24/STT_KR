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
    new_segments = []
    
    for segment in segments:
        words = segment.get("words", [])
        
        # 단어 길이가 2음절 이하인 경우 처리
        for word in words:
            if len(word.get('word', '').strip()) <= 2:
                # 지속 시간 최소 0.2초 보장
                duration = word['end'] - word['start']
                if duration < 0.2:
                    word['end'] = word['start'] + 0.2
        
        # 기존 분할 로직
        if len(words) <= max_words:
            new_segments.append(segment)
            continue
        
        # 단어 수가 최대 단어 수를 초과하는 경우, 분할
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = words[i:i + max_words]
            if chunk:  # 비어 있지 않은 청크만 처리
                start = chunk[0]["start"]
                end = chunk[-1]["end"]
                
                # 화자 정보 유지
                speaker = segment.get("speaker", None)
                
                # 새로운 세그먼트 생성
                new_segment = {
                    "start": start,
                    "end": end,
                    "text": " ".join([word["word"] for word in chunk]),
                    "words": chunk
                }
                
                # 화자 정보가 있으면 추가
                if speaker:
                    new_segment["speaker"] = speaker
                
                chunks.append(new_segment)
        
        # 분할된 세그먼트들 추가
        new_segments.extend(chunks)
    
    return new_segments

def get_video_info(video_path):
    """비디오 파일의 정보 추출 (해상도, 프레임 레이트 등)"""
    try:
        # OpenCV를 사용하여 비디오 정보 읽기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {video_path}")
            return {"width": 1920, "height": 1080, "fps": 30.0}
        
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
        # 기본값 반환
        return {"width": 1920, "height": 1080, "fps": 30.0}

def ffmpeg_available():
    """FFmpeg가 시스템에 설치되어 있는지 확인"""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except FileNotFoundError:
        return False

def add_subtitle_to_video(video_path, subtitle_path, output_path):
    """FFmpeg를 사용하여 자막을 영상에 합성"""
    if not os.path.exists(subtitle_path):
        print(f"자막 파일이 존재하지 않습니다: {subtitle_path}")
        return False
    
    try:
        # 아래 명령어는 자막을 영상에 하드코딩합니다.
        subtitle_abs_path = os.path.abspath(subtitle_path).replace('\\', '/')

        command = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"ass='{subtitle_abs_path}'", 
            "-c:a", "copy",
            "-c:v", "libx264",
            "-crf", "18",
            "-y",
            str(output_path)
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

        if result.returncode != 0:
            print(f"FFmpeg 오류: {result.stderr.decode('utf-8', errors='replace')}")
            return False

        print(f"자막이 합성된 영상이 저장되었습니다: {output_path}")
        return True
    except Exception as e:
        print(f"자막 합성 중 오류 발생: {e}")
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
