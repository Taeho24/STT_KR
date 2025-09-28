#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def read_auth_token(file_path):
    """Hugging Face 인증 토큰 파일 읽기"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            token = f.read().strip()
        return token
    except FileNotFoundError:
        print(f"경고: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def read_gemini_api_key(file_path):
    """Gemini API key 파일 읽기"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            api_key = f.read().strip()
        return api_key
    except FileNotFoundError:
        print(f"경고: 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return None

def split_segment_by_max_words(segments, max_words):
    """최대 단어 수에 따라 세그먼트 분할"""
    new_segments = []
    
    for segment in segments:
        words = segment.get("words", [])
        
        # 단어 유효성 검사 추가
        valid_words = []
        for word in words:
            # 필수 키가 있는지 확인
            if not isinstance(word, dict):
                continue
            if 'word' not in word:
                continue
            
            # start와 end가 없는 경우 세그먼트의 시작/끝 시간 사용
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
        num_chunks = len(valid_words) // max_words + (1 if len(valid_words) % max_words > 0 else 0)
        words_per_chunk = len(valid_words) // num_chunks
        for i in range(num_chunks):
            chunk = valid_words[i * words_per_chunk: (i + 1) * words_per_chunk if i != num_chunks - 1 else len(words)]
            if chunk:
                new_segment = {
                    "start": chunk[0]["start"],
                    "end": chunk[-1]["end"],
                    "text": " ".join([w["word"] for w in chunk]),
                    "words": chunk,
                    "speaker": segment.get("speaker", "Unknown"),
                }
                new_segments.append(new_segment)
    
    return new_segments

def save_segments_to_json(segments, output_path):
    """세그먼트 정보를 JSON 파일로 저장 (디버깅 및 분석용)"""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"JSON 저장 오류: {e}")
        return False
