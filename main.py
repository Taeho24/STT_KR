#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import whisperx
import torch
import numpy as np
from pathlib import Path
import sys
import time
import datetime  # 추가

# 상대 경로 모듈 가져오기
from audio_analyzer import AudioAnalyzer
from emotion_classifier import EmotionClassifier  # 새로운 감정 분류기로 변경
from subtitle_generator import generate_ass_subtitle
from utils import read_auth_token, split_segment_by_max_words, get_video_info, add_subtitle_to_video
from config import config

def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description="동영상에서 스타일 적용된 ASS 자막 생성")
    parser.add_argument("video", type=str, help="입력 동영상 파일 경로")
    parser.add_argument("--output_dir", type=str, default="result", help="결과 저장 디렉토리")
    parser.add_argument("--hf_token_path", type=str, default="private/hf_token.txt", help="Hugging Face 토큰 파일 경로")
    parser.add_argument("--max_words", type=int, default=10, help="세그먼트당 최대 단어 수")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--compute_type", type=str, default=None, help="계산 타입")
    parser.add_argument("--add_to_video", action="store_true", help="자막을 영상에 합성")
    parser.add_argument("--emotion_params", type=str, help="감정 분류기 파라미터 파일 경로")
    parser.add_argument("--load_params", type=str, help="감정 분류기 파라미터 파일 경로")
    parser.add_argument("--tune", action="store_true", help="감정 분류기 파라미터 튜닝 수행")
    parser.add_argument("--ground_truth", type=str, help="튜닝을 위한 정답 데이터 파일 경로")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    return parser.parse_args()

def generate_srt_subtitle(segments, output_path):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text']}\n\n")
    print(f"SRT 파일이 저장되었습니다: {output_path}")

def process_extra_video(video_path):
    """추가 비디오 처리 (음성 인식 및 세그먼트 분할)"""
    # compute_type 자동 선택
    compute_type = "float16" if torch.cuda.is_available() else "float32"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n추가 비디오 처리 시작: {video_path}")
    
    vad_options = {"use_vad": True}
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, vad_options=vad_options)
    
    print("오디오 추출 중...")
    audio = whisperx.load_audio(video_path)
    
    print("음성 인식(STT) 수행 중...")
    result = model.transcribe(audio, batch_size=16)
    language_code = result["language"]
    print(f"감지된 언어: {language_code}")
    
    print("음성 정렬 수행 중...")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    segments = split_segment_by_max_words(result["segments"], 10)
    segments = [s for s in segments if (s["end"] - s["start"]) > 0.7]
    print(f"분할된 세그먼트 수: {len(segments)}")
    
    audio_analyzer = AudioAnalyzer(sample_rate=16000)
    segments = audio_analyzer.analyze_audio_features(segments, audio)
    
    return segments

def process_video(args):
    video_path = args.video
    # 파일 존재 여부 확인 추가
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 비디오 파일을 찾을 수 없습니다: {video_path}\n"
                              f"현재 작업 디렉토리: {os.getcwd()}\n"
                              f"입력된 경로: {video_path}")
    
    video_filename = Path(video_path).stem
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ass_output_path = os.path.join(output_dir, f"{video_filename}.ass")
    srt_output_path = os.path.join(output_dir, f"{video_filename}.srt")
    output_video_path = os.path.join(output_dir, f"{video_filename}_subbed.mp4") if args.add_to_video else None

    auth_token = read_auth_token(args.hf_token_path)
    if not auth_token:
        print("Hugging Face 토큰을 불러오는데 실패했습니다. 화자 분리를 건너뛸 수 있습니다.")

    video_info = get_video_info(video_path)
    print(f"비디오 정보: {video_info}")

    # ✅ 미리 감정 분류 모델 로드 시작
    print("MIT/AST 및 RoBERTa 감정 분류 모델 로드 중...")
    # CPU 모드에서는 더 작은 배치 크기 사용
    if args.device == "cpu":
        args.batch_size = min(args.batch_size, 4)
    
    emotion_classifier = EmotionClassifier(
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=os.path.join(args.output_dir, ".cache")
    )
    print("감정 분류 모델 로드 완료")

    # compute_type 자동 선택
    if args.compute_type is None:
        args.compute_type = "float16" if args.device == "cuda" else "float32"
    print(f"Selected compute type: {args.compute_type}")

    # WhisperX 모델 로드 부분 수정
    print("WhisperX 모델 로드 중...")
    model = whisperx.load_model(
        "large-v2",
        args.device,
        compute_type=args.compute_type
    )

    print(f"오디오 추출 중: {video_path}")
    audio = whisperx.load_audio(video_path)

    print("음성 인식(STT) 수행 중...")
    result = model.transcribe(
        audio,
        batch_size=args.batch_size
    )

    # language_code 추출 추가
    language_code = result["language"]

    print("음성 정렬 수행 중...")
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=args.device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, args.device, return_char_alignments=False)

    print("화자 분리 수행 중...")
    try:
        if auth_token:
            # PyAnnote 파이프라인 초기화
            diarize_model = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.0",
                use_auth_token=auth_token,
                device=args.device
            )
            
            # 화자 분리 실행
            diarize_segments = diarize_model(
                audio,
                min_speakers=1,
                max_speakers=2
            )
            
            # 화자 분리 결과가 있는 경우에만 처리
            if diarize_segments is not None:
                result = whisperx.assign_word_speakers(diarize_segments, result)
                # 화자 수 확인
                speakers = set(s.get('speaker', '') for s in result['segments'])
                print(f"화자 분리 완료: {len(speakers)}명 탐지됨")
                
                # 화자 ID 정규화 (SPEAKER_1부터 시작)
                speaker_mapping = {}
                current_number = 1
                
                for segment in result['segments']:
                    if 'speaker' in segment:
                        original_id = segment['speaker']
                        if original_id not in speaker_mapping:
                            speaker_mapping[original_id] = f"SPEAKER_{current_number}"
                            current_number += 1
                        
                        # 새로운 ID로 변경
                        segment['speaker'] = speaker_mapping[original_id]
                        
                        # 단어 수준 화자 정보도 업데이트
                        for word in segment.get('words', []):
                            word['speaker'] = speaker_mapping[original_id]
                            
                print("화자 ID 변환 완료:", speaker_mapping)
            else:
                print("화자 분리 결과가 없습니다.")
        else:
            print("Hugging Face 토큰이 없어 화자 분리를 건너뜁니다.")
    
    except Exception as e:
        print(f"화자 분리 중 오류 발생: {str(e)}")
        print("화자 분리 없이 계속 진행합니다.")
        
    # 화자 분리 실패시 기본값 설정
    for segment in result["segments"]:
        if "speaker" not in segment:
            segment["speaker"] = "Unknown"
        for word in segment.get("words", []):
            if "speaker" not in word:
                word["speaker"] = segment["speaker"]

    segments = split_segment_by_max_words(result["segments"], args.max_words)
    segments = [s for s in segments if (s["end"] - s["start"]) > 0.7]
    print(f"분할된 세그먼트 수: {len(segments)}")

    print("오디오 특성 분석 중...")
    audio_analyzer = AudioAnalyzer(sample_rate=16000)
    segments = audio_analyzer.analyze_audio_features(segments, audio)

    # 감정 분류기 초기화
    print("감정 분류 모델 로드 중...")
    emotion_classifier = EmotionClassifier(
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=os.path.join(args.output_dir, ".cache")
    )
    print("감정 분류 모델 로드 완료")

    # 감정 분석 배치 처리
    print("감정 분석 중...")
    results = emotion_classifier.process_batch(segments, audio)
    
    # 결과를 segments에 반영
    for segment, result in zip(segments, results):
        segment['emotion'] = result.emotion
        segment['emotion_color'] = EmotionClassifier.get_emotion_color(result.emotion)
        segment['confidence'] = result.confidence
        segment['features'] = result.features

    # 저장된 파라미터 로드
    if args.load_params:
        emotion_classifier.load_parameters(args.load_params)
        print(f"감정 분류기 파라미터를 로드했습니다: {args.load_params}")
    
    # 자동 튜닝 수행
    if args.tune and args.ground_truth:
        print("\n=== 하이퍼파라미터 튜닝 시작 ===")
        
        # 첫 번째 데이터셋 준비
        first_dataset = {
            'segments': segments,
            'audio': audio,
            'ground_truth': emotion_classifier.parse_ground_truth(args.ground_truth)
        }
        
        validation_sets = [first_dataset]
        
        # 추가 데이터셋 처리
        if args.extra_video and args.extra_ground_truth:
            try:
                print("\n추가 검증 데이터셋 처리 중...")
                extra_segments = process_extra_video(args.extra_video)
                extra_audio = whisperx.load_audio(args.extra_video)
                
                extra_dataset = {
                    'segments': extra_segments,
                    'audio': extra_audio,
                    'ground_truth': emotion_classifier.parse_ground_truth(args.extra_ground_truth)
                }
                validation_sets.append(extra_dataset)
                print("추가 데이터셋 처리 완료\n")
                
                # 하이퍼파라미터 튜닝 수행
                score, best_params = emotion_classifier.tune_parameters(validation_sets, args.base_params)
                
                if score is not None and score >= 0.8:
                    print("\n=== 튜닝 성공 ===")
                    print(f"최종 정확도: {score:.2%}")
                    
                    # 튜닝된 파라미터 저장
                    params_path = args.save_params or os.path.join(
                        emotion_classifier.default_params_dir,
                        f"tuned_params_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
                    emotion_classifier.save_parameters(params_path)
                    print(f"최적화된 파라미터 저장됨: {params_path}")
                    
                    # 튜닝된 파라미터 적용
                    emotion_classifier.load_parameters(params_path)
                else:
                    print("\n=== 튜닝 실패 ===")
                    print("목표 정확도(80%)를 달성하지 못했습니다.")
                    if args.base_params:
                        print(f"기존 파라미터 유지: {args.base_params}")
                        emotion_classifier.load_parameters(args.base_params)
            except Exception as e:
                print(f"튜닝 중 오류 발생: {e}")
                if args.base_params:
                    emotion_classifier.load_parameters(args.base_params)
        else:
            print("튜닝을 위해서는 두 개의 데이터셋이 필요합니다.")
            if args.base_params:
                emotion_classifier.load_parameters(args.base_params)
    
    # 감정 분류기 초기화 및 파라미터 로드
    emotion_classifier = EmotionClassifier(device=args.device)
    if args.emotion_params:
        emotion_classifier.load_parameters(args.emotion_params)
        print(f"감정 분류기 파라미터를 로드했습니다: {args.emotion_params}")
    
    print("감정 분석 중...")
    segments = emotion_classifier.classify_emotions(segments, full_audio=audio)

    emotion_stats = {}
    for segment in segments:
        emotion = segment.get('emotion', 'unknown')
        emotion_stats[emotion] = emotion_stats.get(emotion, 0) + 1

    print("\n===== 감정 분석 통계 =====")
    total_segments = len(segments)
    for emotion, count in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_segments) * 100
        print(f" - {emotion}: {count}개 세그먼트 ({percentage:.1f}%)")
    print("=======================\n")

    print("SRT 자막 생성 중...")
    generate_srt_subtitle(segments, srt_output_path)

    print("ASS 자막 생성 중...")
    generate_ass_subtitle(segments, ass_output_path, video_info)

    if args.add_to_video:
        print("자막을 비디오에 합성 중...")
        add_subtitle_to_video(video_path, ass_output_path, output_video_path)
        print(f"자막이 적용된 비디오가 저장되었습니다: {output_video_path}")

    return {
        "segments": segments,
        "ass_path": ass_output_path,
        "srt_path": srt_output_path,
        "video_path": output_video_path if args.add_to_video else None
    }

def main():
    args = parse_arguments()
    
    # 사용자 설정 파일이 있으면 로드
    if args.config:
        config.load_config(args.config)
    
    # 시작 시간 기록
    start_time = time.time()

    print(f"입력 비디오 처리 시작: {args.video}")
    result = process_video(args)

    # 종료 시간 계산
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    print("\n===== 처리 완료 =====")
    print(f"입력 비디오: {args.video}")
    print(f"세그먼트 수: {len(result['segments'])}")
    print(f"SRT 자막: {result['srt_path']}")
    print(f"ASS 자막: {result['ass_path']}")
    if result['video_path']:
        print(f"합성된 비디오: {result['video_path']}")
    print(f"처리 시간: {minutes}분 {seconds}초")
    print("===================\n")

if __name__ == "__main__":
    main()