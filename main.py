#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import whisperx
import torch
_old_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_torch_load(*args, **kwargs)

torch.load = _patched_load
import numpy as np
from pathlib import Path
import sys
import time
import datetime
import json

# 상대 경로 모듈 가져오기
from audio_analyzer import AudioAnalyzer
from emotion_classifier import EmotionClassifier
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
    parser.add_argument("--device", type=str, default="auto", help="계산 디바이스 (auto/cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=16, help="배치 크기")
    parser.add_argument("--compute_type", type=str, default=None, help="계산 타입")
    # 감정 모델 선택 옵션 추가
    parser.add_argument("--audio_model", type=str, default=None, help="오디오 감정 모델 식별자 (Hugging Face hub 경로). 예: xbgoose/... 또는 ehcalabres/...")
    parser.add_argument("--text_model", type=str, default=None, help="텍스트 감정 모델 식별자 (선택). 예: j-hartmann/...")
    parser.add_argument("--no_text", action="store_true", help="텍스트 감정 모델 사용 비활성화 (오디오 전용)")
    parser.add_argument("--add_to_video", action="store_true", help="자막을 영상에 합성")
    return parser.parse_args()

def ask_user_fallback_to_cpu():
    print("\nCUDA를 사용할 수 없습니다.")
    print("선택 사항:")
    print("1. CPU 모드로 계속 진행 (처리 시간 증가)")
    print("2. 프로그램 종료")
    
    while True:
        try:
            choice = input("\n선택하세요 (1 또는 2): ").strip()
            if choice == "1":
                return True
            elif choice == "2":
                return False
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력하세요.")
        except (KeyboardInterrupt, EOFError):
            print("\n프로그램을 종료합니다.")
            return False

def check_cuda_availability():
    """CUDA 사용 가능성을 체크하고 적절한 디바이스 반환"""
    if not torch.cuda.is_available():
        return "cpu", "CUDA가 사용 불가능합니다. CPU 모드를 사용합니다."
    
    try:
        # CUDA 디바이스에 간단한 테스트 수행
        test_tensor = torch.tensor([1.0]).cuda()
        test_tensor.cpu()
        return "cuda", f"CUDA 사용 가능: GPU {torch.cuda.get_device_name()}"
    except Exception as e:
        return "cpu", f"CUDA 테스트 실패: {str(e)}. CPU 모드를 사용합니다."

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

# process_extra_video: 사용처가 없어 제거되었습니다(파이프라인 단일 엔트리 유지)

def process_video(args):
    """비디오 처리 파이프라인"""
    video_path = args.video
    
    # 조기 검증: 파일 존재 여부와 토큰 확인
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"입력 비디오 파일을 찾을 수 없습니다: {video_path}\n"
                              f"현재 작업 디렉토리: {os.getcwd()}\n"
                              f"입력된 경로: {video_path}")
    
    # 디바이스 설정 개선
    if args.device.lower() == "auto":
        device, cuda_msg = check_cuda_availability()
        print(f"\n디바이스 자동 선택: {cuda_msg}")
        args.device = device
    elif args.device.lower() == "cuda":
        device, cuda_msg = check_cuda_availability()
        if device == "cpu":
            print(f"\nCUDA를 요청했지만 사용할 수 없습니다: {cuda_msg}")
            if not ask_user_fallback_to_cpu():
                print("프로그램을 종료합니다.")
                sys.exit(0)
            args.device = "cpu"
            print("CPU 모드로 진행합니다...")
    
    auth_token = read_auth_token(args.hf_token_path)
    if not auth_token:
        print("Hugging Face 토큰을 불러오는데 실패했습니다. 화자 분리를 건너뛸 수 있습니다.")
    
    # 출력 경로 설정
    video_filename = Path(video_path).stem
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    ass_output_path = os.path.join(output_dir, f"{video_filename}.ass")
    srt_output_path = os.path.join(output_dir, f"{video_filename}.srt")
    output_video_path = os.path.join(output_dir, f"{video_filename}_subbed.mp4") if args.add_to_video else None

    # 비디오 정보 추출
    video_info = get_video_info(video_path)
    print(f"비디오 정보: {video_info}")

    # compute_type 자동 선택 및 강제 설정
    if args.compute_type is None:
        args.compute_type = "float16" if args.device == "cuda" else "float32"
    
    # CPU 모드에서는 더 작은 배치 크기 사용
    if args.device == "cpu":
        # CPU 메모리와 성능에 맞게 배치 크기 조정
        original_batch_size = args.batch_size
        args.batch_size = min(args.batch_size, 4)
        if original_batch_size != args.batch_size:
            print(f"CPU 모드에서 배치 크기를 {original_batch_size}에서 {args.batch_size}로 조정했습니다.")
    
    print(f"Selected compute type: {args.compute_type}, batch size: {args.batch_size}")

    # WhisperX 모델 로드
    print("WhisperX 모델 로드 중...")
    try:
        # VAD 옵션을 단순화하여 호환성 문제 해결
        model = whisperx.load_model(
            "large-v2",
            args.device,
            compute_type=args.compute_type
        )
    except Exception as e:
        error_msg = str(e)
        if ("CUDA" in error_msg or "cuda" in error_msg.lower() or 
            "compute_type" in error_msg or "incompatible constructor" in error_msg):
            print(f"\nCUDA/compute_type 오류 발생: {error_msg}")
            
            if args.device == "cuda":
                if not ask_user_fallback_to_cpu():
                    print("프로그램을 종료합니다.")
                    sys.exit(1)
                
                print("CPU 모드로 재시도...")
                args.device = "cpu"
                args.compute_type = "float32"
                args.batch_size = min(args.batch_size, 4)
                
                try:
                    model = whisperx.load_model(
                        "large-v2",
                        args.device,
                        compute_type=args.compute_type
                    )
                    print("CPU 모드로 모델 로드 성공")
                except Exception as retry_error:
                    print(f"CPU 모드 실패: {str(retry_error)}")
                    sys.exit(1)
            else:
                raise
        else:
            print(f"WhisperX 모델 로드 실패: {str(e)}")
            raise

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
    # 정확한 싱크를 위해 word-level 정렬 사용
    result = whisperx.align(result["segments"], model_a, metadata, audio, args.device,
                          return_char_alignments=True)

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
    # 너무 짧은 노이즈만 제거 (200ms 미만)
    segments = [s for s in segments if (s["end"] - s["start"]) > 0.2]
    print(f"분할된 세그먼트 수: {len(segments)}")

    print("오디오 특성 분석 중...")
    audio_analyzer = AudioAnalyzer(sample_rate=16000)
    segments = audio_analyzer.analyze_audio_features(segments, audio)
    # Whisper/Shout 절대 감지는 감정 분석 후 통계 출력 직전에 수행 (로그 순서 요구사항)
    segments = audio_analyzer.classify_voice_types(segments, audio)

    # 감정 분류기 초기화
    print("감정 분류 모델 로드 중...")
    # 언어에 따른 텍스트 모델 자동 비활성화: 영어가 아니고 사용자가 별도 지정하지 않았으면 꺼둠
    lang = str(language_code).lower() if isinstance(language_code, str) else ""
    is_english = lang.startswith("en")
    is_korean = lang.startswith("ko")
    user_text_model = getattr(args, "text_model", None)
    no_text_flag = getattr(args, "no_text", False)
    # 한국어면 config.models.text_ko가 있으면 자동 선택, 없으면 비활성화
    text_ko_model = config.get('models', 'text_ko')
    auto_disable_text = False
    effective_text_model = None
    if no_text_flag:
        auto_disable_text = True
    elif user_text_model is not None:
        effective_text_model = user_text_model
    elif is_korean:
        effective_text_model = text_ko_model  # None일 수 있음(없으면 비활성화)
        auto_disable_text = (text_ko_model is None)
    else:
        # 영어 또는 기타: 영어면 기본 텍스트 사용, 그 외 언어면 비활성화
        if is_english:
            effective_text_model = config.get('models', 'text')
        else:
            auto_disable_text = True
            effective_text_model = None
    effective_enable_text = (False if auto_disable_text else None)
    emotion_classifier = EmotionClassifier(
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=os.path.join(args.output_dir, ".cache"),
        audio_model_name=getattr(args, "audio_model", None),
        text_model_name=effective_text_model,
        enable_text=effective_enable_text
    )
    print("감정 분류 모델 로드 완료")

    # 감정 분석 배치 처리
    print("감정 분석 중...")
    segments = emotion_classifier.classify_emotions(segments, full_audio=audio)

    # 하이퍼파라미터 튜닝을 위한 세그먼트 데이터 덤프
    annotation_dump_path = os.path.join(output_dir, f"{video_filename}_segments_for_labeling.jsonl")

    def _sanitize_scores(scores):
        if not isinstance(scores, dict):
            return {}
        return {str(k): float(v) for k, v in scores.items()}

    with open(annotation_dump_path, "w", encoding="utf-8") as dump_f:
        for segment in segments:
            record = {
                "video": video_filename,
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", 0.0)),
                "text": segment.get("text", ""),
                "speaker": segment.get("speaker", "Unknown"),
                "voice_type": segment.get("voice_type", "normal"),
                "predicted_emotion": segment.get("emotion"),
                "confidence": float(segment.get("confidence", 0.0)),
                "text_scores": _sanitize_scores(segment.get("text_scores")),
                "audio_scores": _sanitize_scores(segment.get("audio_scores")),
                "combined_scores": _sanitize_scores(segment.get("combined_scores"))
            }
            dump_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"라벨링용 세그먼트 데이터가 저장되었습니다: {annotation_dump_path}")

    # 감정 분석 통계 출력
    if segments:
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

    # 이제 Whisper/Shout 결과 로그 (감정 통계 이후)
    whisper_cnt = sum(1 for s in segments if s.get('voice_type') == 'whisper')
    shout_cnt = sum(1 for s in segments if s.get('voice_type') == 'shout')
    print(f"Whisper 감지: {whisper_cnt}개 | Shout 감지: {shout_cnt}개 (총 {len(segments)} 세그먼트)")
    if hasattr(audio_analyzer, 'voice_type_stats'):
        stats = audio_analyzer.voice_type_stats
        print("[VoiceType Stats] mean_rms={:.4f} std_rms={:.4f} whisper_abs_eff={:.4f} shout_abs_eff={:.4f} env_scale={:.2f}".format(
            stats.get('mean_rms', 0), stats.get('std_rms', 0),
            stats.get('whisper_abs_eff', 0), stats.get('shout_abs_eff', 0), stats.get('env_scale', 1.0)))

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
    
    # 시작 시간 기록
    start_time = time.time()

    print(f"입력 비디오 처리 시작: {args.video}")
    result = process_video(args)

    # 종료 시간 계산
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)

    # 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
