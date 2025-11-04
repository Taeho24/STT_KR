#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자동 모델 후보 일괄 평가 스크립트
- 레이블 있는 평가 (Simpson)
- 레이블 없는 평가 (Cross-model consistency)
- 속도 벤치마크
"""

import argparse
import json
import time
import subprocess
from pathlib import Path
from typing import List, Dict
import pandas as pd

# config에서 후보 모델 목록 가져오기
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import config

def run_supervised_evaluation(video_path: str, labels_path: str, model: str, device: str = "auto") -> Dict:
    """레이블 기반 평가 실행"""
    cmd = [
        sys.executable,
        "tools/model_evaluator.py",
        "--video", video_path,
        "--labels", labels_path,
        "--disable-text",
        "--audio-models", model,
        "--device", device
    ]
    
    print(f"\n{'='*80}")
    print(f"🔄 Evaluating: {model}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=600,
            encoding='utf-8',
            errors='replace'
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            # 출력에서 결과 파싱
            output = result.stdout
            
            # Accuracy 추출
            accuracy = None
            macro_f1 = None
            neutral_rate = None
            
            for line in output.split('\n'):
                if 'Accuracy:' in line:
                    try:
                        accuracy = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Macro F1:' in line:
                    try:
                        macro_f1 = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'Neutral Rate:' in line or 'neutral prediction rate' in line.lower():
                    try:
                        neutral_rate = float(line.split(':')[1].strip().replace('%', ''))
                    except:
                        pass
            
            return {
                'model': model,
                'status': 'success',
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'neutral_rate': neutral_rate,
                'elapsed_time': elapsed,
                'output': output
            }
        else:
            return {
                'model': model,
                'status': 'failed',
                'error': result.stderr,
                'elapsed_time': elapsed
            }
    
    except subprocess.TimeoutExpired:
        return {
            'model': model,
            'status': 'timeout',
            'elapsed_time': 600
        }
    except Exception as e:
        return {
            'model': model,
            'status': 'error',
            'error': str(e),
            'elapsed_time': time.time() - start_time
        }

def main():
    parser = argparse.ArgumentParser(description="모델 후보 일괄 평가")
    parser.add_argument("--video", type=str, default="assets/simpson.mp4", help="평가 영상")
    parser.add_argument("--labels", type=str, default="labelled_simpson.jsonl", help="레이블 파일")
    parser.add_argument("--device", type=str, default="auto", help="디바이스")
    parser.add_argument("--output", type=str, default="result/batch_evaluation.json", help="결과 저장 경로")
    parser.add_argument("--models", nargs="+", help="평가할 모델 (기본: config의 audio_candidates)")
    
    args = parser.parse_args()
    
    # 모델 목록
    if args.models:
        models = args.models
    else:
        models = config.get('models', 'audio_candidates', [])
    
    print(f"\n📋 총 {len(models)}개 모델 평가 시작")
    print(f"   영상: {args.video}")
    print(f"   레이블: {args.labels}")
    print(f"   디바이스: {args.device}")
    
    # 평가 실행
    results = []
    for i, model in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model}")
        result = run_supervised_evaluation(args.video, args.labels, model, args.device)
        results.append(result)
        
        # 중간 결과 출력
        if result['status'] == 'success':
            print(f"   ✅ Accuracy: {result.get('accuracy', 'N/A')}")
            print(f"   ✅ Macro F1: {result.get('macro_f1', 'N/A')}")
            print(f"   ⏱️  Time: {result['elapsed_time']:.1f}s")
        else:
            print(f"   ❌ Status: {result['status']}")
    
    # 결과 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    # 순위표 생성
    print(f"\n{'='*80}")
    print("📊 평가 결과 순위")
    print(f"{'='*80}")
    
    # 성공한 모델만 필터링
    successful = [r for r in results if r['status'] == 'success' and r.get('accuracy') is not None]
    
    if not successful:
        print("⚠️  평가에 성공한 모델이 없습니다.")
        return
    
    # 순위 정렬 (정확도 우선, 속도 보조)
    successful.sort(key=lambda x: (x.get('accuracy', 0), -x.get('elapsed_time', 999)), reverse=True)
    
    print(f"\n{'Rank':<6}{'Model':<60}{'Acc':<8}{'F1':<8}{'Neutral':<10}{'Time(s)':<8}")
    print("-" * 100)
    
    for i, result in enumerate(successful, 1):
        model_short = result['model'].split('/')[-1][:55]
        acc = result.get('accuracy', 0)
        f1 = result.get('macro_f1', 0)
        neutral = result.get('neutral_rate', 0)
        elapsed = result.get('elapsed_time', 0)
        
        print(f"{i:<6}{model_short:<60}{acc:<8.3f}{f1:<8.3f}{neutral:<10.3f}{elapsed:<8.1f}")
    
    # 실패한 모델
    failed = [r for r in results if r['status'] != 'success']
    if failed:
        print(f"\n❌ 실패한 모델 ({len(failed)}개):")
        for r in failed:
            print(f"   - {r['model']}: {r['status']}")
    
    # CSV 저장
    csv_path = output_path.with_suffix('.csv')
    df = pd.DataFrame(successful)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📊 CSV 저장: {csv_path}")
    
    # 최고 모델 추천
    if successful:
        best = successful[0]
        print(f"\n🏆 최고 모델: {best['model']}")
        print(f"   정확도: {best.get('accuracy', 0):.3f}")
        print(f"   F1 Score: {best.get('macro_f1', 0):.3f}")
        print(f"   처리 시간: {best.get('elapsed_time', 0):.1f}s")
        
        # config 업데이트 제안
        print(f"\n💡 config.py 업데이트 추천:")
        print(f"   'audio': '{best['model']}'")

if __name__ == "__main__":
    main()
