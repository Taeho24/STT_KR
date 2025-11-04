#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
라벨 없이 모델 품질을 평가하는 도구

사용법:
    python tools/unsupervised_evaluator.py --video assets/simpson.mp4 --models MODEL1 MODEL2 MODEL3
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import entropy
import matplotlib.pyplot as plt
import sys

# 상위 디렉토리를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from emotion_classifier import EmotionClassifier
import whisperx
import torch

class UnsupervisedEvaluator:
    """라벨 없이 모델 품질 평가"""
    
    def __init__(self, models, device="auto"):
        self.models = models
        self.predictions = {}
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 Device: {self.device}")
    
    def load_segments(self, video_path):
        """영상에서 세그먼트 추출 (WhisperX)"""
        print(f"\n📹 Loading video: {video_path}")
        
        # WhisperX로 세그먼트 추출
        compute_type = "float16" if self.device == "cuda" else "float32"
        model = whisperx.load_model("large-v2", self.device, compute_type=compute_type)
        
        audio = whisperx.load_audio(video_path)
        result = model.transcribe(audio, batch_size=16)
        
        segments = result['segments']
        print(f"✅ Extracted {len(segments)} segments")
        
        return segments
    
    def predict_all_models(self, segments, video_path):
        """모든 모델로 예측"""
        print(f"\n🔄 Running predictions with {len(self.models)} models...")
        
        for model_name in self.models:
            print(f"\n  Model: {model_name}")
            try:
                classifier = EmotionClassifier(
                    audio_model_name=model_name,
                    device=self.device,
                    enable_text=False
                )
                
                # 감정 분류 (배치 처리)
                import librosa
                audio_path = video_path.replace('.mp4', '_temp_audio.wav')
                
                # 임시 오디오 추출 (이미 있으면 재사용)
                if not Path(audio_path).exists():
                    audio = whisperx.load_audio(video_path)
                    import soundfile as sf
                    sf.write(audio_path, audio, 16000)
                
                # 배치 처리
                full_audio = librosa.load(audio_path, sr=16000)[0]
                results = classifier.process_batch(segments, full_audio)
                
                # 결과 변환
                predictions = []
                for i, (seg, result) in enumerate(zip(segments, results)):
                    predictions.append({
                        'segment_id': i,
                        'emotion': result.emotion,
                        'confidence': result.confidence,
                        'audio_score': result.audio_score,
                        'audio_distribution': result.audio_distribution,
                        'start': seg.get('start', 0),
                        'end': seg.get('end', 0),
                        'text': seg.get('text', '')
                    })
                
                self.predictions[model_name] = predictions
                print(f"    ✅ Completed: {len(predictions)} predictions")
                
            except Exception as e:
                print(f"    ❌ Model failed: {str(e)}")
                self.predictions[model_name] = []
    
    def calculate_consistency(self):
        """모델 간 예측 일치도 계산"""
        if not self.predictions:
            return {}
        
        n_segments = len(list(self.predictions.values())[0])
        consistency_scores = {}
        
        for model_name in self.predictions:
            agreements = []
            
            for i in range(n_segments):
                # 이 세그먼트에 대한 모든 모델의 예측
                segment_predictions = []
                for m in self.predictions:
                    if i < len(self.predictions[m]):
                        segment_predictions.append(self.predictions[m][i]['emotion'])
                
                if not segment_predictions:
                    continue
                
                # 가장 많은 예측과 일치하는지 확인
                most_common = Counter(segment_predictions).most_common(1)[0][0]
                current_pred = self.predictions[model_name][i]['emotion']
                
                if current_pred == most_common:
                    agreements.append(1)
                else:
                    agreements.append(0)
            
            consistency_scores[model_name] = np.mean(agreements) if agreements else 0
        
        return consistency_scores
    
    def analyze_confidence(self, predictions):
        """신뢰도 분석"""
        if not predictions:
            return {}
        
        confidences = [p['confidence'] for p in predictions]
        
        return {
            'mean': np.mean(confidences),
            'median': np.median(confidences),
            'std': np.std(confidences),
            'high_conf_ratio': sum(1 for c in confidences if c > 0.7) / len(confidences),
            'low_conf_ratio': sum(1 for c in confidences if c < 0.4) / len(confidences)
        }
    
    def analyze_distribution(self, predictions):
        """감정 분포 분석"""
        if not predictions:
            return {}
        
        emotion_counts = Counter([p['emotion'] for p in predictions])
        total = len(predictions)
        
        distribution = {
            emotion: count / total 
            for emotion, count in emotion_counts.items()
        }
        
        # 다양성 (Entropy)
        probs = list(distribution.values())
        diversity = entropy(probs, base=2)
        
        # Gini 계수
        sorted_probs = sorted(probs)
        n = len(sorted_probs)
        gini = sum((2 * i - n - 1) * p for i, p in enumerate(sorted_probs, 1)) / (n * sum(sorted_probs))
        
        return {
            'distribution': distribution,
            'neutral_ratio': distribution.get('neutral', 0),
            'diversity': diversity,
            'gini': gini,
            'dominant_emotion': max(distribution, key=distribution.get),
            'dominant_ratio': max(distribution.values())
        }
    
    def calculate_entropy_score(self, predictions):
        """엔트로피 기반 품질 점수"""
        if not predictions:
            return 0
        
        scores = []
        for pred in predictions:
            dist = pred['audio_distribution']
            probs = list(dist.values())
            
            ent = entropy(probs, base=2)
            
            # 이상적 엔트로피: 1.0~2.0
            if 1.0 <= ent <= 2.0:
                quality = 1.0
            elif ent < 1.0:
                quality = ent / 1.0
            else:
                quality = 2.0 / ent
            
            scores.append(quality)
        
        return np.mean(scores)
    
    def evaluate(self):
        """종합 평가"""
        consistency = self.calculate_consistency()
        
        quality_scores = {}
        for model_name, preds in self.predictions.items():
            if not preds:
                continue
            
            confidence_metrics = self.analyze_confidence(preds)
            distribution_metrics = self.analyze_distribution(preds)
            entropy_score = self.calculate_entropy_score(preds)
            
            # 종합 점수 계산
            overall_score = (
                0.3 * consistency.get(model_name, 0.5) +
                0.3 * entropy_score +
                0.2 * (1 - distribution_metrics['neutral_ratio']) +
                0.2 * confidence_metrics['mean']
            )
            
            quality_scores[model_name] = {
                'overall_score': overall_score,
                'consistency': consistency.get(model_name, 0),
                'entropy_quality': entropy_score,
                'confidence': confidence_metrics,
                'distribution': distribution_metrics
            }
        
        # 순위 매기기
        ranked = sorted(quality_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        return ranked
    
    def print_results(self, ranked):
        """결과 출력"""
        print("\n" + "="*80)
        print("📊 UNSUPERVISED EVALUATION RESULTS (라벨 없는 평가 결과)")
        print("="*80)
        
        for rank, (model, scores) in enumerate(ranked, 1):
            print(f"\n🏆 Rank {rank}: {model}")
            print(f"   Overall Score: {scores['overall_score']:.3f}")
            print(f"   ├─ Consistency (모델 간 일치도): {scores['consistency']:.3f}")
            print(f"   ├─ Entropy Quality (예측 품질): {scores['entropy_quality']:.3f}")
            print(f"   ├─ Mean Confidence (평균 신뢰도): {scores['confidence']['mean']:.3f}")
            print(f"   └─ Neutral Ratio (중립 비율): {scores['distribution']['neutral_ratio']:.3f}")
            
            print(f"\n   📈 Emotion Distribution:")
            for emotion, ratio in sorted(scores['distribution']['distribution'].items(), 
                                         key=lambda x: x[1], reverse=True):
                bar = "█" * int(ratio * 30)
                print(f"      {emotion:10s} {bar} {ratio:.3f}")
        
        print("\n" + "="*80)
    
    def save_results(self, ranked, output_path):
        """결과 저장"""
        results = {
            'ranked_models': [
                {
                    'rank': i,
                    'model': model,
                    'scores': scores
                }
                for i, (model, scores) in enumerate(ranked, 1)
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved: {output_path}")
    
    def visualize(self, ranked, output_path):
        """시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Overall Score 비교
        models = [m for m, _ in ranked]
        scores = [s['overall_score'] for _, s in ranked]
        
        axes[0, 0].barh(models, scores, color='steelblue')
        axes[0, 0].set_xlabel('Overall Score')
        axes[0, 0].set_title('Model Quality Ranking')
        axes[0, 0].set_xlim(0, 1)
        
        # 2. 세부 지표 비교
        consistency_scores = [s['consistency'] for _, s in ranked]
        entropy_scores = [s['entropy_quality'] for _, s in ranked]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, consistency_scores, width, label='Consistency', color='orange')
        axes[0, 1].bar(x + width/2, entropy_scores, width, label='Entropy Quality', color='green')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Detailed Metrics')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([m.split('/')[-1][:15] for m in models], rotation=45, ha='right')
        axes[0, 1].legend()
        
        # 3. 신뢰도 분포
        mean_confs = [s['confidence']['mean'] for _, s in ranked]
        high_confs = [s['confidence']['high_conf_ratio'] for _, s in ranked]
        
        axes[1, 0].scatter(mean_confs, high_confs, s=100, alpha=0.6, c=range(len(models)), cmap='viridis')
        for i, model in enumerate(models):
            axes[1, 0].annotate(i+1, (mean_confs[i], high_confs[i]), 
                               fontsize=12, ha='center', va='center', color='white', weight='bold')
        axes[1, 0].set_xlabel('Mean Confidence')
        axes[1, 0].set_ylabel('High Confidence Ratio')
        axes[1, 0].set_title('Confidence Analysis')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 중립 비율 vs 다양성
        neutral_ratios = [s['distribution']['neutral_ratio'] for _, s in ranked]
        diversities = [s['distribution']['diversity'] for _, s in ranked]
        
        axes[1, 1].scatter(neutral_ratios, diversities, s=100, alpha=0.6, c=range(len(models)), cmap='plasma')
        for i, model in enumerate(models):
            axes[1, 1].annotate(i+1, (neutral_ratios[i], diversities[i]), 
                               fontsize=12, ha='center', va='center', color='white', weight='bold')
        axes[1, 1].set_xlabel('Neutral Ratio')
        axes[1, 1].set_ylabel('Diversity (Entropy)')
        axes[1, 1].set_title('Distribution Quality')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"📊 Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="라벨 없이 모델 품질 평가")
    parser.add_argument("--video", type=str, required=True, help="입력 영상 파일")
    parser.add_argument("--models", nargs="+", required=True, help="평가할 모델 목록")
    parser.add_argument("--device", type=str, default="auto", help="디바이스 (auto/cuda/cpu)")
    parser.add_argument("--output-dir", type=str, default="result", help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 평가 실행
    evaluator = UnsupervisedEvaluator(args.models, device=args.device)
    
    # 세그먼트 로드
    segments = evaluator.load_segments(args.video)
    
    # 모든 모델로 예측
    evaluator.predict_all_models(segments, args.video)
    
    # 평가
    ranked = evaluator.evaluate()
    
    # 결과 출력
    evaluator.print_results(ranked)
    
    # 결과 저장
    video_name = Path(args.video).stem
    results_file = output_dir / f"unsupervised_eval_{video_name}.json"
    evaluator.save_results(ranked, results_file)
    
    # 시각화
    viz_file = output_dir / f"unsupervised_eval_{video_name}.png"
    evaluator.visualize(ranked, viz_file)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
