# 라벨 없이 모델 정확도 평가하는 방법

## 문제 정의

**현재 평가 방식**:
```python
# labelled_simpson.jsonl 필요
python tools/model_evaluator.py \
    --video assets/simpson.mp4 \
    --labels labelled_simpson.jsonl \
    --disable-text
```

**문제점**:
- 라벨링 작업 시간 소요 (1시간+)
- 새로운 영상마다 수동 라벨링 필요
- 주관적 판단 (라벨러마다 다름)

**목표**:
라벨 없이도 모델의 품질을 정량적/정성적으로 평가

---

## ✅ 방법 1: Cross-Model Consistency (모델 간 일치도)

### 원리
여러 모델이 같은 세그먼트에 대해 **일치하는 예측**을 할 경우, 높은 신뢰도로 간주

### 구현 방법

#### Step 1: 다중 모델 예측

```python
# 3개 이상의 모델로 동일한 영상 예측
models = [
    'superb/wav2vec2-large-superb-er',
    'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
    'speechbrain/emotion-recognition-wav2vec2-IEMOCAP'
]

predictions = {}
for model_name in models:
    classifier = EmotionClassifier(audio_model_name=model_name)
    predictions[model_name] = classifier.classify_batch(segments)
```

#### Step 2: 일치도 계산

```python
def calculate_consistency(predictions):
    """모델 간 예측 일치도 계산"""
    n_segments = len(predictions[list(predictions.keys())[0]])
    consistency_scores = []
    
    for i in range(n_segments):
        # 각 세그먼트에 대한 모델들의 예측
        segment_predictions = [
            predictions[model][i]['emotion'] 
            for model in predictions
        ]
        
        # 가장 많이 예측된 감정
        from collections import Counter
        most_common = Counter(segment_predictions).most_common(1)[0]
        agreement_ratio = most_common[1] / len(predictions)
        
        consistency_scores.append({
            'segment_id': i,
            'consensus_emotion': most_common[0],
            'agreement': agreement_ratio,
            'predictions': segment_predictions
        })
    
    return consistency_scores
```

#### Step 3: 품질 지표

```python
# 1. 평균 일치도
avg_consistency = np.mean([s['agreement'] for s in consistency_scores])

# 2. 고신뢰 세그먼트 비율 (80% 이상 일치)
high_confidence_ratio = sum(
    1 for s in consistency_scores if s['agreement'] >= 0.8
) / len(consistency_scores)

# 3. 감정 분포 다양성 (Shannon Entropy)
from scipy.stats import entropy
emotion_counts = Counter([s['consensus_emotion'] for s in consistency_scores])
emotion_probs = [count / len(consistency_scores) for count in emotion_counts.values()]
diversity = entropy(emotion_probs)
```

### 평가 기준

| 지표 | 좋은 모델 | 나쁜 모델 |
|------|-----------|-----------|
| 평균 일치도 | > 0.6 | < 0.4 |
| 고신뢰 비율 | > 50% | < 30% |
| 다양성 (Entropy) | 1.5~2.0 | < 1.0 (중립 편향) |

### 장점
- ✅ 라벨링 불필요
- ✅ 정량적 지표 제공
- ✅ 이상값 탐지 (일치도 낮은 세그먼트)

### 단점
- ❌ 모든 모델이 틀릴 수 있음 (집단 편향)
- ❌ 최소 3개 이상 모델 필요

---

## ✅ 방법 2: Confidence Distribution Analysis (신뢰도 분포 분석)

### 원리
좋은 모델은 **명확한 예측** (높은 confidence)과 **감정 다양성**을 보임

### 구현 방법

#### Step 1: 신뢰도 통계

```python
def analyze_confidence(predictions):
    """예측 신뢰도 분석"""
    confidences = [p['confidence'] for p in predictions]
    
    metrics = {
        'mean_confidence': np.mean(confidences),
        'median_confidence': np.median(confidences),
        'std_confidence': np.std(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        
        # 고신뢰 예측 비율 (> 0.7)
        'high_confidence_ratio': sum(1 for c in confidences if c > 0.7) / len(confidences),
        
        # 저신뢰 예측 비율 (< 0.4)
        'low_confidence_ratio': sum(1 for c in confidences if c < 0.4) / len(confidences)
    }
    
    return metrics
```

#### Step 2: 감정 분포 분석

```python
def analyze_emotion_distribution(predictions):
    """감정 예측 분포 분석"""
    from collections import Counter
    
    emotion_counts = Counter([p['emotion'] for p in predictions])
    total = len(predictions)
    
    distribution = {
        emotion: count / total 
        for emotion, count in emotion_counts.items()
    }
    
    # 중립 비율 체크
    neutral_ratio = distribution.get('neutral', 0)
    
    # 다양성 (Entropy)
    from scipy.stats import entropy
    diversity = entropy(list(distribution.values()))
    
    # Gini 계수 (불균형 측정, 0=완전 균등, 1=완전 불균등)
    sorted_probs = sorted(distribution.values())
    n = len(sorted_probs)
    gini = sum((2 * i - n - 1) * p for i, p in enumerate(sorted_probs, 1)) / (n * sum(sorted_probs))
    
    return {
        'distribution': distribution,
        'neutral_ratio': neutral_ratio,
        'diversity': diversity,
        'gini_coefficient': gini,
        'dominant_emotion': max(distribution, key=distribution.get),
        'dominant_ratio': max(distribution.values())
    }
```

### 평가 기준

#### 좋은 모델
- **평균 신뢰도**: 0.6~0.8 (너무 높으면 과신뢰)
- **고신뢰 비율**: 40~60%
- **중립 비율**: 20~40% (너무 낮으면 비현실적)
- **다양성 (Entropy)**: 1.5~2.0
- **Gini 계수**: 0.2~0.5 (적당한 불균형)

#### 나쁜 모델
- 평균 신뢰도 < 0.4 (불확실한 예측)
- 중립 비율 > 70% (중립 편향)
- 다양성 < 1.0 (한두 감정만 예측)
- Gini 계수 > 0.8 (극단적 불균형)

### 시각화

```python
import matplotlib.pyplot as plt

def visualize_model_quality(predictions):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 신뢰도 분포 히스토그램
    confidences = [p['confidence'] for p in predictions]
    axes[0, 0].hist(confidences, bins=20, edgecolor='black')
    axes[0, 0].set_title('Confidence Distribution')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Count')
    
    # 2. 감정 분포 바 차트
    emotion_dist = analyze_emotion_distribution(predictions)['distribution']
    axes[0, 1].bar(emotion_dist.keys(), emotion_dist.values())
    axes[0, 1].set_title('Emotion Distribution')
    axes[0, 1].set_ylabel('Ratio')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. 세그먼트별 신뢰도 변화
    axes[1, 0].plot(confidences)
    axes[1, 0].axhline(y=0.7, color='g', linestyle='--', label='High Conf')
    axes[1, 0].axhline(y=0.4, color='r', linestyle='--', label='Low Conf')
    axes[1, 0].set_title('Confidence over Segments')
    axes[1, 0].set_xlabel('Segment Index')
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].legend()
    
    # 4. 감정 전환 빈도 (Transition Matrix)
    emotions = [p['emotion'] for p in predictions]
    transitions = {}
    for i in range(len(emotions) - 1):
        key = (emotions[i], emotions[i+1])
        transitions[key] = transitions.get(key, 0) + 1
    
    # 전환 행렬 시각화 (간단히)
    unique_emotions = sorted(set(emotions))
    matrix = np.zeros((len(unique_emotions), len(unique_emotions)))
    for i, e1 in enumerate(unique_emotions):
        for j, e2 in enumerate(unique_emotions):
            matrix[i, j] = transitions.get((e1, e2), 0)
    
    axes[1, 1].imshow(matrix, cmap='YlOrRd')
    axes[1, 1].set_xticks(range(len(unique_emotions)))
    axes[1, 1].set_yticks(range(len(unique_emotions)))
    axes[1, 1].set_xticklabels(unique_emotions, rotation=45)
    axes[1, 1].set_yticklabels(unique_emotions)
    axes[1, 1].set_title('Emotion Transition Matrix')
    
    plt.tight_layout()
    plt.savefig('model_quality_report.png')
    plt.show()
```

---

## ✅ 방법 3: Entropy-Based Quality Score (엔트로피 기반 품질 점수)

### 원리
모델의 예측 분포가 **너무 확신적이거나 너무 불확실하지 않은 적정 수준**을 유지

### 구현

```python
def calculate_entropy_score(predictions):
    """각 예측의 엔트로피 기반 품질 점수"""
    from scipy.stats import entropy
    
    scores = []
    for pred in predictions:
        # 각 세그먼트의 감정 분포 (audio_distribution)
        dist = pred['audio_distribution']
        probs = list(dist.values())
        
        # 엔트로피 계산 (0=확실, 2.8=완전 불확실 for 7 emotions)
        ent = entropy(probs, base=2)
        
        # 이상적 엔트로피: 1.0~2.0
        # 너무 낮으면 과신뢰, 너무 높으면 불확실
        if 1.0 <= ent <= 2.0:
            quality = 1.0
        elif ent < 1.0:
            quality = ent / 1.0  # 0~1 범위로 스케일
        else:
            quality = 2.0 / ent  # 2.0 이상은 패널티
        
        scores.append({
            'segment_id': pred.get('segment_id', 0),
            'entropy': ent,
            'quality_score': quality,
            'emotion': pred['emotion'],
            'confidence': pred['confidence']
        })
    
    avg_quality = np.mean([s['quality_score'] for s in scores])
    return avg_quality, scores
```

### 평가 기준

| 평균 품질 점수 | 판단 |
|---------------|------|
| > 0.8 | 우수 (적절한 확신도) |
| 0.6~0.8 | 양호 |
| 0.4~0.6 | 보통 (개선 필요) |
| < 0.4 | 불량 (극단적 예측) |

---

## ✅ 방법 4: Perceptual Validation (지각적 검증)

### 원리
인간이 **샘플링된 결과**를 빠르게 검토하여 정성적 평가

### 구현

#### Step 1: 대표 샘플 선택

```python
def select_representative_samples(predictions, n_samples=10):
    """각 감정별 대표 샘플 선택"""
    from collections import defaultdict
    
    by_emotion = defaultdict(list)
    for i, pred in enumerate(predictions):
        by_emotion[pred['emotion']].append((i, pred))
    
    samples = []
    for emotion, preds in by_emotion.items():
        # 각 감정에서 가장 높은 신뢰도 샘플 선택
        sorted_preds = sorted(preds, key=lambda x: x[1]['confidence'], reverse=True)
        top_sample = sorted_preds[0] if sorted_preds else None
        if top_sample:
            samples.append({
                'segment_index': top_sample[0],
                'emotion': emotion,
                'confidence': top_sample[1]['confidence'],
                'start': predictions[top_sample[0]].get('start', 0),
                'end': predictions[top_sample[0]].get('end', 0)
            })
    
    return samples
```

#### Step 2: 리뷰 인터페이스

```python
def generate_review_html(video_path, samples, output_html='review.html'):
    """HTML 기반 리뷰 인터페이스 생성"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Prediction Review</title>
        <style>
            .sample { margin: 20px; padding: 10px; border: 1px solid #ccc; }
            video { width: 640px; }
        </style>
    </head>
    <body>
        <h1>모델 예측 샘플 검토</h1>
    """
    
    for sample in samples:
        html += f"""
        <div class="sample">
            <h3>감정: {sample['emotion']} (신뢰도: {sample['confidence']:.2f})</h3>
            <p>시간: {sample['start']:.1f}s - {sample['end']:.1f}s</p>
            <video controls>
                <source src="{video_path}#t={sample['start']},{sample['end']}" type="video/mp4">
            </video>
            <p>
                <label>정확함: <input type="checkbox" name="correct_{sample['segment_index']}"></label>
                <label>부정확함: <input type="checkbox" name="incorrect_{sample['segment_index']}"></label>
            </p>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 리뷰 페이지 생성: {output_html}")
    print(f"   브라우저에서 열어 10개 샘플만 검토하세요 (5분 소요)")
```

#### Step 3: 간단한 정확도 추정

```python
# 10개 샘플 중 7개 정확 → 70% 정확도 추정
estimated_accuracy = correct_count / total_samples
```

### 장점
- ✅ 빠른 검증 (10분 이내)
- ✅ 인간의 직관 활용
- ✅ 극단적 오류 탐지

### 단점
- ❌ 통계적 유의성 낮음
- ❌ 주관적

---

## ✅ 방법 5: External Benchmark Correlation (외부 벤치마크 상관관계)

### 원리
이미 검증된 **공개 데이터셋 점수**와 프로젝트 데이터 간 상관관계 추정

### 구현

#### Step 1: 공개 벤치마크 수집

```python
# 각 모델의 논문/HuggingFace 페이지에서 공개된 정확도
external_benchmarks = {
    'superb/wav2vec2-large-superb-er': {
        'IEMOCAP': 0.65,
        'RAVDESS': 0.72
    },
    'speechbrain/emotion-recognition-wav2vec2-IEMOCAP': {
        'IEMOCAP': 0.79
    }
}
```

#### Step 2: 상관 분석

```python
def estimate_performance(model_name, cross_consistency, external_benchmarks):
    """외부 벤치마크와 일치도를 결합한 성능 추정"""
    
    # 외부 벤치마크 평균
    if model_name in external_benchmarks:
        external_avg = np.mean(list(external_benchmarks[model_name].values()))
    else:
        external_avg = 0.6  # 기본값
    
    # 교차 일치도와 외부 벤치마크 가중 평균
    estimated_accuracy = 0.6 * cross_consistency + 0.4 * external_avg
    
    return estimated_accuracy
```

---

## 📊 통합 평가 프레임워크

모든 방법을 결합한 최종 평가 스크립트:

```python
class UnsupervisedEvaluator:
    """라벨 없이 모델 품질 평가"""
    
    def __init__(self, models):
        self.models = models
        self.predictions = {}
    
    def evaluate(self, video_path):
        # 1. 모든 모델로 예측
        for model_name in self.models:
            classifier = EmotionClassifier(audio_model_name=model_name)
            self.predictions[model_name] = classifier.classify_video(video_path)
        
        # 2. 교차 일치도 계산
        consistency = calculate_consistency(self.predictions)
        
        # 3. 각 모델별 품질 지표
        quality_scores = {}
        for model_name, preds in self.predictions.items():
            confidence_metrics = analyze_confidence(preds)
            distribution_metrics = analyze_emotion_distribution(preds)
            entropy_score, _ = calculate_entropy_score(preds)
            
            # 종합 점수 계산
            overall_score = (
                0.3 * consistency.get(model_name, 0.5) +
                0.3 * entropy_score +
                0.2 * (1 - distribution_metrics['neutral_ratio']) +  # 중립 패널티
                0.2 * confidence_metrics['mean_confidence']
            )
            
            quality_scores[model_name] = {
                'overall_score': overall_score,
                'consistency': consistency.get(model_name, 0),
                'entropy_quality': entropy_score,
                'neutral_ratio': distribution_metrics['neutral_ratio'],
                'mean_confidence': confidence_metrics['mean_confidence'],
                'diversity': distribution_metrics['diversity']
            }
        
        # 4. 순위 매기기
        ranked = sorted(quality_scores.items(), key=lambda x: x[1]['overall_score'], reverse=True)
        
        return ranked
```

### 사용 예시

```python
# 라벨 없이 모델 평가
evaluator = UnsupervisedEvaluator([
    'superb/wav2vec2-large-superb-er',
    'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
    'speechbrain/emotion-recognition-wav2vec2-IEMOCAP'
])

ranked_models = evaluator.evaluate('assets/simpson.mp4')

for rank, (model, scores) in enumerate(ranked_models, 1):
    print(f"{rank}. {model}")
    print(f"   Overall Score: {scores['overall_score']:.3f}")
    print(f"   Consistency: {scores['consistency']:.3f}")
    print(f"   Entropy Quality: {scores['entropy_quality']:.3f}")
    print(f"   Neutral Ratio: {scores['neutral_ratio']:.3f}")
    print()
```

---

## 📈 실제 프로젝트 적용 예시

### Simpson 데이터셋 (라벨 있음) vs 새 영상 (라벨 없음)

#### 검증 시나리오

1. **Simpson으로 supervised 평가** (Ground Truth)
```python
supervised_accuracy = evaluate_with_labels('assets/simpson.mp4', 'labelled_simpson.jsonl')
# superb: 0.645
```

2. **Simpson으로 unsupervised 평가** (라벨 숨김)
```python
unsupervised_score = evaluator.evaluate('assets/simpson.mp4')
# superb: 0.687 (overall score)
```

3. **상관관계 확인**
```python
correlation = np.corrcoef([supervised_accuracy], [unsupervised_score])[0, 1]
# 0.85 이상이면 신뢰 가능
```

4. **새 영상에 적용**
```python
# 이제 라벨 없는 새 영상도 unsupervised 방식으로 평가 가능
new_video_scores = evaluator.evaluate('assets/new_movie.mp4')
```

---

## ✅ 추천 전략

### 실전 프로토콜

1. **Simpson 데이터셋 (라벨 있음)**:
   - Supervised 평가로 절대 정확도 측정
   - 상위 3개 모델 선정

2. **새 영상 10개 (라벨 없음)**:
   - Unsupervised 평가로 일관성 확인
   - Simpson과 동일한 순위 유지하는지 검증

3. **최종 모델 선택**:
   - 두 평가에서 모두 상위권 모델 선택
   - 신뢰도 95% 이상

### 최소 작업량

- **라벨링**: Simpson만 (31개 세그먼트, 이미 완료)
- **Unsupervised 평가**: 자동 (라벨 불필요)
- **최종 검증**: 10개 샘플만 수동 확인 (10분)

---

## 요약

| 방법 | 라벨 필요 | 소요 시간 | 신뢰도 | 사용 시점 |
|------|----------|----------|--------|-----------|
| **Cross-Model Consistency** | ❌ | 자동 | ⭐⭐⭐⭐ | 기본 평가 |
| **Confidence Analysis** | ❌ | 자동 | ⭐⭐⭐ | 품질 진단 |
| **Entropy Score** | ❌ | 자동 | ⭐⭐⭐⭐ | 정량 평가 |
| **Perceptual Validation** | ❌ | 10분 | ⭐⭐⭐⭐⭐ | 최종 검증 |
| **External Benchmark** | ❌ | 자동 | ⭐⭐⭐ | 보조 지표 |
| **Supervised (Ground Truth)** | ✅ | 1시간+ | ⭐⭐⭐⭐⭐ | 절대 기준 |

**권장 조합**: 
1. Simpson으로 Supervised 평가 (1회)
2. 모든 영상에 Unsupervised 평가 적용
3. 10개 샘플로 Perceptual Validation (최종 확인)

이 방식으로 **라벨링 시간 90% 절감** 가능합니다!
