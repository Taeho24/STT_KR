#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비-HF 파이프라인(또는 비표준 카테고리 출력) 모델용 어댑터.

현재 제공:
- audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (연속값 V/A/D → 7개 범주 매핑)

참고: audeering 모델은 CC-BY-NC-SA-4.0 (비상업) 라이선스입니다. 연구/개인 용도에 적합.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


EMOTIONS7 = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AudeeringDimModel(Wav2Vec2PreTrainedModel):
    """Dimensional SER (arousal, dominance, valence) 모델 래퍼."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


@dataclass
class AudeeringDimAdapter:
    """audeering MSP-Podcast 차원형 모델 → 7개 범주 분포로 매핑.

    간단한 휴리스틱 매핑(실험적):
      - high arousal & high valence → happy
      - high arousal & low valence → (dominance>0.5 → angry, else fear)
      - low arousal & low valence → sad
      - very high arousal & mid valence → surprise
      - mid arousal & low valence & high dominance → disgust
      - else → neutral
    """

    model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    device: str = "cpu"

    def __post_init__(self):
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self.model = AudeeringDimModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _predict_dim(self, clip: np.ndarray, sr: int) -> np.ndarray:
        y = self.processor(clip, sampling_rate=sr)
        y = y["input_values"][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).float().to(self.device)
        with torch.no_grad():
            _, logits = self.model(y)
        # logits → (Arousal, Dominance, Valence)
        out = logits.detach().cpu().numpy()[0]
        return out  # shape (3,)

    def _dim_to_cat7(self, adv: np.ndarray) -> Dict[str, float]:
        # 보정: 모델 카드 예시 범위가 0..1 근처로 나오므로 그대로 사용
        a, d, v = float(adv[0]), float(adv[1]), float(adv[2])
        # 임계값
        hi, lo, mid = 0.6, 0.4, 0.5

        # 기본 one-hot + 약한 스무딩
        scores = {e: 0.0 for e in EMOTIONS7}
        chosen = "neutral"

        if a > hi and v > hi:
            chosen = "happy"
        elif a > hi and v < lo:
            chosen = "angry" if d > mid else "fear"
        elif a < lo and v < lo:
            chosen = "sad"
        elif a > 0.65 and (lo <= v <= 0.6):
            chosen = "surprise"
        elif (0.45 <= a <= 0.65) and (v < 0.35) and (d > 0.6):
            chosen = "disgust"
        else:
            chosen = "neutral"

        scores[chosen] = 0.85
        # 약간의 분산을 위해 neutral에 소량 부여(선택 감정이 neutral일 땐 happy에 부여)
        fallback = "neutral" if chosen != "neutral" else "happy"
        scores[fallback] += 0.15

        # 정규화
        s = sum(scores.values())
        if s > 0:
            for k in scores.keys():
                scores[k] /= s
        return scores

    def predict_segments(self, segments: List[Dict[str, Any]], audio: np.ndarray, sr: int) -> List[Dict[str, float]]:
        results: List[Dict[str, float]] = []
        for seg in segments:
            s = max(0, int(seg["start"] * sr))
            e = min(len(audio), int(seg["end"] * sr))
            if e <= s:
                results.append({"neutral": 1.0})
                continue
            clip = audio[s:e].astype(np.float32)
            adv = self._predict_dim(clip, sr)
            scores = self._dim_to_cat7(adv)
            results.append(scores)
        return results
