"""Check emotion labels for candidate models"""
import sys
from transformers import AutoConfig

models = [
    'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim',
    'harshit345/xlsr-wav2vec-speech-emotion-recognition',
    'DunnBC22/wav2vec2-base-Speech_Emotion_Recognition',
    'superb/wav2vec2-base-superb-er',
    'xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned',
    'Aniemore/wavlm-emotion-russian-resd',
    'jungjongho/korean-wav2vec2-xlsr-emotions'
]

print("=== EMOTION LABEL CONFIGURATIONS ===\n")

for model_name in models:
    try:
        config = AutoConfig.from_pretrained(model_name, cache_dir=".cache")
        if hasattr(config, 'label2id'):
            labels = list(config.label2id.keys())
            print(f"{model_name}:")
            print(f"  Labels: {labels}")
            print(f"  Count: {len(labels)} emotions")
            print()
        else:
            print(f"{model_name}: No label2id found")
            print()
    except Exception as e:
        print(f"{model_name}: ERROR - {e}")
        print()
