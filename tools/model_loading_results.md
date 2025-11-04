# Model Loading Test Results - Phase 1

## Test Date
2025-01-XX

## Objective
Verify which models from `config.py` audio_candidates can be successfully loaded using `AutoModelForAudioClassification.from_pretrained()` before attempting evaluation.

---

## HuggingFace Models Test Results (8 total)

### ✅ Successfully Loaded (3 models)

1. **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**
   - Status: ✅ LOADS
   - Test: Batch evaluation completed
   - Issue: Outputs 100% neutral predictions (not useful for emotion classification)
   - Recommendation: EXCLUDE from final candidates

2. **facebook/wav2vec2-large-robust-ft-swbd-300h**
   - Status: ✅ LOADS
   - Size: 1.26GB
   - Warning: "Some weights were not initialized... You should probably TRAIN this model on a down-stream task"
   - Note: This is a speech recognition model, NOT emotion classification
   - Recommendation: EXCLUDE (wrong task type)

3. **jonatasgrosman/wav2vec2-large-xlsr-53-english**
   - Status: ✅ LOADS
   - Size: 1.26GB
   - Warning: "Some weights were not initialized... You should probably TRAIN this model on a down-stream task"
   - Note: This is a speech recognition model, NOT emotion classification
   - Recommendation: EXCLUDE (wrong task type)

### ✅ Previously Tested (1 model)

4. **jungjongho/korean-wav2vec2-xlsr-emotions**
   - Status: ✅ LOADS
   - Test: Batch evaluation completed
   - Performance: 0.065 accuracy on English audio
   - Issue: Trained on Korean, poor English performance
   - Recommendation: Keep for Korean-only evaluation

### ❌ Failed to Load (2 models)

5. **speechbrain/emotion-recognition-wav2vec2-IEMOCAP**
   - Status: ❌ FAILS - 404 Error
   - Error: `EntryNotFoundError: 404 Client Error... config.json`
   - Reason: Model doesn't have config.json structure needed for AutoModelForAudioClassification
   - Note: Speechbrain uses custom loading interface, not transformers
   - Recommendation: Would require custom loader implementation

6. **jonatasgrosman/wav2vec2-large-xlsr-53-korean**
   - Status: ❌ FAILS - 404 Error
   - Error: `RepositoryNotFoundError: 404 Client Error`
   - Reason: Repository doesn't exist on HuggingFace
   - Note: jonatasgrosman doesn't have Korean model, only English
   - Recommendation: REMOVE from config

### 🔄 Not Yet Tested (2 models)

7. **superb/wav2vec2-base-superb-er**
   - Status: ✅ PREVIOUSLY TESTED (batch evaluator)
   - Performance: 0.645 accuracy
   - Note: Best performing model so far
   - Recommendation: KEEP as primary candidate

8. **microsoft/wavlm-base-plus**
   - Status: ✅ PREVIOUSLY TESTED (batch evaluator)
   - Issue: Outputs LABEL_0, LABEL_1 only (not emotion labels)
   - Note: Not configured for emotion classification
   - Recommendation: EXCLUDE

---

## GitHub Models Test Results

### IliaZenkov/transformer-cnn-emotion-recognition
- Status: ❌ CLONED but NOT INTEGRATED
- Location: `test_models/transformer-cnn/`
- Issue: Custom architecture, requires custom loading code
- Recommendation: Skip unless simple integration method found

### marcogdepinto/...
- Status: ❌ REPOSITORY NOT FOUND (404)
- Recommendation: REMOVE from consideration

---

## Kaggle Models Test Results

### RAVDESS Dataset Models
- Status: ⏸️ NOT ATTEMPTED
- Reason: Kaggle API configured but no specific model downloaded
- Recommendation: Skip unless HuggingFace candidates insufficient

---

## Summary

### Loadable & Suitable for Emotion Classification: 2
1. `superb/wav2vec2-base-superb-er` (0.645 acc) - **PRIMARY CANDIDATE**
2. `jungjongho/korean-wav2vec2-xlsr-emotions` (Korean-specific)

### Loadable but WRONG TASK (ASR not Emotion): 3
- `facebook/wav2vec2-large-robust-ft-swbd-300h`
- `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (loads but broken)

### Cannot Load with AutoModelForAudioClassification: 2
- `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` (needs custom loader)
- `jonatasgrosman/wav2vec2-large-xlsr-53-korean` (doesn't exist)

### Not Emotion Models: 1
- `microsoft/wavlm-base-plus` (binary classification, not emotions)

---

## Recommended Actions

### Immediate
1. ✅ Remove non-existent models from config:
   - `jonatasgrosman/wav2vec2-large-xlsr-53-korean`
   
2. ✅ Remove wrong-task models from config:
   - `facebook/wav2vec2-large-robust-ft-swbd-300h` (ASR)
   - `jonatasgrosman/wav2vec2-large-xlsr-53-english` (ASR)
   - `microsoft/wavlm-base-plus` (not emotion model)
   - `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (broken)

3. 🔄 Search for additional emotion classification models:
   - HuggingFace task: "audio-classification" + "emotion"
   - Filter: wav2vec2, Wav2Vec2ForSequenceClassification

### Optional
- Investigate speechbrain custom loading if time permits
- Search for better Korean emotion models

### Final Candidate Pool (Current)
**Only 2 models are viable:**
1. `superb/wav2vec2-base-superb-er` - English, 0.645 accuracy
2. `jungjongho/korean-wav2vec2-xlsr-emotions` - Korean, untested on Korean audio

**This is insufficient. Need to find more emotion classification models.**
