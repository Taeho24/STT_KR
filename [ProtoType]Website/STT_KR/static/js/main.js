import { FFmpeg } from '/static/@ffmpeg/ffmpeg/dist/esm/index.js'
import { fetchFile } from '/static/@ffmpeg/util/dist/esm/index.js'

// 폰트 크기 상태 관리
const fontSizes = {
min: 16,
default: 24,
max: 36
};

// 슬라이더 설정
const SLIDER_MIN = 6;
const SLIDER_MAX = 96;

// 드래그 상태
let isDragging = null;
let sliderRect = null;

// 감정 데이터 정의
const EMOTIONS = {
neutral: { name: '중립', defaultColor: '#FFFFFF' },
happy: { name: '행복', defaultColor: '#00FF00' },
sad: { name: '슬픔', defaultColor: '#0000FF' },
angry: { name: '분노', defaultColor: '#FF0000' },
fear: { name: '공포', defaultColor: '#800080' },
surprise: { name: '놀람', defaultColor: '#00FFFF' },
disgust: { name: '혐오', defaultColor: '#008080' }
// newEmotion: { name: '감정', defaultColor: '#색상코드' }
};

const DEFAULT_HIGHLIGHT_COLOR = '#FFFF00';

// DOM 요소 선택
const uploadArea = document.getElementById('upload-area');
const fileInput = document.getElementById('file-input');
const selectFileBtn = document.getElementById('select-file-btn');
const previewSection = document.getElementById('preview-section');
const videoPreview = document.getElementById('video-preview');
const previewBtn = document.getElementById('preview-btn');
const generateBtn = document.getElementById('generate-btn');
const cancelBtn = document.getElementById('cancel-btn');

const captionSection = document.getElementById('caption-section');
const loading = document.getElementById('loading');
//const captionContent = document.getElementById('caption-content');
//const captionTextarea = document.getElementById('caption-textarea');
//const downloadBtn = document.getElementById('download-btn');
//const formatSelect = document.getElementById('format-select');

// 멀티 슬라이더 요소들
const multiSlider = document.getElementById('multi-slider');
const minHandle = document.getElementById('min-handle');
const defaultHandle = document.getElementById('default-handle');
const maxHandle = document.getElementById('max-handle');

// 자막 설정 요소
const defaultFontSize = document.getElementById('default-font-size');
const minFontSize = document.getElementById('min-font-size');
const maxFontSize = document.getElementById('max-font-size');
const highlightColor = document.getElementById('highlight-color');
const emotionColorsGrid = document.getElementById('emotion-colors-grid');
const resetEmotionsBtn = document.getElementById('reset-emotions-btn');
const resethighlightBtn = document.getElementById('reset-highlight-btn');

const tasksList = document.getElementById('tasks-list');
const tasksContent = document.getElementById('tasks-content');

const ffmpeg = new FFmpeg();
await ffmpeg.load();

let audioBlob = null;

// 값을 퍼센트로 변환
function valueToPercent(value) {
return ((value - SLIDER_MIN) / (SLIDER_MAX - SLIDER_MIN)) * 100;
}

// 퍼센트를 값으로 변환
function percentToValue(percent) {
return Math.round(SLIDER_MIN + (percent / 100) * (SLIDER_MAX - SLIDER_MIN));
}

// 마우스 위치를 퍼센트로 변환
function getPercentFromMouse(e) {
if (!sliderRect) return 0;
const percent = ((e.clientX - sliderRect.left) / sliderRect.width) * 100;
return Math.max(0, Math.min(100, percent));
}

// UI 업데이트
function updateSliderUI() {
// 핸들 위치 업데이트
minHandle.style.left = valueToPercent(fontSizes.min) + '%';
defaultHandle.style.left = valueToPercent(fontSizes.default) + '%';
maxHandle.style.left = valueToPercent(fontSizes.max) + '%';

// 구간 색상 업데이트
const minRange = document.getElementById('min-range');
const defaultRange = document.getElementById('default-range');
const maxRange = document.getElementById('max-range');

minRange.style.left = '0%';
minRange.style.width = valueToPercent(fontSizes.min) + '%';

defaultRange.style.left = valueToPercent(fontSizes.min) + '%';
defaultRange.style.width = (valueToPercent(fontSizes.default) - valueToPercent(fontSizes.min)) + '%';

maxRange.style.left = valueToPercent(fontSizes.default) + '%';
maxRange.style.width = (valueToPercent(fontSizes.max) - valueToPercent(fontSizes.default)) + '%';

// 라벨 및 툴팁 업데이트
document.getElementById('min-label').textContent = `최소: ${fontSizes.min}px`;
document.getElementById('default-label').textContent = `기본: ${fontSizes.default}px`;
document.getElementById('max-label').textContent = `최대: ${fontSizes.max}px`;

document.getElementById('min-tooltip').textContent = `최소: ${fontSizes.min}px`;
document.getElementById('default-tooltip').textContent = `기본: ${fontSizes.default}px`;
document.getElementById('max-tooltip').textContent = `최대: ${fontSizes.max}px`;

// 숫자 입력 업데이트
minFontSize.value = fontSizes.min;
defaultFontSize.value = fontSizes.default;
maxFontSize.value = fontSizes.max;

// 입력 범위 업데이트
minFontSize.max = fontSizes.default - 1;
defaultFontSize.min = fontSizes.min + 1;
defaultFontSize.max = fontSizes.max - 1;
maxFontSize.min = fontSizes.default + 1;
}

// 폰트 크기 값 업데이트
function updateFontSize(type, value) {
const newValue = Math.max(SLIDER_MIN, Math.min(SLIDER_MAX, value));

switch (type) {
case 'min':
fontSizes.min = Math.min(newValue, fontSizes.default - 1);
break;
case 'default':
fontSizes.default = Math.max(fontSizes.min + 1, Math.min(newValue, fontSizes.max - 1));
break;
case 'max':
fontSizes.max = Math.max(newValue, fontSizes.default + 1);
break;
}

updateSliderUI();
}

// 드래그 시작
function handleMouseDown(e) {
e.preventDefault();
isDragging = e.target.closest('.slider-handle').dataset.type;
sliderRect = multiSlider.getBoundingClientRect();

// 드래그 중 스타일 적용
e.target.closest('.slider-handle').classList.add('dragging');
}

// 드래그 중
function handleMouseMove(e) {
if (!isDragging) return;

const percent = getPercentFromMouse(e);
const newValue = percentToValue(percent);
updateFontSize(isDragging, newValue);
}

// 드래그 종료
function handleMouseUp() {
if (isDragging) {
document.querySelector(`[data-type="${isDragging}"]`).classList.remove('dragging');
isDragging = null;
sliderRect = null;
}
}

// 숫자 입력 핸들러
function handleNumberInput(type, value) {
updateFontSize(type, parseInt(value));
}

async function extractAudio(videoFile) {
await ffmpeg.writeFile('input.mp4', await fetchFile(videoFile));
await ffmpeg.exec(['-i', 'input.mp4', '-vn', 'output.wav']);
const audioData = await ffmpeg.readFile('output.wav');
audioBlob = new Blob([audioData.buffer], { type: 'audio/wav' });
}

async function generateCaptions() {
const videoFile = fileInput.files[0];
if (!videoFile) {
alert('비디오 파일을 선택하세요.');
return;
}

try {
await extractAudio(videoFile);
} catch (err) {
console.error('오디오 추출 실패:', err);
alert('오디오 추출에 실패했습니다.');
return;
}

// 오디오 파일 추가
const formData = new FormData();
formData.append('audio', audioBlob, 'audio.wav');

// 자막 스타일 수집
formData.append('default_font_size', defaultFontSize.value);
formData.append('min_font_size', minFontSize.value);
formData.append('max_font_size', maxFontSize.value);
formData.append('highlight_color', highlightColor.value);

const emotionColors = getEmotionColors();
for (const [emotion, color] of Object.entries(emotionColors)) {
formData.append(`${emotion}`, color);
}

try {
const res = await fetch('/STT/generate-caption/', {
method: 'POST',
body: formData,
});
if (!res.ok) throw new Error('서버 오류');

const captions = await res.text();
//captionTextarea.value = captions;
tasksList.value = captions;
loading.style.display = 'none';
//captionContent.style.display = 'block';
tasksContent.style.display = 'block';
} catch (err) {
console.error('자막 생성 오류:', err);
alert('자막 생성에 실패했습니다.');
}
}

// 감정 색상 설정 UI 생성 함수
function createEmotionColorSettings() {
emotionColorsGrid.innerHTML = '';

Object.entries(EMOTIONS).forEach(([emotionKey, emotionData]) => {
const emotionItem = document.createElement('div');
emotionItem.className = 'emotion-item';

emotionItem.innerHTML = `
<label class="emotion-label" for="emotion-${emotionKey}">${emotionData.name}</label>
<input 
type="color" 
id="emotion-${emotionKey}" 
class="emotion-color-input" 
value="${emotionData.defaultColor}"
data-emotion="${emotionKey}"
>
`;

emotionColorsGrid.appendChild(emotionItem);
});
}

// 감정 색상 설정 값 수집 함수
function getEmotionColors() {
const emotionColors = {};

Object.keys(EMOTIONS).forEach(emotionKey => {
const colorInput = document.getElementById(`emotion-${emotionKey}`);
if (colorInput) {
emotionColors[emotionKey] = colorInput.value;
}
});

return emotionColors;
}

// 감정 색상을 기본값으로 리셋하는 함수
function resetEmotionColors() {
Object.entries(EMOTIONS).forEach(([emotionKey, emotionData]) => {
const colorInput = document.getElementById(`emotion-${emotionKey}`);
if (colorInput) {
colorInput.value = emotionData.defaultColor;
}
});
}

// 감정 색상 설정 UI 생성
createEmotionColorSettings();

// 감정 색상 기본값 복원 버튼 이벤트
resetEmotionsBtn.addEventListener('click', () => {
if (confirm('모든 감정 색상을 기본값으로 복원하시겠습니까?')) {
resetEmotionColors();
}
});

// 폰트 크기 유효성 검사??
function validateFontSizes() {
const defaultSize = parseInt(defaultFontSize.value);
const minSize = parseInt(minFontSize.value);
const maxSize = parseInt(maxFontSize.value);

if (minSize > defaultSize) {
alert('최소 폰트 크기는 기본 폰트 크기보다 작아야 합니다.');
minFontSize.value = defaultSize - 4;
return false;
}

if (maxSize < defaultSize) {
alert('최대 폰트 크기는 기본 폰트 크기보다 커야 합니다.');
maxFontSize.value = defaultSize + 4;
return false;
}

return true;
}

// 폰트 크기 입력 필드 변경 시 유효성 검사
defaultFontSize.addEventListener('change', validateFontSizes);
minFontSize.addEventListener('change', validateFontSizes);
maxFontSize.addEventListener('change', validateFontSizes);

// 하이라이트 기본값 복원 버튼 이벤트
resethighlightBtn.addEventListener('click', function() {
if (confirm('하이라이트 색상을 기본값으로 복원하시겠습니까?')) {
highlightColor.value = '#FFFF00';
}
});

// 미리보기 VTT 생성
function generatePreviewVTT() {
// 설정 값 가져오기
const defaultSize = defaultFontSize.value;
const minSize = minFontSize.value;
const maxSize = maxFontSize.value;
const highlight = highlightColor.value;
const emotionColors = getEmotionColors();

// VTT 내부에 포함할 STYLE 블록 생성
const styleBlock = `STYLE
::cue(.default-size) { font-size: ${defaultSize}px; }
::cue(.min-size) { font-size: ${minSize}px; }
::cue(.max-size) { font-size: ${maxSize}px; }
::cue(.highlight) { color: ${highlight}; }
::cue(.neutral) { color: ${emotionColors.neutral}; }
::cue(.happy) { color: ${emotionColors.happy}; }
::cue(.sad) { color: ${emotionColors.sad}; }
::cue(.angry) { color: ${emotionColors.angry}; }
::cue(.fear) { color: ${emotionColors.fear}; }
::cue(.surprise) { color: ${emotionColors.surprise}; }
::cue(.disgust) { color: ${emotionColors.disgust}; }
`;

// 자막 내용 생성
const cueBlock = `00:00:01.000 --> 00:00:03.000
<c.default-size.neutral>기본 폰트 크기가 적용된 자막입니다.</c>

00:00:03.000 --> 00:00:05.000
<c.min-size.neutral>최소 폰트 크기가 적용된 자막입니다.</c>

00:00:05.000 --> 00:00:07.000
<c.max-size.neutral>최대 폰트 크기가 적용된 자막입니다.</c>

00:00:07.000 --> 00:00:09.000
<c.default-size.highlight>하이라이트 색상이 적용된 자막입니다.</c>

00:00:09.000 --> 00:00:11.000
<c.default-size.happy>행복 감정 색상이 적용된 자막입니다.</c>

00:00:11.000 --> 00:00:13.000
<c.default-size.sad>슬픔 감정 색상이 적용된 자막입니다.</c>

00:00:13.000 --> 00:00:15.000
<c.default-size.angry>분노 감정 색상이 적용된 자막입니다.</c>

00:00:15.000 --> 00:00:17.000
<c.default-size.fear>공포 감정 색상이 적용된 자막입니다.</c>

00:00:17.000 --> 00:00:19.000
<c.default-size.surprise>놀람 감정 색상이 적용된 자막입니다.</c>

00:00:19.000 --> 00:00:21.000
<c.default-size.disgust>혐오 감정 색상이 적용된 자막입니다.</c>`;

// 4. WEBVTT 헤더, STYLE 블록, Cue 블록을 합쳐서 최종 VTT 콘텐츠 반환
return `WEBVTT\n${styleBlock}\n${cueBlock}`;
}

// 미리보기 버튼 클릭 시
previewBtn.addEventListener('click', () => {
const video = videoPreview;

// 이전 자막 트랙이 있다면 제거
const existingTrack = document.getElementById('subtitleTrack');
if (existingTrack) {
existingTrack.remove();
}

// 새로운 VTT 데이터와 Blob URL 생성
const vttData = generatePreviewVTT();
const vttBlob = new Blob([vttData], { type: 'text/vtt' });
const vttUrl = URL.createObjectURL(vttBlob);

// 새로운 <track> 요소를 생성하고 속성 설정
const newTrack = document.createElement('track');
newTrack.id = 'subtitleTrack';
newTrack.kind = 'subtitles';
newTrack.label = 'Korean';
newTrack.srclang = 'ko';
newTrack.src = vttUrl;

// 새로운 트랙을 video 요소에 추가
video.appendChild(newTrack);

// 새로 추가된 트랙을 활성화(showing).
newTrack.track.mode = 'showing';
});

// 자막 생성 버튼 클릭 시
generateBtn.addEventListener('click', async () => {
captionSection.style.display = 'block';
loading.style.display = 'block';
//captionContent.style.display = 'none';
tasksContent.style.display = 'none';
captionSection.scrollIntoView({ behavior: 'smooth' });

await generateCaptions();
});

// 파일 선택 버튼 클릭 시 파일 입력 창 열기
selectFileBtn.addEventListener('click', () => {
fileInput.click();
});

// 파일 입력 변경 시 처리
fileInput.addEventListener('change', handleFileSelect);

// 드래그 앤 드롭 이벤트 처리
uploadArea.addEventListener('dragover', (e) => {
e.preventDefault();
uploadArea.style.borderColor = '#4a00e0';
uploadArea.style.backgroundColor = '#f9f9f9';
});

uploadArea.addEventListener('dragleave', () => {
uploadArea.style.borderColor = '#ccc';
uploadArea.style.backgroundColor = 'transparent';
});

uploadArea.addEventListener('drop', (e) => {
e.preventDefault();
uploadArea.style.borderColor = '#ccc';
uploadArea.style.backgroundColor = 'transparent';

const files = e.dataTransfer.files;
if (files.length > 0) {
handleFile(files[0]);
}
});

// 파일 선택 처리 함수
function handleFileSelect(e) {
const files = e.target.files;
  if (files.length > 0) {
  handleFile(files[0]);
  }
}

// 파일 처리 함수
function handleFile(file) {
  // 파일 유형 검사
  if (!file.type.startsWith('video/')) {
  alert('비디오 파일만 업로드할 수 있습니다.');
  return;
}

// 파일 크기 검사 (500MB 제한)
if (file.size > 500 * 1024 * 1024) {
alert('파일 크기는 500MB를 초과할 수 없습니다.');
return;
}

// 비디오 미리보기 설정
const videoURL = URL.createObjectURL(file);
videoPreview.src = videoURL;

// 미리보기 섹션 표시
previewSection.style.display = 'block';

// 페이지 스크롤
previewSection.scrollIntoView({ behavior: 'smooth' });
}

// 취소 버튼 클릭 시
cancelBtn.addEventListener('click', () => {
// 비디오 미리보기 초기화
videoPreview.src = '';

// 섹션 숨기기
previewSection.style.display = 'none';
captionSection.style.display = 'none';

// 파일 입력 초기화
fileInput.value = '';
});

// 자막 다운로드 버튼 클릭 시
// downloadBtn.addEventListener('click', () => {
// const format = formatSelect.value;
//const content = captionTextarea.value;

// if (!content.trim()) {
// alert('다운로드할 자막이 없습니다.');
// return;
// }

// 파일 이름 설정
const filename = `captions.${format}`;

// Blob 생성
const blob = new Blob([content], { type: 'text/plain' });

// 다운로드 링크 생성
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = filename;

// 링크 클릭하여 다운로드 시작
document.body.appendChild(a);
a.click();

// 정리
document.body.removeChild(a);
URL.revokeObjectURL(url);

// 작업 항목 클릭
tasksList.addEventListener('click', (e) => {
  const taskItem = e.target.closest('.task-item'); // task 항목 class
  if (!taskItem) return;
  const taskId = taskItem.dataset.id; // <div class="task-item" data-id="{{id}}">...
  window.location.href = `/STT/task/${taskId}/`;
});

// 슬라이더 이벤트 리스너 등록
minHandle.addEventListener('mousedown', handleMouseDown);
defaultHandle.addEventListener('mousedown', handleMouseDown);
maxHandle.addEventListener('mousedown', handleMouseDown);

document.addEventListener('mousemove', handleMouseMove);
document.addEventListener('mouseup', handleMouseUp);

minFontSize.addEventListener('input', (e) => handleNumberInput('min', e.target.value));
defaultFontSize.addEventListener('input', (e) => handleNumberInput('default', e.target.value));
maxFontSize.addEventListener('input', (e) => handleNumberInput('max', e.target.value));

document.getElementById('add-word-btn').addEventListener('click', function() {
  const container = document.getElementById('custom-words-container');

  // 단어 입력 row 생성
  const wordItem = document.createElement('div');
  wordItem.className = 'word-item';

  // 입력창 생성
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'custom-word-input';
  input.placeholder = '고유명사를 입력하세요';

  // 삭제 버튼 생성
  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove-btn';
  removeBtn.textContent = '❌';

  // 삭제 이벤트 연결
  removeBtn.addEventListener('click', function() {
    container.removeChild(wordItem);
  });

  // 구조 조립
  wordItem.appendChild(input);
  wordItem.appendChild(removeBtn);
  container.appendChild(wordItem);
});
