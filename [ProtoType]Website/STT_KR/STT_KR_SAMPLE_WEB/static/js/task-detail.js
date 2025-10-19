// 작업 상세 페이지 스크립트

// DOM 요소
const taskIdElement = document.getElementById("task-id")
const fileNameElement = document.getElementById("file-name")
const createdTimeElement = document.getElementById("created-time")
const taskStatusElement = document.getElementById("task-status")
const captionTextarea = document.getElementById("caption-textarea")
const downloadBtn = document.getElementById("download-btn")
const formatSelect = document.getElementById("format-select")
const copyBtn = document.getElementById("copy-btn")
const loading = document.getElementById("loading")
const errorState = document.getElementById("error-state")
const errorMessage = document.getElementById("error-message")
const retryBtn = document.getElementById("retry-btn")
const captionContent = document.getElementById("caption-content")

// ✅ URL에서 task ID 추출 (예: /STT/task/15/)
const pathParts = window.location.pathname.split("/");
const taskIndex = pathParts.indexOf("task");
const taskId = taskIndex !== -1 ? pathParts[taskIndex + 1] : null;

// 페이지 로드 시 작업 상세 정보 불러오기
document.addEventListener("DOMContentLoaded", async () => {
  if (taskId) {
    await loadTaskDetail(taskId)
  } else {
    showError("작업 ID를 찾을 수 없습니다.")
  }
})

// 작업 상세 정보 불러오기
async function loadTaskDetail(id) {
  showLoading()

  try {
    //modified  실제 API 엔드포인트로 교체 필요!!!!!!!!!
    // 수정 예시: /STT/task/<id>/data/
    const response = await fetch(`/STT/task/${id}/data/`);

    if (!response.ok) {
      throw new Error("작업 정보를 불러오는데 실패했습니다.")
    }

    const task = await response.json()
    displayTaskDetail(task)
  } catch (error) {
    console.error("작업 상세 정보 불러오기 실패:", error)
    showError(error.message)
  }
}

// 작업 상세 정보 표시
function displayTaskDetail(task) {
  hideLoading()
  hideError()

  // 작업 정보 표시
  taskIdElement.textContent = task.id || "-"
  fileNameElement.textContent = task.file_name || "-"
  createdTimeElement.textContent = formatDateTime(task.created_at) || "-"

  // 상태 표시
  const statusText = getStatusText(task.status)
  const statusClass = getStatusClass(task.status)
  taskStatusElement.textContent = statusText
  taskStatusElement.className = `info-value status ${statusClass}`

  // 자막 내용 표시
  if (task.caption_content) {
    captionTextarea.value = task.caption_content
    captionContent.style.display = "block"
  } else if (task.status === "processing") {
    showError("자막이 아직 생성 중입니다. 잠시 후 다시 시도해주세요.")
  } else {
    showError("자막 내용이 없습니다.")
  }
}

// 상태 텍스트 반환
function getStatusText(status) {
  const statusMap = {
    completed: "완료",
    processing: "처리중",
    failed: "실패",
  }
  return statusMap[status] || "알 수 없음"
}

// 상태 클래스 반환
function getStatusClass(status) {
  const classMap = {
    completed: "completed",
    processing: "processing",
    failed: "failed",
  }
  return classMap[status] || ""
}

// 날짜/시간 포맷팅
function formatDateTime(dateString) {
  if (!dateString) return "-"

  const date = new Date(dateString)
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  const hours = String(date.getHours()).padStart(2, "0")
  const minutes = String(date.getMinutes()).padStart(2, "0")

  return `${year}-${month}-${day} ${hours}:${minutes}`
}

// 로딩 표시
function showLoading() {
  loading.style.display = "block"
  errorState.style.display = "none"
  captionContent.style.display = "none"
}

// 로딩 숨기기
function hideLoading() {
  loading.style.display = "none"
}

// 에러 표시
function showError(message) {
  hideLoading()
  errorMessage.textContent = message
  errorState.style.display = "block"
  captionContent.style.display = "none"
}

// 에러 숨기기
function hideError() {
  errorState.style.display = "none"
}

// 다운로드 버튼 클릭 이벤트
downloadBtn.addEventListener("click", () => {
  const format = formatSelect.value
  const content = captionTextarea.value

  if (!content) {
    alert("다운로드할 자막이 없습니다.")
    return
  }

  // 파일 이름 설정
  const fileName = fileNameElement.textContent || "caption"
  const fullFileName = `${fileName}.${format}`

  // Blob 생성 및 다운로드
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = fullFileName
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
})

// 클립보드 복사 버튼 클릭 이벤트
copyBtn.addEventListener("click", async () => {
  const content = captionTextarea.value

  if (!content.trim()) {
    alert("복사할 자막이 없습니다.")
    return
  }

  try {
    await navigator.clipboard.writeText(content)

    // 버튼 텍스트 일시적으로 변경
    const originalText = copyBtn.innerHTML
    copyBtn.innerHTML = '<span class="btn-icon">✓</span> 복사됨!'
    copyBtn.style.background = "#28a745"

    setTimeout(() => {
      copyBtn.innerHTML = originalText
      copyBtn.style.background = "#6c757d"
    }, 2000)
  } catch (error) {
    console.error("클립보드 복사 실패:", error)
    alert("클립보드 복사에 실패했습니다.")
  }
})

// 재시도 버튼 클릭 이벤트
retryBtn.addEventListener("click", () => {
  loadTaskDetail(taskId)
})
