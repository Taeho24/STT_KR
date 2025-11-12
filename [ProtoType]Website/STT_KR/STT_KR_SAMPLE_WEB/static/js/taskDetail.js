document.addEventListener('DOMContentLoaded', () => {
    // UI 요소 (로딩, 상태)
    const taskStatusElement = document.getElementById('task-status');
    const captionTextarea = document.getElementById('caption-textarea');
    const loadingElement = document.getElementById('loading');
    const errorStateElement = document.getElementById('error-state');
    const captionContent = document.getElementById('caption-content')

    const formatSelect = document.getElementById('format-select');

    const downloadBtn = document.getElementById('download-btn'); 
    const deleteButtons = document.querySelectorAll('.delete-btn');
    const applyBtn = document.getElementById('apply-btn'); 

    const speakerMappingSection = document.getElementById("speaker-mapping-section")
    const speakerMappingContainer = document.getElementById("speaker-mapping-container")

    // 화자 이름 매핑 UI를 동적으로 생성
    function createSpeakerMappingUI(speakerMap) {
        const speakers = Object.keys(speakerMap);
        
        if (speakers.length === 0 || !speakerMappingSection) {
            if (speakerMappingSection) speakerMappingSection.style.display = "none";
            return;
        }

        speakerMappingSection.style.display = "block";
        speakerMappingContainer.innerHTML = "";

        // 딕셔너리의 {originalName: newName} 쌍을 순회하며 입력 필드 생성
        Object.entries(speakerMap).forEach(([originalSpeaker, newName], index) => {
            const mappingItem = document.createElement("div");
            mappingItem.className = "speaker-mapping-item";

            const speakerNumber = index + 1;

            // HTML 생성 (newName을 value에 넣어 기본값으로 표시)
            mappingItem.innerHTML = `
                <label class="speaker-label" for="speaker-${originalSpeaker}">
                    <span class="speaker-original">${originalSpeaker}</span>
                    <span class="speaker-arrow">→</span>
                </label>
                <input 
                    type="text" 
                    id="speaker-${originalSpeaker}" 
                    class="speaker-input" 
                    placeholder="화자 ${speakerNumber} 이름 입력"
                    data-speaker="${originalSpeaker}"
                    value="${newName || ''}"
                >
            `;
            speakerMappingContainer.appendChild(mappingItem);
        });
        
        const applyBtn = document.getElementById("apply-btn");
        if (applyBtn) {
            applyBtn.addEventListener('click', applySpeakerMapping);
        }
    }

    // 화자 이름 변경 적용 (POST 요청)
    const applySpeakerMapping = async () => {
        const newNamesMap = {};
        const updateNamesMap = {};
        const speakerInputs = document.querySelectorAll(".speaker-input");
        
        // 사용자 입력 값 수집
        speakerInputs.forEach((input) => {
            const originalSpeaker = input.dataset.speaker;
            const newName = input.value.trim();
            if (newName) {
                newNamesMap[originalSpeaker] = newName;
                updateNamesMap[newName] = '';
            }
            else updateNamesMap[originalSpeaker] = '';
        });

        if (Object.keys(newNamesMap).length === 0) {
            alert("변경할 화자 이름을 입력해주세요.");
            return;
        }

        try {
            if(loadingElement) {
                loadingElement.style.display = 'block';
            }
            
            const response = await fetch(`/STT/update-speaker-name/${TASK_ID}/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ new_names: newNamesMap }),
            });

            if(loadingElement) {
                loadingElement.style.display = 'none';
            }

            if (response.ok) {
                speakerMappingContainer.innerHTML = "";

                // 딕셔너리의 {originalName: newName} 쌍을 순회하며 입력 필드 생성
                Object.entries(updateNamesMap).forEach(([originalName, newName], index) => {
                    const mappingItem = document.createElement("div");
                    mappingItem.className = "speaker-mapping-item";

                    const speakerNumber = index + 1;

                    // HTML 생성 (newName을 value에 넣어 기본값으로 표시)
                    mappingItem.innerHTML = `
                        <label class="speaker-label" for="speaker-${originalName}">
                            <span class="speaker-original">${originalName}</span>
                            <span class="speaker-arrow">→</span>
                        </label>
                        <input 
                            type="text" 
                            id="speaker-${originalName}" 
                            class="speaker-input" 
                            placeholder="화자 ${speakerNumber} 이름 입력"
                            data-speaker="${originalName}"
                            value="${newName || ''}"
                        >
                    `;
                    speakerMappingContainer.appendChild(mappingItem);
                });

                const newSubtitle = await response.text();
                captionTextarea.value = newSubtitle;
                alert('화자 이름 변경이 성공적으로 적용되었습니다.');
            } else {
                const errorText = await response.text();
                alert(`화자 이름 변경 실패: ${errorText}`);
            }
        } catch (error) {
            console.error('화자 이름 변경 요청 오류:', error);
            if(loadingElement) {
                loadingElement.style.display = 'none';
            }
            alert('네트워크 오류로 변경 요청에 실패했습니다.');
        }
    };

    // 상태를 조회하고 업데이트하는 폴링 함수
    const checkStatus = async () => {
        try {
            const response = await fetch(`/STT/status/${TASK_ID}/`);

            if (response.status === 200) {
                const data = await response.json();

                if (data.status === 'SUCCESS') {
                    // SUCCESS: subtitle을 받은 경우 (작업 완료)
                    const subtitle = data.subtitle;
                    const speakerMap = data.speaker_map || {};
                    
                    if (loadingElement) loadingElement.style.display = 'none';
                    if (captionTextarea) {
                        captionTextarea.value = subtitle;
                    }
                    if (taskStatusElement) {
                        taskStatusElement.textContent = 'SUCCESS';
                        taskStatusElement.className = 'info-value status status-success';
                    }
                    if (speakerMappingSection) {
                        speakerMappingSection.style.display = 'block';
                    }
                    if (captionContent) {
                        captionContent.style.display = 'block';
                    }

                    // 화자 매핑 UI 생성/표시 (성공 상태로의 전환)
                    createSpeakerMappingUI(speakerMap);
                    
                    // Polling 종료
                    return;

                } else if (contentType && contentType.includes('application/json')) {
                    // PROCESSING: JSON 상태를 받은 경우
                    const data = await response.json();
                    
                    if (data.status === 'processing' || data.current_status === 'STARTED' || data.current_status === 'PENDING') {
                        if (taskStatusElement) {
                            taskStatusElement.textContent = `처리 중 (${data.current_status})...`;
                            taskStatusElement.className = 'info-value status status-started'; 
                        }
                        // Polling 계속
                        setTimeout(checkStatus, 3000); 
                    } else {
                        // 작업이 JSON으로 완료되었으나 예상치 못한 상태 (예: SUCCESS/FAILURE JSON)
                        // 안전을 위해 다시 폴링하여 텍스트/500 응답을 기다립니다.
                        setTimeout(checkStatus, 1000); 
                    }
                } else {
                    // 예상치 못한 응답
                    console.warn("서버로부터 예상치 못한 응답을 받았습니다.");
                    // 재시도
                    setTimeout(checkStatus, 5000);
                }

            } else if (response.status === 500) {
                // FAILURE/ERROR: 서버 에러 (500)
                if (taskStatusElement) {
                    taskStatusElement.textContent = 'FAILURE';
                    taskStatusElement.className = 'info-value status status-failure';
                }
                // 오류 상태 UI 표시 (null 체크 필수)
                if (loadingElement) loadingElement.style.display = 'none';
                if (errorStateElement) {
                    errorStateElement.style.display = 'block';
                    const errorMessageElement = document.getElementById('error-message');
                    if (errorMessageElement) {
                        // 오류 메시지 업데이트
                        errorMessageElement.textContent = `자막 생성에 실패했습니다. (서버 오류: 500)`;
                    }
                }
                // Polling 종료

            } else {
                // 기타 오류 (404 Not Found 등, 삭제된 작업일 가능성)
                console.warn(`Unexpected response status: ${response.status}. Stopping polling.`);
                if (taskStatusElement) {
                    taskStatusElement.textContent = 'N/A';
                    taskStatusElement.className = 'info-value status status-failure';
                }
                if (loadingElement) loadingElement.style.display = 'none';
                if (errorStateElement) {
                    errorStateElement.style.display = 'block';
                    const errorMessageElement = document.getElementById('error-message');
                    if (errorMessageElement) {
                        // 오류 메시지 업데이트
                        errorMessageElement.textContent = `자막 생성에 실패했습니다. (서버 오류: 500)`;
                    }
                }
                // Polling 종료
            }
        } catch (error) {
            console.error('폴링 오류:', error);
            // 5초 후 다시 시도
            setTimeout(checkStatus, 5000);
        }
    };

    const deleteTask = async () => { 
        if (!confirm(`정말로 작업 ID ${TASK_ID}를 영구적으로 삭제하시겠습니까? 이 작업은 되돌릴 수 없습니다.`)) {
            return;
        }

        try {
            // 서버에 삭제 요청 (POST 메서드 사용)
            const response = await fetch(`/STT/delete-task/${TASK_ID}/`, {
                method: 'POST',
                headers: {
                    // Django의 CSRF 보호를 위한 헤더가 필요할 수 있습니다.
                    // 이 뷰도 @csrf_exempt로 처리할 예정이므로 여기서는 생략합니다.
                },
            });

            if (response.ok) {
                alert('작업이 성공적으로 삭제되었습니다.');
                // 삭제 성공 시 작업 목록 페이지로 이동
                window.location.href = `/STT/task-list/${USER_ID}/`; 
            } else {
                const errorText = await response.text();
                alert(`작업 삭제 실패: ${errorText}`);
            }
        } catch (error) {
            console.error('삭제 요청 중 오류:', error);
            alert('네트워크 오류 또는 서버 통신 실패로 삭제할 수 없습니다.');
        }
    };

    // 다운로드 로직 (downloadTask 함수를 만들어 내부로 이동)
    const downloadTask = () => {
        const format = formatSelect.value;
        const content = captionTextarea.value;

        if (!content.trim()) {
            alert('다운로드할 자막이 없습니다.');
            return;
        }

        // 확장자 붙이기
        const filename = `${FILE_NAME}.${format}`;

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
    };

    // 이벤트 연결
    if (downloadBtn) { 
        downloadBtn.addEventListener('click', downloadTask);
    }

    deleteButtons.forEach(button => {
        button.addEventListener('click', deleteTask);
    });

     if (applyBtn) { 
        applyBtn.addEventListener('click', applySpeakerMapping);
    }

    if (INITIAL_STATUS === "success" && captionTextarea.value) {
        createSpeakerMappingUI(SPEAKER_NAMES_MAP)
    }
    
    // 초기 로딩 시작
    if(loadingElement) {
        loadingElement.style.display = 'block';
    }
    checkStatus();
});