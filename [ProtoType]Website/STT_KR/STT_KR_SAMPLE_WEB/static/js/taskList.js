document.addEventListener('DOMContentLoaded', () => {
    const tasksContainer = document.getElementById('tasks-container');

    // 전체 작업을 조회
    const UpdateTasks = async () => {
        try {
            // 💡 전체 목록을 JSON으로 가져오는 API 호출
            const response = await fetch(`/STT/list/${USER_ID}/`); 
            
            if (response.ok) {
                const data = await response.json();
                
                tasksContainer.innerHTML = ''; // 기존 목록을 지웁니다.

                if (data.tasks && data.tasks.length > 0) {
                    data.tasks.forEach(task => {
                        const isFinished = task.status === 'SUCCESS' || task.status === 'FAILURE' || task.status === 'REVOKED';
                        
                        const detailUrl = `/STT/task-detail/${task.task_id}/`; 
                        
                        const taskHTML = `
                            <a href="${detailUrl}"
                               class="task-list-item status-${task.status.toLowerCase()}"
                               data-task-id="${task.task_id}"
                               data-is-finished="${isFinished}"
                            >
                                <div class="task-item-left">
                                    <span class="task-title">작업 ID: ${task.task_id.slice(0, 8)}...</span>
                                    <span class="task-timestamp">생성 순번: #${task.timestamp_id}</span>
                                    <span class="task-filename">${task.file_name}</span>
                                </div>
                                <div class="task-item-right">
                                    <span class="task-status status-${task.status.toLowerCase()}">${task.status}</span>
                                </div>
                            </a>
                        `;
                        tasksContainer.innerHTML += taskHTML;
                    });
                }

            } else {
                console.error(`API Error: ${response.status}`);
            }

        } catch (error) {
            console.error('전체 목록 폴링 오류:', error);
        }
        
        // 3초 후 다시 폴링
        setTimeout(UpdateTasks, 3000); 
    };

    // 페이지 로드 후 폴링 시작
    UpdateTasks();
});