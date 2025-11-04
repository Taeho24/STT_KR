from django.db import models
from django.contrib.auth.models import User

# User ID와 Task ID 연결
class UserTask(models.Model):
    """
    사용자(User)가 요청한 비동기 작업(Celery Task)의 ID를 저장합니다.
    """
    # user_id: Django의 기본 User 모델을 외래 키로 참조 (IntegerField 대신 권장)
    user = models.ForeignKey(
        User, 
        on_delete=models.CASCADE, 
        verbose_name="사용자"
    )
    
    # task_id: Celery에서 발급하는 고유한 작업 ID (문자열)
    task_id = models.CharField(
        max_length=255, 
        unique=True, 
        verbose_name="Celery 작업 ID"
    )

    # status: 작업 진행 상태 ("PENDING", "SUCCESS", "FAILURE")
    status = models.CharField(max_length=50, default='PENDING', verbose_name="작업 상태")

    class Meta:
        verbose_name = "사용자 작업 목록"
        verbose_name_plural = "사용자 작업 목록"
        
    def __str__(self):
        return f"User {self.user.username} - Task ID: {self.task_id}"

# Task ID별 정보
class TaskInfo(models.Model):
    """
    특정 작업(Task ID)의 결과 데이터(자막, 설정 등)를 저장합니다.
    """
    # task_id: (CharField 대신) UserTask를 ForeignKey로 참조하여 두 테이블을 연결합니다.
    # UserTask 인스턴스가 삭제되면, 이 결과도 함께 삭제됩니다 (CASCADE).
    task = models.OneToOneField( 
        UserTask, 
        on_delete=models.CASCADE, 
        primary_key=True,
        verbose_name="Celery 작업"
    )

    file_name = models.CharField(
        max_length=255, 
        verbose_name="파일명",
        null=True,  # 파일을 찾지 못할 경우를 대비하여 null 허용
        blank=True  # 폼에서 필수가 아님
    )
    
    # srt_subtitle: 생성된 최종 srt 자막 텍스트
    srt_subtitle = models.TextField(
        verbose_name="자막 내용", 
        null=True, 
        blank=True
    )
    
    # config: 자막 생성에 사용된 설정(폰트, 색상 등)을 JSON 문자열로 저장
    config = models.JSONField(
        verbose_name="설정 데이터", 
        default=dict
    )

    segment = models.JSONField(
        verbose_name="자막 데이터", 
        default=dict
    )

    class Meta:
        verbose_name = "작업 정보"
        verbose_name_plural = "작업 정보"

    def __str__(self):
        return f"Information for Task ID: {self.task.task_id}"