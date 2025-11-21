# app/tasks/celery_app.py
import os
from celery import Celery
from urllib.parse import urlparse

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Defensive check: ensure it's a URL with a scheme
parsed = urlparse(REDIS_URL)
if not parsed.scheme or not REDIS_URL.startswith(("redis://", "rediss://")):
    raise RuntimeError(
        f"REDIS_URL appears invalid: {REDIS_URL!r}. "
        "It must be a full URL like 'redis://localhost:6379/0'. "
        "Set REDIS_URL env accordingly."
    )

# include the worker module so tasks are auto-registered when worker starts
celery_app = Celery(
    "startup_researcher",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.worker_tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_time_limit=int(os.getenv("CELERY_TASK_TIME_LIMIT", "3600")),
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "3600")),
)