"""Web GUI services — stateful managers extracted from create_app()."""

from .job_manager import JobManager
from .foraging_manager import ForagingManager

__all__ = ["JobManager", "ForagingManager"]
