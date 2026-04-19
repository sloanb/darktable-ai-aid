from __future__ import annotations

from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn


class CliProgress:
    def __init__(self, description: str = "scanning") -> None:
        self._description = description
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def __enter__(self) -> "CliProgress":
        self._progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeRemainingColumn(),
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task(self._description, total=0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self._progress is not None
        self._progress.__exit__(exc_type, exc, tb)

    def update(self, _msg: str, current: int, total: int) -> None:
        assert self._progress is not None and self._task_id is not None
        self._progress.update(self._task_id, completed=current, total=total)
