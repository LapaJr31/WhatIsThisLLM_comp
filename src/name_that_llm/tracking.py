import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

_uri = os.environ.get("MLFLOW_TRACKING_URI")
_enabled = False

if _uri:
    try:
        import mlflow

        mlflow.set_tracking_uri(_uri)
        _enabled = True
    except ImportError:
        print("MLFLOW_TRACKING_URI set but mlflow not installed - tracking disabled")


class _NullLogger:
    def log_params(self, params): pass
    def log_epoch(self, step, metrics): pass
    def log_summary(self, metrics): pass
    def log_artifact_text(self, name, text): pass
    @property
    def run_id(self): return ""


class _MlflowLogger:
    def __init__(self, mlflow, run_id):
        self._mlflow = mlflow
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_params(self, params: dict):
        self._mlflow.log_params(params)

    def log_epoch(self, step: int, metrics: dict):
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_summary(self, metrics: dict):
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_artifact_text(self, name: str, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / name
            p.write_text(text)
            self._mlflow.log_artifact(str(p))


@contextmanager
def session(tags: dict[str, str] | None = None):
    """Parent run covering a whole CV training session."""
    if not _enabled:
        yield _NullLogger()
        return

    import mlflow

    mlflow.set_experiment("name-that-llm")
    with mlflow.start_run(run_name="cv-session", tags=tags) as run:
        yield _MlflowLogger(mlflow, run.info.run_id)


@contextmanager
def fold(parent_run_id: str, fold_idx: int):
    """Child run for one CV fold, nested under the session run."""
    if not _enabled:
        yield _NullLogger()
        return

    import mlflow

    with mlflow.start_run(
        run_name=f"fold-{fold_idx}",
        nested=True,
        parent_run_id=parent_run_id,
    ) as run:
        yield _MlflowLogger(mlflow, run.info.run_id)
