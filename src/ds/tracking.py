"""MLflow setup shared by every DS step."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402

from config import MLFLOW_ARTIFACT_ROOT, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, ROOT  # noqa: E402


@contextmanager
def staged_code_paths():
    """Plain copies of src/**/*.py and config.py for mlflow `code_paths`.

    Files under OneDrive are reparse points; copying them directly leaves temp files that MLflow's cleanup
    cannot delete (WinError 5). Rewriting the bytes produces ordinary files, and skips __pycache__.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for src in (ROOT / "src").rglob("*.py"):
            dst = tmp / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        (tmp / "config.py").write_bytes((ROOT / "config.py").read_bytes())
        yield [str(tmp / "src"), str(tmp / "config.py")]


def setup() -> str:
    """Point MLflow at the project store and return the experiment id (created on first use)."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if exp is None:
        exp_id = mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACT_ROOT)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id
