import os
import subprocess
from pathlib import Path


def test_export_dopc_env_script_dry_run_uses_expected_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "DOPC_EXPORT_DRY_RUN": "1",
            "DOPC_ENV_NAME": "dopc",
            "DOPC_REQUIREMENTS_OUTPUT": "requirements.lock.txt",
            "DOPC_CONDA_OUTPUT": "environment.lock.yaml",
        }
    )

    result = subprocess.run(
        ["bash", "export_dopc_env.sh"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        "conda run -n dopc python -m pip freeze --exclude-editable > requirements.lock.txt",
        "conda env export -n dopc > environment.lock.yaml",
    ]


def test_environment_yaml_declares_python_3_11() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment_yaml = (repo_root / "environment.yaml").read_text(encoding="utf-8")

    assert "name: dopc" in environment_yaml
    assert "python=3.11" in environment_yaml
