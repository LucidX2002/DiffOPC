import os
import subprocess
from pathlib import Path


def test_run_debug_visual_script_uses_single_case_iccad13_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({"PYTHON": "echo"})

    result = subprocess.run(
        ["bash", "run_debug_visual.sh"],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_output_dir = repo_root / "visual_outputs" / "ICCAD2013"
    expected_data_dir = repo_root / "benchmark" / "ICCAD2013"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

    assert lines == [
        f"src/diffopc.py opc=debug opc.IsInsertSRAF=True opc.VISUAL_DEBUG=1 opc.VISUAL_OUTPUT_DIR={expected_output_dir} "
        f"data=single data.data_dir={expected_data_dir} data.data_idx=3 extras.print_config=false "
        "logger.aim.experiment=debug_visual"
    ]
