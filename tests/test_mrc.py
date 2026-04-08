import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import numpy as np
import torch

import src.mrc.mrc as mrc_module
import src.opc.evaluation as evaluation_module
from src.mrc.mrc import build_case_files
from src.utils.adabox_compat import compatible_get_separation_value, patch_adabox_mode_compat


def test_mrc_curvlarge_config_matches_local_benchmark_layout(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(repo_root))

    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "mrc")):
        cfg = compose(config_name="mrc_curvlarge")

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    case_files = build_case_files(
        case_count=resolved["case_count"],
        mask_pattern=resolved["mask_pattern"],
        target_pattern=resolved["target_pattern"],
        mask_start_idx=resolved["mask_start_idx"],
        target_start_idx=resolved["target_start_idx"],
    )

    assert resolved["mask_dir"] == str(repo_root / "benchmark" / "baseline" / "curvmulti-large" / "mask")
    assert resolved["target_dir"] == str(repo_root / "benchmark" / "baseline" / "curvmulti-large" / "target_bk")
    assert case_files[0] == ("MultiLevel_mask1.png", "t11_0_mask.png")
    assert case_files[-1] == ("MultiLevel_mask10.png", "t20_0_mask.png")


def test_mrc_default_config_uses_matching_multilevel_names(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PROJECT_ROOT", str(repo_root))

    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs" / "mrc")):
        cfg = compose(config_name="mrc")

    resolved = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    case_files = build_case_files(
        case_count=resolved["case_count"],
        mask_pattern=resolved["mask_pattern"],
        target_pattern=resolved["target_pattern"],
        mask_start_idx=resolved["mask_start_idx"],
        target_start_idx=resolved["target_start_idx"],
    )

    assert resolved["mask_dir"] == str(repo_root / "benchmark" / "baseline" / "multilevel" / "mask")
    assert resolved["target_dir"] == str(repo_root / "benchmark" / "baseline" / "multilevel" / "target")
    assert case_files[0] == ("MultiLevel_mask1.png", "MultiLevel_target1.png")
    assert case_files[-1] == ("MultiLevel_mask10.png", "MultiLevel_target10.png")


def test_compatible_get_separation_value_handles_scalar_scipy_mode() -> None:
    data = np.array(
        [
            [0, 0, 0],
            [0, 2, 0],
            [2, 0, 0],
            [2, 2, 0],
        ],
        dtype=float,
    )

    assert compatible_get_separation_value(data) == 2.0


def test_patch_adabox_mode_compat_replaces_target_function() -> None:
    class DummyModule:
        pass

    proc = DummyModule()
    tools = DummyModule()

    patch_adabox_mode_compat(proc, tools)

    assert proc.get_separation_value is compatible_get_separation_value
    assert tools.get_separation_value is compatible_get_separation_value


def test_run_mrc_script_names_experiments_by_area_and_width() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update({"PYTHON": "echo", "MIN_AREAS": "5", "MIN_WHS": "1 3"})

    result = subprocess.run(["bash", "run_mrc.sh"], cwd=repo_root, env=env, check=True, capture_output=True, text=True)

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines == [
        'src/mrc/mrc.py --config-name mrc_curvlarge min_area=5 min_wh=1 exp_name=mrc_marea_5_mwh_1',
        'src/mrc/mrc.py --config-name mrc_curvlarge min_area=5 min_wh=3 exp_name=mrc_marea_5_mwh_3',
    ]


def test_eval_filtered_omits_shot_output_when_shot_counting_is_disabled(monkeypatch, capsys) -> None:
    class DummyRun:
        def __init__(self) -> None:
            self.tracked = []

        def track(self, value, name) -> None:
            self.tracked.append((name, value))

    class DummyLitho:
        def __init__(self, _config, _device) -> None:
            pass

    def fake_evaluate(mask, target, litho, device, shots=False):
        assert shots is False
        return 1.0, 2.0, 3.0, -1

    monkeypatch.setattr(mrc_module.OmegaConf, "load", lambda _path: OmegaConf.create({"litho_config": {}}))
    monkeypatch.setattr(mrc_module, "LithoSim", DummyLitho)
    monkeypatch.setattr(mrc_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(mrc_module.cv2, "imread", lambda _path: np.ones((2, 2, 3), dtype=np.uint8) * 255)

    run = DummyRun()
    mrc_module.eval_filtered("mask_dir", "target_dir", run, [("mask1.png", "target1.png")])

    captured = capsys.readouterr()
    assert "Shot:" not in captured.out
    assert run.tracked == [("L2", 1.0), ("PVB", 2.0), ("EPE", 3.0)]


def test_evaluate_verbose_output_does_not_require_maskfile(monkeypatch, capsys) -> None:
    class DummyBasic:
        def __init__(self, litho, device, thresh=0.5) -> None:
            pass

        def run(self, mask, target, scale=1):
            return 1.0, 2.0

    class DummyEPEChecker:
        def __init__(self, litho, device, thresh=0.5) -> None:
            pass

        def run(self, mask, target, scale=1):
            return 3.0, 4.0

    class DummyShotCounter:
        def __init__(self, litho, device, thresh=0.5) -> None:
            pass

        def run(self, mask, shape=(512, 512)):
            return 5.0

    monkeypatch.setattr(evaluation_module, "Basic", DummyBasic)
    monkeypatch.setattr(evaluation_module, "EPEChecker", DummyEPEChecker)
    monkeypatch.setattr(evaluation_module, "ShotCounter", DummyShotCounter)

    result = evaluation_module.evaluate(
        mask=np.zeros((2, 2), dtype=np.float32),
        target=np.zeros((2, 2), dtype=np.float32),
        litho=object(),
        device=torch.device("cpu"),
        shots=True,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert captured.out.strip() == "L2 1; PVBand 2; EPE 7; Shot: 5"
    assert result == (1.0, 2.0, 7.0, 5.0)
