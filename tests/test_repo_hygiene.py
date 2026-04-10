from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_stale_template_files_are_not_present() -> None:
    stale_paths = [
        REPO_ROOT / "src" / "train.py",
        REPO_ROOT / "src" / "eval.py",
        REPO_ROOT / "src" / "data" / "mnist_datamodule.py",
        REPO_ROOT / "docs" / "FRAMEWORK-README.md",
        REPO_ROOT / "configs" / "hparams_search" / "mnist_optuna.yaml",
        REPO_ROOT / "tests" / "conftest.py",
        REPO_ROOT / "tests" / "helpers",
        REPO_ROOT / "tests" / "test_datamodules.py",
        REPO_ROOT / "tests" / "test_eval.py",
        REPO_ROOT / "tests" / "test_sweeps.py",
        REPO_ROOT / "tests" / "test_train.py",
    ]

    for stale_path in stale_paths:
        assert not stale_path.exists(), f"stale template path should be removed: {stale_path}"


def test_public_entrypoints_do_not_reference_removed_template_content() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    setup_py = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "src/train.py" not in makefile
    assert "git pull origin main" not in makefile
    assert "src.train" not in setup_py
    assert "src.eval" not in setup_py
    assert "Describe Your Cool Project" not in setup_py
    assert "github.com/user/project" not in setup_py
    assert "FRAMEWORK-README" not in readme
    assert "src/opc/levelset.py" not in readme
    assert "## Run segments" not in readme
