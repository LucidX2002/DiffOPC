from hydra import compose, initialize
from omegaconf import OmegaConf

from src.opc.edgeilt import EdgeILTCfg


def test_diffopc_default_config_has_runtime_required_keys() -> None:
    """Ensure the default DiffOPC config resolves all runtime-required OPC settings."""

    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="diffopc.yaml",
            overrides=[
                "extras.enforce_tags=false",
                "extras.print_config=false",
            ],
        )

    resolved_opc = OmegaConf.to_container(cfg.opc, resolve=True, throw_on_missing=True)
    resolved_data = OmegaConf.to_container(cfg.data, resolve=True, throw_on_missing=True)
    opc_cfg = EdgeILTCfg(resolved_opc)

    assert resolved_data["sraf_forbidden"] == opc_cfg["SRAF_FORBIDDEN"]
    assert opc_cfg["IsInsertSRAF"] is False
    assert opc_cfg["WeightL2"] > 0
