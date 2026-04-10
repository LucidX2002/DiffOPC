import struct
from pathlib import Path

import numpy as np

from src.utils.gds_export import export_case_mask, export_mask_to_gds, mask_to_rectangles


def _read_gds_record_types(path: Path) -> list[int]:
    data = path.read_bytes()
    record_types = []
    offset = 0
    while offset < len(data):
        length, record_type, _data_type = struct.unpack(">HBB", data[offset : offset + 4])
        record_types.append(record_type)
        offset += length
    assert offset == len(data)
    return record_types


def test_mask_to_rectangles_merges_identical_runs_across_rows() -> None:
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    rectangles = mask_to_rectangles(mask)

    assert rectangles == [(1, 0, 3, 2), (0, 2, 3, 3)]


def test_export_mask_to_gds_writes_expected_boundary_records(tmp_path: Path) -> None:
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    output_path = tmp_path / "case1.gds"

    gds_path, rect_count = export_mask_to_gds(
        mask,
        output_path,
        layer=7,
        datatype=3,
        library_name="DiffOPC",
        structure_name="M1_test1",
    )

    assert gds_path == output_path
    assert rect_count == 2
    assert output_path.exists()

    record_types = _read_gds_record_types(output_path)
    assert record_types[0] == 0x00
    assert record_types.count(0x08) == 2
    assert record_types[-1] == 0x04

    data = output_path.read_bytes()
    assert b"DIFFOPC" in data
    assert b"M1_TEST1" in data


def test_export_case_mask_uses_configured_output_location(tmp_path: Path) -> None:
    mask = np.array([[1, 1], [1, 1]], dtype=np.uint8)
    export_cfg = {
        "output_dir": str(tmp_path),
        "layer": 1,
        "datatype": 0,
        "library_name": "DiffOPC",
        "file_prefix": "case_",
        "flip_y": False,
    }

    gds_path, rect_count = export_case_mask(mask, export_cfg, 5)

    assert gds_path == tmp_path / "case_5.gds"
    assert gds_path.exists()
    assert rect_count == 1
