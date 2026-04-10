from __future__ import annotations

import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np


Rectangle = Tuple[int, int, int, int]


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if hasattr(cfg, "get"):
        value = cfg.get(key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def _sanitize_gds_name(name: str, fallback: str) -> str:
    filtered = "".join(ch if ch.isalnum() or ch in {"_", "$", "?"} else "_" for ch in name.upper())
    filtered = filtered[:32].strip("_")
    return filtered or fallback


def _encode_real8(value: float) -> bytes:
    if value == 0:
        return b"\0" * 8

    sign = 0x80 if value < 0 else 0
    value = abs(value)
    exponent = 64

    while value >= 1:
        value /= 16.0
        exponent += 1
    while value < 0.0625:
        value *= 16.0
        exponent -= 1

    mantissa = int(value * (1 << 56))
    if mantissa == 1 << 56:
        mantissa //= 16
        exponent += 1

    return bytes([sign | exponent]) + mantissa.to_bytes(7, byteorder="big")


def _gds_record(record_type: int, data_type: int, payload: bytes = b"") -> bytes:
    if len(payload) % 2:
        payload += b"\0"
    return struct.pack(">HBB", 4 + len(payload), record_type, data_type) + payload


def _pack_int2(values: Sequence[int]) -> bytes:
    return b"".join(struct.pack(">h", value) for value in values)


def _pack_int4(values: Sequence[int]) -> bytes:
    return b"".join(struct.pack(">i", value) for value in values)


def _pack_ascii(value: str) -> bytes:
    return value.encode("ascii", errors="ignore")


def _normalize_mask(mask: Any) -> np.ndarray:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {array.shape}")
    return np.ascontiguousarray(array > 0.5, dtype=bool)


def mask_to_rectangles(mask: Any) -> List[Rectangle]:
    """Decompose a binary mask into vertically merged rectangles.

    The decomposition is exact in raster space: each rectangle covers a run of on-pixels and
    adjacent rows are merged when they share the same horizontal span.
    """

    binary_mask = _normalize_mask(mask)
    height, width = binary_mask.shape
    rectangles: List[Rectangle] = []
    active_runs: dict[Tuple[int, int], int] = {}

    for y in range(height):
        row = binary_mask[y]
        current_runs: set[Tuple[int, int]] = set()
        x = 0
        while x < width:
            if not row[x]:
                x += 1
                continue

            x0 = x
            while x < width and row[x]:
                x += 1
            run = (x0, x)
            current_runs.add(run)
            active_runs.setdefault(run, y)

        finished_runs = [run for run in active_runs if run not in current_runs]
        for run in finished_runs:
            start_y = active_runs.pop(run)
            rectangles.append((run[0], start_y, run[1], y))

    for run, start_y in active_runs.items():
        rectangles.append((run[0], start_y, run[1], height))

    return rectangles


def write_gds(
    output_path: str | Path,
    rectangles: Iterable[Rectangle],
    *,
    height: int,
    layer: int = 1,
    datatype: int = 0,
    library_name: str = "DIFFOPC",
    structure_name: str = "TOP",
    user_unit_meters: float = 1e-6,
    database_unit_meters: float = 1e-9,
    flip_y: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    library_name = _sanitize_gds_name(library_name, "DIFFOPC")
    structure_name = _sanitize_gds_name(structure_name, "TOP")
    timestamp = datetime.now(timezone.utc)
    time_fields = [
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        timestamp.second,
    ]

    rectangles = list(rectangles)
    units = (
        _encode_real8(database_unit_meters / user_unit_meters),
        _encode_real8(database_unit_meters),
    )

    with output_path.open("wb") as stream:
        stream.write(_gds_record(0x00, 0x02, _pack_int2([600])))
        stream.write(_gds_record(0x01, 0x02, _pack_int2(time_fields + time_fields)))
        stream.write(_gds_record(0x02, 0x06, _pack_ascii(library_name)))
        stream.write(_gds_record(0x03, 0x05, b"".join(units)))
        stream.write(_gds_record(0x05, 0x02, _pack_int2(time_fields + time_fields)))
        stream.write(_gds_record(0x06, 0x06, _pack_ascii(structure_name)))

        for x0, y0, x1, y1 in rectangles:
            if flip_y:
                y0, y1 = height - y1, height - y0

            points = [
                x0,
                y0,
                x1,
                y0,
                x1,
                y1,
                x0,
                y1,
                x0,
                y0,
            ]
            stream.write(_gds_record(0x08, 0x00))
            stream.write(_gds_record(0x0D, 0x02, _pack_int2([layer])))
            stream.write(_gds_record(0x0E, 0x02, _pack_int2([datatype])))
            stream.write(_gds_record(0x10, 0x03, _pack_int4(points)))
            stream.write(_gds_record(0x11, 0x00))

        stream.write(_gds_record(0x07, 0x00))
        stream.write(_gds_record(0x04, 0x00))

    return output_path


def export_mask_to_gds(
    mask: Any,
    output_path: str | Path,
    *,
    layer: int = 1,
    datatype: int = 0,
    library_name: str = "DIFFOPC",
    structure_name: str = "TOP",
    flip_y: bool = False,
) -> Tuple[Path, int]:
    binary_mask = _normalize_mask(mask)
    rectangles = mask_to_rectangles(binary_mask)
    gds_path = write_gds(
        output_path,
        rectangles,
        height=binary_mask.shape[0],
        layer=layer,
        datatype=datatype,
        library_name=library_name,
        structure_name=structure_name,
        flip_y=flip_y,
    )
    return gds_path, len(rectangles)


def export_case_mask(
    mask: Any,
    export_cfg: Any,
    case_id: Any,
) -> Tuple[Path, int]:
    file_prefix = str(_cfg_value(export_cfg, "file_prefix", "M1_test"))
    structure_name = f"{file_prefix}{case_id}"
    output_path = Path(str(_cfg_value(export_cfg, "output_dir", "."))) / f"{structure_name}.gds"
    return export_mask_to_gds(
        mask,
        output_path,
        layer=int(_cfg_value(export_cfg, "layer", 1)),
        datatype=int(_cfg_value(export_cfg, "datatype", 0)),
        library_name=str(_cfg_value(export_cfg, "library_name", "DIFFOPC")),
        structure_name=structure_name,
        flip_y=bool(_cfg_value(export_cfg, "flip_y", False)),
    )
