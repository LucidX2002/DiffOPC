import torch
from torch.utils.data import Dataset

import src.data.loaders.glp_seg as glp_seg
from src.data.loaders.segments import SegmentsInitTorch


class Iccad13Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tile_size_x: int,
        tile_size_y: int,
        offset_x: int,
        offset_y: int,
        seg_length: int,
        start_idx: int,
        end_idx: int,
        down_scale: int,
        device: torch.device,
    ):
        self.data_dir = data_dir
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.seg_length = seg_length
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.down_scale = down_scale
        self.device = device

    def __len__(self):
        return self.end_idx - self.start_idx + 1

    def __getitem__(self, index):
        data_idx = index + self.start_idx
        assert data_idx <= self.end_idx, f"Index {index} out of bounds"
        glp_path = f"{self.data_dir}/M1_test{data_idx}.glp"
        design = glp_seg.Design(glp_path, down=self.down_scale)
        target, edge_params, metadata = SegmentsInitTorch().run(
            design,
            self.tile_size_x,
            self.tile_size_y,
            self.offset_x,
            self.offset_y,
            self.seg_length,
            self.device,
        )
        return target, edge_params, metadata, data_idx


class Iccad13Single(Dataset):
    def __init__(
        self,
        data_dir: str,
        tile_size_x: int,
        tile_size_y: int,
        offset_x: int,
        offset_y: int,
        seg_length: int,
        data_idx: int,
        down_scale: int,
        device: torch.device,
    ):
        self.data_dir = data_dir
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.seg_length = seg_length
        self.data_idx = data_idx
        self.down_scale = down_scale
        self.device = device

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # assert index == 0, f"Index {index} out of bounds"
        data_idx = index + self.data_idx
        glp_path = f"{self.data_dir}/M1_test{data_idx}.glp"
        print(glp_path)
        design = glp_seg.Design(glp_path, down=self.down_scale)
        target, edge_params, metadata = SegmentsInitTorch().run(
            design,
            self.tile_size_x,
            self.tile_size_y,
            self.offset_x,
            self.offset_y,
            self.seg_length,
            self.device,
        )
        return target, edge_params, metadata, data_idx
