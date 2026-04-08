import pickle
from pathlib import Path

import cv2
import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

from aim import Run

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.litho.simple import LithoSim
from src.opc.evaluation import evaluate, format_metrics
from src.utils.adabox_rectangles import binary_array_to_rectangles


def build_case_files(case_count, mask_pattern, target_pattern, mask_start_idx=1, target_start_idx=1):
    case_files = []
    for offset in range(case_count):
        mask_idx = mask_start_idx + offset
        target_idx = target_start_idx + offset
        case_files.append((mask_pattern.format(idx=mask_idx), target_pattern.format(idx=target_idx)))
    return case_files


def image2rects(img_path, resize_shape=(512, 512)):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    threshold = 127
    _, binary_array = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    binary_array = binary_array.astype(np.uint8)
    binary_array = np.rot90(binary_array, 3)
    binary_array_resized = cv2.resize(binary_array, resize_shape, interpolation=cv2.INTER_NEAREST)
    return binary_array_to_rectangles(binary_array_resized)


def filter_rect(rect, area=400, wh=20):
    if rect.get_area() < area:
        return False
    width = abs(rect.x2 - rect.x1)
    height = abs(rect.y2 - rect.y1)
    if width < wh or height < wh:
        return False
    return True


def rects2image(rects, shape=(512, 512), min_area=400, min_wh=20):
    image = Image.new("1", shape)
    draw = ImageDraw.Draw(image)
    all_rect_num = len(rects)
    filtered_rect_num = 0
    for rect in rects:
        if filter_rect(rect, min_area, min_wh):
            filtered_rect_num += 1
            x1, x2, y1, y2 = rect.x1, rect.x2, rect.y1, rect.y2
            draw.rectangle((x1, y1, x2, y2), fill=1)
    print(f"Remaining {filtered_rect_num} from {all_rect_num} rectangles")
    return image


def save_rects(mask_dir, case_files, resize_shape=(2048, 2048)):
    for m_name, _ in case_files:
        m_path = f"{mask_dir}/{m_name}"
        # print(f"Processing {m_path}")
        out_dir = Path(mask_dir).parent / "rects" / f"{resize_shape[0]}x{resize_shape[1]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        o_name = f"{Path(m_name).stem}.pkl"
        o_path = out_dir / o_name
        if not o_path.exists():
            rects = image2rects(m_path, resize_shape=resize_shape)
            with open(o_path, "wb") as f:
                pickle.dump(rects, f)
        else:
            print(f"{o_path} already exists, pass")


def save_images(rects_dir, case_files, rect_shape=(2048, 2048), min_area=400, min_wh=20):
    shape = rect_shape
    for mask_name, _ in case_files:
        m_name = f"{Path(mask_name).stem}.pkl"
        m_path = f"{rects_dir}/{m_name}"
        # print(f"Processing {m_path}")
        out_dir = Path(rects_dir).parent / f"{shape[0]}x{shape[1]}_filtered" / f"area_{min_area}_wh_{min_wh}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = f"{str(out_dir)}/{Path(m_name).stem}.png"
        if Path(out_path).exists():
            print(f"{out_path} already exists, pass")
            continue
        rects = pickle.load(open(m_path, "rb"))
        image = rects2image(rects, shape, min_area, min_wh)
        # image = np.rot90(image, 2)
        image = image.rotate(180)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image.save(out_path)
        print(f"save to {str(out_dir)}/{Path(m_name).stem}.png")


def eval_filtered(mask_dir, target_dir, run, case_files):
    lithoCfg = OmegaConf.load("./configs/litho/default.yaml")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    litho = LithoSim(lithoCfg.litho_config, device)
    l2s = []
    pvbs = []
    epes = []
    res_str = ""
    for case_num, (m_name, t_name) in enumerate(case_files, start=1):
        m_path = f"{mask_dir}/{m_name}"
        target_path = f"{target_dir}/{t_name}"
        # print(f"Processing {m_path}")
        mask = cv2.imread(m_path)[:, :, 0] / 255
        target = cv2.imread(target_path)[:, :, 0] / 255
        l2, pvb, epe, nshot = evaluate(mask, target, litho, device, shots=False)
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        metrics = format_metrics(l2, pvb, epe, nshot)
        print(f"[{m_path}]:\n {metrics}")
        res_str += f"[Testcase{case_num}]: {metrics}\n"
    print(res_str)
    print(f"[Result]: L2 {np.mean(l2s):.0f}; PVBand {np.mean(pvbs):.0f}; EPE {np.mean(epes):.1f}; ")
    run.track(np.mean(l2s), name="L2")
    run.track(np.mean(pvbs), name="PVB")
    run.track(np.mean(epes), name="EPE")


def init_logger(cfg, repo, experiment):
    run = Run(
        repo=repo, experiment=experiment, system_tracking_interval=None, log_system_params=False, capture_terminal_logs=True
    )
    for key, value in cfg.items():
        run.set(("hparams", key), value, strict=False)
    return run


@hydra.main(version_base=None, config_path="../../configs/mrc", config_name="mrc_curvlarge")
def main(cfg: DictConfig):
    mask_dir = cfg.mask_dir
    target_dir = cfg.target_dir
    case_files = build_case_files(
        case_count=cfg.case_count,
        mask_pattern=cfg.mask_pattern,
        target_pattern=cfg.target_pattern,
        mask_start_idx=cfg.mask_start_idx,
        target_start_idx=cfg.target_start_idx,
    )
    rect_shape = (cfg.rect_shape_w, cfg.rect_shape_h)
    min_area = cfg.min_area
    min_wh = cfg.min_wh
    exp_folder = cfg.exp_folder
    exp_name = cfg.exp_name
    # mask_dir = "./benchmark/baseline/multilevel/mask"
    run = init_logger(cfg, exp_folder, exp_name)
    save_rects(mask_dir, case_files, resize_shape=rect_shape)
    # rects_dir = f"./benchmark/baseline/multilevel/rects/{shape[0]}x{shape[1]}"
    rects_dir = Path(mask_dir).parent / "rects" / f"{rect_shape[0]}x{rect_shape[1]}"
    save_images(rects_dir, case_files, rect_shape=rect_shape, min_area=min_area, min_wh=min_wh)
    filterd_dir = Path(rects_dir).parent / f"{rect_shape[0]}x{rect_shape[1]}_filtered" / f"area_{min_area}_wh_{min_wh}"
    run.set(("hparams", "filterd_dir"), str(filterd_dir), strict=False)
    eval_filtered(filterd_dir, target_dir, run, case_files)


if __name__ == "__main__":
    main()
    # resize_shape = (2048, 2048)
    # resize_shape = (512, 512)
    # resize_shape = (256, 256)
    # shape = (256, 256)

    # save_rects(resize_shape=(256, 256))
    # save_images(rect_shape=(256, 256))

    # save_rects(resize_shape=(2048, 2048))
    # min_area = 60
    # min_wh = 3
    # save_images(rect_shape=(2048, 2048), min_area=min_area, min_wh=min_wh)

    # mask_dir = f"./benchmark/baseline/multilevel/rects/2048x2048_filtered/area_{min_area}_wh_{min_wh}"
    # target_dir = "./benchmark/baseline/multilevel/target"
    # eval_filtered(mask_dir, target_dir)
