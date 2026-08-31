# =========================================================
# SOLAR FARM EXTRACTION - DINOv2 VERSION (PyTorch / HuggingFace)
#
#
# USAGE:
#   python train_dinov2_solar.py <DATA_PATH> <OUTPUT_FOLDER> <MODEL_NAME>
#       [--backbone D:\dinov2-small-local] [--no-pretrained] [--freeze-backbone]
#       [--unfreeze-after-epoch 0] [--head-channels 256] [--head-norm group]
#       [--epochs 200] [--batch-size 8] [--image-size 126] [--no-amp]
#       [--threshold 0.5] [--backbone-lr 1e-5] [--head-lr 1e-4]
#       [--weight-decay 0.05] [--layer-decay 0.75]
#       [--loss dice_bce] [--auto-pos-weight]
#       [--scheduler cosine] [--warmup-epochs 5] [--min-lr-ratio 0.01]
#       [--split-by scene] [--scene-regex REGEX]
#       [--elastic] [--gradient-checkpointing] [--early-stop-patience 10]
#
# Recommended extra dependency (optional, graceful fallback if missing):
#   pip install -U albumentations
#
# =========================================================

import os
import re
import csv
import glob
import time
import json
import math
import random
import platform
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import rasterio
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import warnings
warnings.filterwarnings('ignore')


# =========================================================
# HARDWARE / GPU INFO  (best-effort, never raises)
# =========================================================
def get_hardware_info_lines():
    """Returns a list of human-readable hardware/software info lines."""
    lines = []
    lines.append(f"PyTorch version    : {torch.__version__}")
    lines.append(f"Python version     : {platform.python_version()}")
    lines.append(f"OS                 : {platform.platform()}")

    if torch.cuda.is_available():
        idx   = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        lines.append(f"Compute device     : GPU (CUDA)")
        lines.append(f"GPU name           : {props.name}")
        lines.append(f"GPU count          : {torch.cuda.device_count()}")
        lines.append(f"Total VRAM         : {props.total_memory / 1024**3:.2f} GB")
        lines.append(f"Compute capability : {props.major}.{props.minor}")
        lines.append(f"Multiprocessors    : {props.multi_processor_count}")
        lines.append(f"CUDA (build)       : {torch.version.cuda}")
        try:
            lines.append(f"cuDNN version      : {torch.backends.cudnn.version()}")
        except Exception:
            pass
        try:
            driver = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                timeout=5, stderr=subprocess.DEVNULL
            ).decode().strip().splitlines()[0]
            lines.append(f"NVIDIA driver      : {driver}")
        except Exception:
            pass
    else:
        lines.append("Compute device     : CPU")
        lines.append(f"CPU                : {platform.processor() or 'unknown'}")

    return lines


def print_hardware_info(header="GPU / HARDWARE INFO"):
    """Prints a boxed hardware info block and returns the raw lines (for logging)."""
    lines = get_hardware_info_lines()
    bar = "=" * 60
    print(f"\n{bar}")
    print(header)
    print(bar)
    for line in lines:
        print(f"  {line}")
    print(f"{bar}\n")
    return lines


def hardware_summary_line():
    """One-line hardware label, e.g. 'GPU: NVIDIA RTX 6000 Ada Generation' or 'CPU'."""
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_properties(torch.cuda.current_device()).name}"
    return "CPU"

try:
    import albumentations as A
    import cv2 as _cv2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False

# -------------------------------------------------
# CLI ARGS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('output_folder')
parser.add_argument('model_name')
parser.add_argument('--backbone', default=r'D:\dinov2-small-local',
                    help='HuggingFace DINOv2 backbone name or local path '
                         '(default: D:\\dinov2-small-local). Pass --backbone to '
                         'override, e.g. facebook/dinov2-small or another local path.')
parser.add_argument('--no-pretrained', action='store_true',
                    help='Randomly initialise the backbone (no DINOv2 pretrained weights)')
parser.add_argument('--freeze-backbone', action='store_true',
                    help='Permanently freeze the DINOv2 backbone and train only the conv '
                         'segmentation head (linear-probe style). Recommended for '
                         '-large/-giant backbones or small labeled datasets.')
parser.add_argument('--unfreeze-after-epoch', type=int, default=0,
                    help='Start with the backbone frozen (head-only warmup) for this many '
                         'epochs, then unfreeze it and add it to the optimizer at '
                         '--backbone-lr. 0 disables this (default). Ignored if '
                         '--freeze-backbone is set (that freeze is permanent).')
parser.add_argument('--head-channels', type=int, default=256,
                    help='Hidden width of the conv segmentation head on top of DINOv2 features')
parser.add_argument('--head-norm', choices=['group', 'batch'], default='group',
                    help='Normalization used inside the segmentation head. GroupNorm '
                         '(default) is more stable than BatchNorm at small batch sizes '
                         '(e.g. batch<=8).')
parser.add_argument('--epochs',     type=int, default=200)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--image-size', type=int, default=126,
                    help='Auto-rounded to the nearest multiple of the DINOv2 patch size (14)')
parser.add_argument('--no-amp', action='store_true',
                    help='Disable mixed precision (AMP) training')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Probability threshold used for IoU/F1/Dice/precision/recall during training')
parser.add_argument('--backbone-lr', type=float, default=1e-5,
                    help='Learning rate for the pretrained DINOv2 backbone (top-layer LR '
                         'if --layer-decay < 1.0; ignored while the backbone is frozen)')
parser.add_argument('--head-lr', type=float, default=1e-4,
                    help='Learning rate for the randomly-initialised segmentation head')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help="AdamW weight decay. 0.05 (Meta's common DINOv2 fine-tuning value) "
                         'by default.')
parser.add_argument('--layer-decay', type=float, default=0.75,
                    help='Layer-wise LR decay factor across backbone transformer blocks '
                         '(deeper layers closer to --backbone-lr, earlier layers smaller). '
                         'Set to 1.0 to disable and use a single flat backbone LR.')
parser.add_argument('--loss', choices=['dice_bce', 'dice_focal_bce', 'dice_lovasz', 'dice_tversky'],
                    default='dice_bce',
                    help='Segmentation loss. dice_bce (default) is the standard modern combo; '
                         'dice_focal_bce reproduces the previous default; dice_lovasz and '
                         'dice_tversky are also available.')
parser.add_argument('--auto-pos-weight', action='store_true',
                    help='Compute a BCE pos_weight from the training set\'s positive/negative '
                         'pixel ratio (capped to [1, 50]) instead of leaving BCE unweighted.')
parser.add_argument('--scheduler', choices=['cosine', 'plateau'], default='cosine',
                    help='cosine (default): linear warmup then cosine decay. plateau: the '
                         'previous ReduceLROnPlateau behavior.')
parser.add_argument('--warmup-epochs', type=int, default=5,
                    help='Linear warmup epochs before cosine decay begins (--scheduler cosine only)')
parser.add_argument('--min-lr-ratio', type=float, default=0.01,
                    help='Cosine schedule floor, as a fraction of each param group\'s base LR '
                         '(--scheduler cosine only)')
parser.add_argument('--split-by', choices=['scene', 'random'], default='scene',
                    help='scene (default): split train/val by scene id parsed from the filename '
                         '(GroupShuffleSplit, no scene appears in both sets). Falls back to '
                         'random automatically (with a warning) if the regex only finds one '
                         'scene. random: the previous behavior.')
parser.add_argument('--scene-regex', default=None,
                    help='Regex with one capture group, matched against the image filename stem, '
                         'used to derive a scene id for --split-by scene. Default pattern strips '
                         '1-3 trailing "_<number>" groups, e.g. "sceneA_003_012" -> "sceneA".')
parser.add_argument('--elastic', action='store_true',
                    help='Include ElasticTransform in the geometric augmentation pipeline '
                         '(off by default — optional/heavier augmentation).')
parser.add_argument('--gradient-checkpointing', action='store_true',
                    help='Enable gradient checkpointing on the backbone to reduce memory use '
                         '(useful for dinov2-large/giant). Ignored while the backbone is frozen.')
parser.add_argument('--early-stop-patience', type=int, default=10)
args = parser.parse_args()

DINOV2_PATCH_SIZE = 14  # fixed for all official facebook/dinov2-* checkpoints
DEFAULT_SCENE_REGEX = r'^(.*?)(?:_\d+){1,3}$'


def _round_to_patch_multiple(size, patch=DINOV2_PATCH_SIZE):
    if size % patch == 0:
        return size
    rounded = int(round(size / patch)) * patch
    return max(rounded, patch)


_requested_image_size = args.image_size
IMAGE_SIZE = _round_to_patch_multiple(args.image_size)
if IMAGE_SIZE != _requested_image_size:
    print(f'[image-size] {_requested_image_size} is not a multiple of the DINOv2 patch size '
          f'({DINOV2_PATCH_SIZE}) -> rounded to {IMAGE_SIZE}.')

BATCH_SIZE           = args.batch_size
EPOCHS                = args.epochs
BACKBONE              = args.backbone
PRETRAINED            = not args.no_pretrained
FREEZE_BACKBONE       = args.freeze_backbone
UNFREEZE_AFTER_EPOCH  = args.unfreeze_after_epoch
HEAD_CHANNELS         = args.head_channels
HEAD_NORM             = args.head_norm
USE_AMP               = not args.no_amp
THRESHOLD             = args.threshold
BACKBONE_LR           = args.backbone_lr
HEAD_LR               = args.head_lr
WEIGHT_DECAY          = args.weight_decay
LAYER_DECAY           = args.layer_decay
LOSS_NAME             = args.loss
AUTO_POS_WEIGHT       = args.auto_pos_weight
SCHEDULER_KIND        = args.scheduler
WARMUP_EPOCHS         = args.warmup_epochs
MIN_LR_RATIO          = args.min_lr_ratio
SPLIT_BY              = args.split_by
SCENE_REGEX           = args.scene_regex
ELASTIC_AUGMENT       = args.elastic
GRADIENT_CHECKPOINTING = args.gradient_checkpointing
EARLY_STOP_PATIENCE   = args.early_stop_patience
DEVICE                = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY            = torch.cuda.is_available()

if FREEZE_BACKBONE and UNFREEZE_AFTER_EPOCH > 0:
    print('[freeze] --freeze-backbone is permanent; ignoring --unfreeze-after-epoch.')
    UNFREEZE_AFTER_EPOCH = 0

DATA_PATH     = args.data_path
OUTPUT_FOLDER = args.output_folder
model_name    = args.model_name
BEST_MODEL    = os.path.join(OUTPUT_FOLDER, model_name + '_best.pth')
LAST_MODEL    = os.path.join(OUTPUT_FOLDER, model_name + '_last.pth')
MEAN_STD_FILE = os.path.join(OUTPUT_FOLDER, 'normalization_values.txt')
TRAIN_LOG     = os.path.join(OUTPUT_FOLDER, 'training_log.txt')
TRAIN_LOG_CSV = os.path.join(OUTPUT_FOLDER, 'training_log.csv')
CONFIG_FILE   = os.path.join(OUTPUT_FOLDER, 'dinov2_config.json')

# Exact filename inference_dinov2_solar.py expects (MODEL_PATH = MODEL_FOLDER/<model_name>.pth).
# Written at the very end of training as a clean, weights_only-loadable state_dict.
INFERENCE_WEIGHTS_FILE = os.path.join(OUTPUT_FOLDER, model_name + '.pth')
# Threshold search output, read by inference's load_optimal_threshold().
OPTIMAL_THRESHOLD_FILE = os.path.join(OUTPUT_FOLDER, 'optimal_threshold.json')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# [OPT: TF32] Ampere/Ada/Hopper tensor cores can run float32 matmuls and cudnn
# convolutions in TF32 precision — a large speedup for negligible accuracy
# impact. Safe no-op on older GPUs/CPU (the flags just aren't consulted there).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision('high')

# [OPT: channels_last] NHWC memory format speeds up the Conv2d layers in this
# model (the DINOv2 patch-embedding projection and the conv segmentation head)
# on both GPU (cudnn/tensor-core kernels) and CPU (oneDNN). It's a pure memory
# layout change — numerically identical results — so it's safe to apply
# unconditionally. Only 4-D tensors (images, conv activations) are affected;
# the ViT's (B, N, C) token tensors are untouched.
CHANNELS_LAST = True

CSV_FIELDS = ['epoch', 'time_s',
              'train_loss', 'val_loss',
              'train_iou', 'val_iou',
              'train_f1', 'val_f1',
              'train_dice', 'val_dice',
              'train_precision', 'val_precision',
              'train_recall', 'val_recall',
              'train_specificity', 'val_specificity',
              'train_balanced_acc', 'val_balanced_acc']


def append_csv_row(row: dict):
    write_header = not os.path.exists(TRAIN_LOG_CSV)
    with open(TRAIN_LOG_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# -------------------------------------------------
# MEAN / STD  (identical to the SegFormer pair — same 4-channel
# G, R, NIR, NDVI stack derived from LISS-IV TOA-corrected tiles)
# -------------------------------------------------
def compute_mean_std(img_paths):
    sum_   = np.zeros(4)
    sum_sq = np.zeros(4)
    count  = 0

    for path in tqdm(img_paths, desc='Computing mean/std'):
        with rasterio.open(path) as src:
            g = src.read(1).astype(np.float32)
            r = src.read(2).astype(np.float32)
            n = src.read(3).astype(np.float32)

        ndvi   = (n - r) / (n + r + 1e-6)
        img    = np.stack([g, r, n, ndvi], axis=-1)
        pixels = img.reshape(-1, 4)

        sum_   += pixels.sum(axis=0)
        sum_sq += (pixels ** 2).sum(axis=0)
        count  += pixels.shape[0]

    mean = sum_ / count
    std  = np.sqrt((sum_sq / count) - mean ** 2)
    std  = np.where(std < 1e-6, 1e-6, std)
    return mean, std


def save_mean_std(mean, std):
    with open(MEAN_STD_FILE, 'w') as f:
        f.write('Mean:\n')
        f.write(','.join(map(str, mean)) + '\n')
        f.write('Std:\n')
        f.write(','.join(map(str, std)) + '\n')


# -------------------------------------------------
# CLASS BALANCE  (for --auto-pos-weight)
# -------------------------------------------------
def compute_pos_weight(img_paths, cap=(1.0, 50.0)):
    pos, total = 0, 0
    for path in tqdm(img_paths, desc='Computing class balance'):
        mask_path = path.replace('images', 'masks')
        with rasterio.open(mask_path) as src:
            m = src.read(1) > 0
        pos   += int(m.sum())
        total += m.size

    pos = max(pos, 1)
    neg = max(total - pos, 0)
    ratio = float(np.clip(neg / pos, cap[0], cap[1]))
    print(f'[pos-weight] positive pixel ratio: {pos / total:.4%}  -> '
          f'pos_weight={ratio:.2f} (capped to [{cap[0]}, {cap[1]}])')
    return ratio


# -------------------------------------------------
# SCENE-AWARE TRAIN/VAL SPLIT
# Random tile splits can leak near-duplicate neighboring tiles from the same
# scene into both train and val, inflating validation metrics. Splitting by
# scene id (parsed from the filename) avoids that.
# -------------------------------------------------
def extract_scene_id(path, pattern):
    stem = os.path.splitext(os.path.basename(path))[0]
    m = pattern.match(stem)
    return m.group(1) if m else stem


def split_train_val(all_img_paths, split_by, scene_regex, test_size, seed):
    if split_by == 'scene':
        pattern = re.compile(scene_regex) if scene_regex else re.compile(DEFAULT_SCENE_REGEX)
        groups  = [extract_scene_id(p, pattern) for p in all_img_paths]
        n_unique = len(set(groups))

        if n_unique < 2:
            print(f'[split] --split-by scene requested but only {n_unique} unique scene id(s) '
                  'were parsed from filenames (the default regex may not match your naming '
                  'convention) -> falling back to a random tile split. Pass --scene-regex to '
                  'fix parsing, or --split-by random to silence this warning.')
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_idx, val_idx = next(gss.split(all_img_paths, groups=groups))
            train_paths = [all_img_paths[i] for i in train_idx]
            val_paths   = [all_img_paths[i] for i in val_idx]
            n_train_scenes = len({groups[i] for i in train_idx})
            n_val_scenes   = len({groups[i] for i in val_idx})
            print(f'[split] scene-based split: {n_unique} unique scenes -> '
                  f'{n_train_scenes} train scenes / {n_val_scenes} val scenes (no overlap).')
            return train_paths, val_paths

    train_paths, val_paths = train_test_split(all_img_paths, test_size=test_size, random_state=seed)
    print('[split] random tile split (scene identity not enforced) -- tiles from the same '
          'scene may appear in both train and val.')
    return train_paths, val_paths


# -------------------------------------------------
# GEOMETRIC AUGMENTATION (Albumentations, when available)
# Covers: flip, rotate90, affine (scale/translate/rotate/shear), perspective,
# random-resized-crop (covers both "random crop" and "scale augmentation"),
# and optionally elastic transform. All of these are pure spatial
# interpolation/reordering ops, so they're safe to run on already-normalized,
# arbitrary-range, 4-channel float data.
# -------------------------------------------------
def build_geo_augmenter(image_size, enable_elastic=False):
    if not ALBUMENTATIONS_AVAILABLE:
        return None
    try:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(0.0, 0.05),
                     rotate=(-15, 15), shear=(-5, 5),
                     mode=_cv2.BORDER_REFLECT_101, p=0.5),
            A.Perspective(scale=(0.02, 0.06), p=0.2),
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0),
                                 ratio=(0.9, 1.11), p=0.3),
        ]
        if enable_elastic:
            transforms.append(A.ElasticTransform(alpha=1.0, sigma=25, p=0.15))
        # NOTE: 'mask' is already a built-in Albumentations target (handled via each
        # transform's apply_to_mask, nearest-neighbor by default) — no additional_targets
        # needed, and declaring it there would raise a ValueError (can't redeclare a
        # default target name).
        return A.Compose(transforms)
    except TypeError as e:
        print(f'[augment] Albumentations version mismatch building the geometric augmenter '
              f'({e}) -> falling back to flip/rot90-only geometric augmentation. '
              f'Try `pip install -U albumentations`.')
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])


if not ALBUMENTATIONS_AVAILABLE:
    print('[augment] albumentations not installed -> geometric augmentation falls back to '
          'flip/rot90 only. Install with `pip install -U albumentations` for affine/'
          'perspective/random-resized-crop/elastic augmentation.')

GEO_AUGMENTER = build_geo_augmenter(IMAGE_SIZE, enable_elastic=ELASTIC_AUGMENT)


# -------------------------------------------------
# WORKER INIT  (must be at module level — NOT inside if __name__ == '__main__' —
# so that Windows' spawn-based multiprocessing can unpickle it in worker
# sub-processes.  Defining it inside __main__ causes the AttributeError:
# "Can't get attribute '_worker_init_fn' on <module '__mp_main__'>")
# -------------------------------------------------
def _worker_init_fn(worker_id):
    # torch seeds its own per-worker RNG automatically, but Albumentations
    # (and any other code using the `random` / `numpy.random` globals) does
    # not get reseeded by PyTorch across forked workers -- without this,
    # workers can produce correlated/duplicate augmentation draws within a
    # batch. torch.initial_seed() is already unique per worker, so derive
    # the python/numpy seeds from it.
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


# -------------------------------------------------
# DATASET
# -------------------------------------------------
class SolarDataset(Dataset):
    def __init__(self, img_paths, mean, std, is_train=True):
        self.img_paths = img_paths
        self.mean      = mean
        self.std       = std
        self.is_train  = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = img_path.replace('images', 'masks')

        with rasterio.open(img_path) as src:
            g = src.read(1).astype(np.float32)
            r = src.read(2).astype(np.float32)
            n = src.read(3).astype(np.float32)

        ndvi = (n - r) / (n + r + 1e-6)
        img  = np.stack([g, r, n, ndvi], axis=-1)   # HWC
        img  = (img - self.mean) / self.std

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # CHW

        # Skip resize entirely if the tile is already the target size
        if img.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            img = F.interpolate(
                img.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
                mode='bilinear', align_corners=False
            ).squeeze(0)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        mask = (mask > 0).astype(np.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        if mask.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            mask = F.interpolate(
                mask.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE), mode='nearest'
            ).squeeze(0)

        if self.is_train:
            img, mask = self._augment(img, mask)

        return img, mask

    def _augment(self, img, mask):
        # ---- Geometric augmentation ----
        if GEO_AUGMENTER is not None:
            img_np  = img.permute(1, 2, 0).numpy()   # HWC
            mask_np = mask.squeeze(0).numpy()          # HW
            out = GEO_AUGMENTER(image=img_np, mask=mask_np)
            img  = torch.from_numpy(out['image']).permute(2, 0, 1).contiguous().float()
            mask = torch.from_numpy(out['mask']).unsqueeze(0).contiguous().float()
            mask = (mask > 0.5).float()   # re-binarize after any interpolation
        else:
            if torch.rand(1) > 0.5:
                img, mask = torch.flip(img, dims=[2]), torch.flip(mask, dims=[2])
            if torch.rand(1) > 0.5:
                img, mask = torch.flip(img, dims=[1]), torch.flip(mask, dims=[1])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                img, mask = torch.rot90(img, k, dims=[1, 2]), torch.rot90(mask, k, dims=[1, 2])

        # ---- Pixel-level augmentation (hand-rolled, see module docstring) ----
        img = self._pixel_augment(img)
        return img, mask

    def _pixel_augment(self, img):
        # Calibrated for standardized data (mean~0, std~1 per channel).
        if torch.rand(1).item() < 0.3:
            noise_std = 0.05 + torch.rand(1).item() * 0.10
            img = img + torch.randn_like(img) * noise_std

        if torch.rand(1).item() < 0.2:
            k = int(torch.randint(1, 3, (1,)).item()) * 2 + 1   # 3 or 5
            img = self._gaussian_blur(img, kernel_size=k)

        if torch.rand(1).item() < 0.3:
            gamma  = 0.7 + torch.rand(1).item() * 0.6   # [0.7, 1.3]
            c_min  = img.amin(dim=(1, 2), keepdim=True)
            c_max  = img.amax(dim=(1, 2), keepdim=True)
            denom  = (c_max - c_min).clamp_min(1e-6)
            unit   = ((img - c_min) / denom).clamp(0, 1) ** gamma
            img    = unit * denom + c_min

        if torch.rand(1).item() < 0.5:
            delta = torch.rand(1).item() * 0.2 - 0.1
            img   = img + delta

        if torch.rand(1).item() < 0.5:
            factor = torch.rand(1).item() * 0.2 + 0.9
            mean_c = img.mean(dim=(1, 2), keepdim=True)
            img    = (img - mean_c) * factor + mean_c

        return img

    @staticmethod
    def _gaussian_blur(img, kernel_size=3, sigma=1.0):
        C = img.shape[0]
        coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).to(img.dtype)
        kernel_x = g.view(1, 1, 1, kernel_size).repeat(C, 1, 1, 1)
        kernel_y = g.view(1, 1, kernel_size, 1).repeat(C, 1, 1, 1)
        pad = kernel_size // 2
        x = img.unsqueeze(0)
        x = F.conv2d(x, kernel_x, padding=(0, pad), groups=C)
        x = F.conv2d(x, kernel_y, padding=(pad, 0), groups=C)
        return x.squeeze(0)


# -------------------------------------------------
# SEGMENTATION HEAD
# DINOv2 has no dense-prediction head, so this small conv stack turns the
# (B, hidden, H/14, W/14) patch-token grid into (B, num_labels, H/14, W/14)
# logits, which the model's forward() then bilinear-upsamples to full
# input resolution. Attribute names (bn1/bn2) are kept regardless of the
# underlying norm type so state_dict keys stay stable either way.
# -------------------------------------------------
def _make_norm2d(norm_type, channels, max_groups=32):
    if norm_type == 'group':
        groups = max_groups
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    return nn.BatchNorm2d(channels)


class Dinov2SegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_labels=1, dropout=0.1,
                 norm_type='group'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = _make_norm2d(norm_type, hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = _make_norm2d(norm_type, hidden_channels)
        self.dropout    = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden_channels, num_labels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = self.dropout(x)
        return self.classifier(x)


# -------------------------------------------------
# MODEL — returns raw LOGITS (B, 1, H, W), no sigmoid.
# -------------------------------------------------
class Dinov2Solar(nn.Module):
    def __init__(self, in_channels=4, num_labels=1, backbone='facebook/dinov2-small',
                 pretrained=True, image_size=126, head_channels=256, freeze_backbone=False,
                 head_norm='group'):
        super().__init__()
        # AutoConfig/AutoModel (rather than hardcoding Dinov2Model/Dinov2Config) so that
        # both plain DINOv2 checkpoints and "-with-registers" variants — which HuggingFace
        # implements as a distinct model class — work through the same code path.
        from transformers import AutoConfig, AutoModel

        if pretrained:
            self.backbone = AutoModel.from_pretrained(backbone)
            self._adapt_input_channels(in_channels)
        else:
            config = AutoConfig.from_pretrained(backbone)
            config.num_channels = in_channels
            self.backbone = AutoModel.from_config(config)

        self.patch_size      = getattr(self.backbone.config, 'patch_size', DINOV2_PATCH_SIZE)
        self.hidden_size     = self.backbone.config.hidden_size
        self.image_size      = image_size
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = Dinov2SegHead(self.hidden_size, hidden_channels=head_channels,
                                   num_labels=num_labels, norm_type=head_norm)

    def train(self, mode=True):
        # Keep the frozen backbone in eval() (disables its internal dropout) even when
        # the wrapper module is put in train() mode for the head — standard linear-probe
        # practice, avoids stochastic "frozen" features.
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def set_backbone_frozen(self, frozen: bool):
        """Toggle backbone requires_grad at runtime (for --unfreeze-after-epoch)."""
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        self.freeze_backbone = frozen

    def _adapt_input_channels(self, in_channels):
        """
        Dynamically find the first Conv2d (DINOv2's patch-embedding projection —
        it is the ONLY Conv2d in the whole encoder, everything else is Linear/LayerNorm)
        and rebuild it for `in_channels` input channels. Works across all transformers
        versions and across the whole DINOv2 family (small/base/large/giant, with or
        without register tokens) — no hardcoded attribute paths.
        NOTE: inference_dinov2_solar.py uses this exact same implementation, so the two
        scripts can't silently diverge on a transformers version bump.
        """
        target_name = None
        old_conv    = None

        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                target_name = name
                old_conv    = module
                break

        if old_conv is None:
            raise RuntimeError('Could not find any Conv2d inside the DINOv2 model.')

        if old_conv.in_channels == in_channels:
            return  # nothing to do

        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        with torch.no_grad():
            copy_c = min(3, in_channels, old_conv.in_channels)
            new_conv.weight[:, :copy_c] = old_conv.weight[:, :copy_c]
            if in_channels > copy_c:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                for c in range(copy_c, in_channels):
                    new_conv.weight[:, c:c + 1] = mean_w
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        parts  = target_name.split('.')
        parent = self.backbone
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

        # IMPORTANT: swapping the Conv2d alone is not enough. Dinov2PatchEmbeddings.forward()
        # validates pixel_values.shape[1] against a `self.num_channels` attribute that was
        # copied from config at __init__ time — it does NOT re-derive this from the conv
        # layer's actual in_channels. Left unpatched, this raises "Expected 3 but got 4"
        # even though the conv itself now accepts 4 channels. Update both the module
        # attribute (what forward() actually checks) and config (for consistency/exports).
        if hasattr(parent, 'num_channels'):
            parent.num_channels = in_channels
        self.backbone.config.num_channels = in_channels

        print(f'  [channel adapt] {target_name}: {old_conv.in_channels}ch -> {in_channels}ch')

    def forward(self, x):
        B, C, H, W = x.shape

        # interpolate_pos_encoding lets DINOv2 accept an input resolution different from
        # its pretraining resolution by resampling the position embeddings. Older/newer
        # transformers versions expose this kwarg slightly differently, so fall back
        # gracefully if the installed version doesn't accept it at this level.
        try:
            outputs = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        except TypeError:
            outputs = self.backbone(pixel_values=x)

        tokens = outputs.last_hidden_state          # (B, n_prefix + num_patches, hidden)

        h_p = H // self.patch_size
        w_p = W // self.patch_size
        num_patches = h_p * w_p

        # Patch tokens are always the LAST `num_patches` tokens in the sequence, regardless
        # of how many prefix tokens (CLS, and — for "-with-registers" checkpoints — register
        # tokens) come before them. Slicing from the end is robust across the whole family.
        patch_tokens = tokens[:, -num_patches:, :]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, self.hidden_size, h_p, w_p)

        logits = self.head(patch_tokens)             # (B, num_labels, h_p, w_p)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return logits   # raw logits — NOT sigmoided


def build_model(backbone, pretrained, image_size, in_channels, num_labels,
                 head_channels, freeze_backbone, head_norm='group'):
    model = Dinov2Solar(
        in_channels=in_channels, num_labels=num_labels, backbone=backbone,
        pretrained=pretrained, image_size=image_size,
        head_channels=head_channels, freeze_backbone=freeze_backbone,
        head_norm=head_norm
    ).to(DEVICE)

    # [OPT: channels_last] See CHANNELS_LAST comment above. Only affects the
    # model's 4-D conv weights; try/except keeps this fully optional in case a
    # future backbone variant has a conv layer that rejects the conversion.
    if CHANNELS_LAST:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:
            print(f'[channels_last] Could not convert model ({e}); continuing without it.')

    if GRADIENT_CHECKPOINTING and not freeze_backbone:
        if hasattr(model.backbone, 'gradient_checkpointing_enable'):
            model.backbone.gradient_checkpointing_enable()
            print('[memory] Gradient checkpointing enabled on backbone.')
        else:
            print('[memory] --gradient-checkpointing requested but the backbone has no '
                  'gradient_checkpointing_enable() method; ignoring.')

    return model


# [OPT: fused AdamW] The fused kernel fuses the AdamW update into a single
# CUDA kernel launch per param group instead of one launch per tensor —
# meaningfully faster on GPU, especially with many small parameter tensors
# like a ViT's. Not all PyTorch/CUDA/param combinations support it, so this
# tries fused=True first and transparently falls back to the standard
# implementation if that raises.
def _make_adamw(params, weight_decay):
    if torch.cuda.is_available():
        try:
            return torch.optim.AdamW(params, weight_decay=weight_decay, fused=True)
        except (TypeError, RuntimeError) as e:
            print(f'[optim] fused AdamW unavailable ({e}); falling back to standard AdamW.')
    return torch.optim.AdamW(params, weight_decay=weight_decay)


def build_optimizer(model, freeze_backbone, backbone_lr, head_lr, weight_decay=0.05,
                     layer_decay=1.0):
    if freeze_backbone:
        return _make_adamw(
            [{'params': model.head.parameters(), 'lr': head_lr}], weight_decay
        )

    if layer_decay is None or layer_decay >= 1.0:
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': model.head.parameters(),     'lr': head_lr},
        ]
        return _make_adamw(param_groups, weight_decay)

    # ---- Layer-wise LR decay across backbone transformer blocks ----
    try:
        blocks = model.backbone.encoder.layer
        num_layers = len(blocks)
    except AttributeError:
        print('[layer-decay] Could not find backbone.encoder.layer on this model class; '
              'falling back to a single flat backbone LR.')
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': backbone_lr},
            {'params': model.head.parameters(),     'lr': head_lr},
        ]
        return _make_adamw(param_groups, weight_decay)

    by_depth = {}

    def add(depth, p):
        by_depth.setdefault(depth, []).append(p)

    if hasattr(model.backbone, 'embeddings'):
        for p in model.backbone.embeddings.parameters():
            add(0, p)

    for i, layer in enumerate(blocks):
        for p in layer.parameters():
            add(i + 1, p)

    captured = {id(p) for plist in by_depth.values() for p in plist}
    leftover = [p for p in model.backbone.parameters() if id(p) not in captured]
    if leftover:
        by_depth.setdefault(num_layers + 1, []).extend(leftover)

    max_depth = num_layers + 1
    param_groups = [
        {'params': params, 'lr': backbone_lr * (layer_decay ** (max_depth - depth))}
        for depth, params in by_depth.items()
    ]
    param_groups.append({'params': model.head.parameters(), 'lr': head_lr})
    return _make_adamw(param_groups, weight_decay)


def build_scheduler(optimizer, kind, epochs, warmup_epochs, min_lr_ratio):
    """Returns (scheduler, needs_metric) — needs_metric tells the caller whether to
    call scheduler.step(val_iou) (plateau) or scheduler.step() (cosine)."""
    if kind == 'plateau':
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7), True

    warmup_epochs = max(0, min(warmup_epochs, max(epochs - 1, 0)))

    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda), False


# -------------------------------------------------
# LOSS FUNCTIONS — operate on LOGITS.
# -------------------------------------------------
def dice_loss_from_logits(y_true, logits, smooth=1e-6):
    y_pred = torch.sigmoid(logits)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    inter  = torch.sum(y_true * y_pred)
    return 1 - (2 * inter + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)


def focal_loss_from_logits(y_true, logits, gamma=2, alpha=0.25):
    """alpha does real per-pixel class-balancing (alpha for positives, 1-alpha for
    negatives) — down-weights the abundant negative class for a sparse positive class
    like solar panels."""
    y_true_flat = y_true.view(-1)
    logits_flat = logits.view(-1)
    bce = F.binary_cross_entropy_with_logits(logits_flat, y_true_flat, reduction='none')
    p   = torch.sigmoid(logits_flat)
    pt  = torch.where(y_true_flat == 1, p, 1 - p)
    alpha_t = torch.where(y_true_flat == 1,
                          torch.full_like(y_true_flat, alpha),
                          torch.full_like(y_true_flat, 1 - alpha))
    return torch.mean(alpha_t * (1 - pt) ** gamma * bce)


def tversky_loss_from_logits(y_true, logits, alpha=0.7, beta=0.3, smooth=1e-6):
    """alpha weights false negatives, beta weights false positives — alpha=beta=0.5
    reduces to Dice. alpha>beta (default) trades some precision for recall, which
    tends to suit sparse-positive segmentation like solar panels."""
    y_pred = torch.sigmoid(logits).view(-1)
    y_true = y_true.view(-1)
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))
    return 1 - (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_loss_from_logits(y_true, logits):
    """Binary Lovasz hinge loss — a direct, differentiable surrogate for IoU."""
    logits_flat = logits.view(-1)
    labels_flat = y_true.view(-1)
    if labels_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def build_loss_fn(loss_name, pos_weight=None, device=None):
    pw = torch.tensor(pos_weight, device=device) if pos_weight is not None else None
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    def dice_bce(y_true, logits):
        return dice_loss_from_logits(y_true, logits) + bce_fn(logits, y_true)

    def dice_focal_bce(y_true, logits):
        return (dice_loss_from_logits(y_true, logits)
                + focal_loss_from_logits(y_true, logits)
                + 0.1 * bce_fn(logits, y_true))

    def dice_lovasz(y_true, logits):
        return dice_loss_from_logits(y_true, logits) + lovasz_loss_from_logits(y_true, logits)

    def dice_tversky(y_true, logits):
        return dice_loss_from_logits(y_true, logits) + tversky_loss_from_logits(y_true, logits)

    return {
        'dice_bce':       dice_bce,
        'dice_focal_bce': dice_focal_bce,
        'dice_lovasz':    dice_lovasz,
        'dice_tversky':   dice_tversky,
    }[loss_name]


# -------------------------------------------------
# METRICS — confusion-count based, so callers can accumulate TP/FP/FN/TN
# across an entire epoch and compute IoU/F1/Dice/precision/recall/
# specificity/balanced-accuracy ONCE at the end, instead of averaging
# per-batch metrics (which biases the result for a sparse positive class).
# -------------------------------------------------
def confusion_counts(y_true, logits, threshold=THRESHOLD):
    y_pred = (torch.sigmoid(logits) > threshold).float()
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))
    tn = torch.sum((1 - y_true) * (1 - y_pred))
    return tp, fp, fn, tn


def metrics_from_counts(tp, fp, fn, tn, eps=1e-6):
    precision    = (tp + eps) / (tp + fp + eps)
    recall       = (tp + eps) / (tp + fn + eps)
    specificity  = (tn + eps) / (tn + fp + eps)
    iou          = (tp + eps) / (tp + fp + fn + eps)
    dice         = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    f1           = 2 * precision * recall / (precision + recall + eps)
    balanced_acc = (recall + specificity) / 2
    return iou, f1, dice, precision, recall, specificity, balanced_acc


# -------------------------------------------------
# TRAIN / VALIDATE  (AMP + gradient clipping)
# -------------------------------------------------
def train_epoch(model, forward_fn, dataloader, optimizer, scaler, epoch, loss_fn):
    model.train()
    total_loss = 0.0
    tp_sum = fp_sum = fn_sum = tn_sum = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [train]')
    for images, masks in pbar:
        # [OPT: non_blocking] Overlaps the H2D copy with prior-batch GPU work when
        # paired with pin_memory=True on the DataLoader (set below).
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)
        if CHANNELS_LAST and DEVICE.type == 'cuda':
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = forward_fn(images)   # [OPT: torch.compile] compiled-or-eager, see main()
            loss   = loss_fn(masks, logits)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            tp, fp, fn, tn = confusion_counts(masks, logits, THRESHOLD)
        tp_i, fp_i, fn_i, tn_i = tp.item(), fp.item(), fn.item(), tn.item()
        tp_sum += tp_i
        fp_sum += fp_i
        fn_sum += fn_i
        tn_sum += tn_i
        total_loss += loss.item()

        # Per-batch numbers shown on the progress bar are just live feedback —
        # the numbers that actually get logged/checkpointed are the
        # epoch-accumulated ones computed below, after the loop.
        b_iou, b_f1, b_dice, _, _, _, _ = metrics_from_counts(tp_i, fp_i, fn_i, tn_i)
        pbar.set_postfix(
            loss=f'{loss.item():.4f}',
            iou=f'{b_iou:.4f}',
            f1=f'{b_f1:.4f}',
            dice=f'{b_dice:.4f}'
        )

    n = len(dataloader)
    metrics = metrics_from_counts(tp_sum, fp_sum, fn_sum, tn_sum)
    return (total_loss / n,) + metrics


def validate_epoch(model, forward_fn, dataloader, loss_fn):
    model.eval()
    total_loss = 0.0
    tp_sum = fp_sum = fn_sum = tn_sum = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)
            if CHANNELS_LAST and DEVICE.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = forward_fn(images)
                loss   = loss_fn(masks, logits)

            total_loss += loss.item()
            tp, fp, fn, tn = confusion_counts(masks, logits, THRESHOLD)
            tp_sum += tp.item()
            fp_sum += fp.item()
            fn_sum += fn.item()
            tn_sum += tn.item()

    n = len(dataloader)
    metrics = metrics_from_counts(tp_sum, fp_sum, fn_sum, tn_sum)
    return (total_loss / n,) + metrics


# -------------------------------------------------
# OPTIMAL THRESHOLD SEARCH — sweeps a grid of thresholds over the validation
# set (reusing the already-computed sigmoid probabilities per batch) and
# picks the threshold that maximizes F1. Writes the result so inference can
# use it. Grid is now 0.01..0.99 step 0.01 (was 0.05..0.95 step 0.02).
# -------------------------------------------------
def search_optimal_threshold(model, forward_fn, dataloader, device, use_amp,
                              thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 4)

    model.eval()
    tp_sums = np.zeros(len(thresholds))
    fp_sums = np.zeros(len(thresholds))
    fn_sums = np.zeros(len(thresholds))

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Threshold search'):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            if CHANNELS_LAST and device.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = forward_fn(images)
            probs = torch.sigmoid(logits).float()
            masks = masks.float()

            for i, t in enumerate(thresholds):
                pred = (probs > t).float()
                tp_sums[i] += torch.sum(masks * pred).item()
                fp_sums[i] += torch.sum((1 - masks) * pred).item()
                fn_sums[i] += torch.sum(masks * (1 - pred)).item()

    eps = 1e-6
    precision = (tp_sums + eps) / (tp_sums + fp_sums + eps)
    recall    = (tp_sums + eps) / (tp_sums + fn_sums + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    best_idx = int(np.argmax(f1))
    return {
        'optimal_threshold': float(thresholds[best_idx]),
        'f1':        float(f1[best_idx]),
        'precision': float(precision[best_idx]),
        'recall':    float(recall[best_idx]),
    }


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == '__main__':

    print_hardware_info(header="GPU / HARDWARE INFO")
    print(f'Model type   : DINOv2')
    print(f'Model name   : {model_name}')
    print(f'Backbone     : {BACKBONE}  (pretrained={PRETRAINED}, '
          f'frozen={FREEZE_BACKBONE}, unfreeze_after_epoch={UNFREEZE_AFTER_EPOCH})')
    print(f'Image size   : {IMAGE_SIZE}  (patch={DINOV2_PATCH_SIZE})')
    print(f'AMP          : {USE_AMP}')
    print(f'LRs          : backbone={BACKBONE_LR}  head={HEAD_LR}  '
          f'layer_decay={LAYER_DECAY}  weight_decay={WEIGHT_DECAY}')
    print(f'Loss         : {LOSS_NAME}  (auto_pos_weight={AUTO_POS_WEIGHT})')
    print(f'Scheduler    : {SCHEDULER_KIND}'
          + (f'  (warmup_epochs={WARMUP_EPOCHS}, min_lr_ratio={MIN_LR_RATIO})'
             if SCHEDULER_KIND == 'cosine' else ''))
    print(f'Head norm    : {HEAD_NORM}')
    print(f'Threshold    : {THRESHOLD}  (training-time metric threshold; '
          f'a separate optimal threshold is searched after training)')

    all_img_paths = sorted(glob.glob(f'{DATA_PATH}/images/*.tif'))
    if len(all_img_paths) == 0:
        raise FileNotFoundError(
            f'No .tif files found in {DATA_PATH}/images/  '
            'Check your DATA_PATH and folder structure.'
        )
    print(f'Total images : {len(all_img_paths)}')

    train_paths, val_paths = split_train_val(
        all_img_paths, SPLIT_BY, SCENE_REGEX, test_size=0.2, seed=SEED
    )
    print(f'Train: {len(train_paths)}  |  Val: {len(val_paths)}')

    # [OPT: DataLoader] More workers + deeper prefetch queue keeps a fast GPU fed;
    # capped at 16 since going higher rarely helps and just adds process overhead.
    num_workers = min(16, os.cpu_count() or 8)

    # ---- Resume logic ----
    start_epoch        = 1
    best_val_iou        = 0.0
    backbone_unfrozen   = not (FREEZE_BACKBONE or UNFREEZE_AFTER_EPOCH > 0)
    resume_ckpt         = LAST_MODEL if os.path.exists(LAST_MODEL) else None

    if resume_ckpt:
        # Full checkpoint dict (model/optimizer/scheduler/scaler/mean/std/epoch) —
        # weights_only=True is only for pure tensor state_dicts and breaks this,
        # so we load the full pickle instead. Only load checkpoints you trust.
        ckpt = torch.load(resume_ckpt, map_location=DEVICE, weights_only=False)

        MEAN = ckpt['mean']
        STD  = ckpt['std']

        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE) as f:
                saved_cfg = json.load(f)
            BACKBONE             = saved_cfg.get('backbone',              BACKBONE)
            PRETRAINED           = saved_cfg.get('pretrained',            PRETRAINED)
            IMAGE_SIZE           = saved_cfg.get('image_size',            IMAGE_SIZE)
            HEAD_CHANNELS        = saved_cfg.get('head_channels',         HEAD_CHANNELS)
            HEAD_NORM            = saved_cfg.get('head_norm',             HEAD_NORM)
            FREEZE_BACKBONE      = saved_cfg.get('freeze_backbone',       FREEZE_BACKBONE)
            UNFREEZE_AFTER_EPOCH = saved_cfg.get('unfreeze_after_epoch',  UNFREEZE_AFTER_EPOCH)
            BACKBONE_LR          = saved_cfg.get('backbone_lr',           BACKBONE_LR)
            HEAD_LR              = saved_cfg.get('head_lr',               HEAD_LR)
            WEIGHT_DECAY         = saved_cfg.get('weight_decay',          WEIGHT_DECAY)
            LAYER_DECAY          = saved_cfg.get('layer_decay',           LAYER_DECAY)
            LOSS_NAME            = saved_cfg.get('loss',                  LOSS_NAME)
            SCHEDULER_KIND       = saved_cfg.get('scheduler',             SCHEDULER_KIND)
            WARMUP_EPOCHS        = saved_cfg.get('warmup_epochs',         WARMUP_EPOCHS)
            MIN_LR_RATIO         = saved_cfg.get('min_lr_ratio',          MIN_LR_RATIO)
        POS_WEIGHT = saved_cfg.get('pos_weight', None) if os.path.exists(CONFIG_FILE) else None

        # Reconstruct with the SAME initial (pre-unfreeze) structure the checkpoint's
        # optimizer state was saved with, then replay any mid-run unfreeze below —
        # this keeps the rebuilt optimizer's param-group structure aligned with
        # ckpt['optimizer'] before load_state_dict() is called.
        initial_frozen = FREEZE_BACKBONE or (UNFREEZE_AFTER_EPOCH > 0)
        model = build_model(BACKBONE, PRETRAINED, IMAGE_SIZE, in_channels=4, num_labels=1,
                             head_channels=HEAD_CHANNELS, freeze_backbone=initial_frozen,
                             head_norm=HEAD_NORM)

        optimizer = build_optimizer(model, initial_frozen, BACKBONE_LR, HEAD_LR,
                                     WEIGHT_DECAY, LAYER_DECAY)

        # If the backbone was unfrozen mid-run in a previous session, replay that here
        # (before loading optimizer state) so the param-group structure matches.
        backbone_unfrozen = ckpt.get('backbone_unfrozen', not initial_frozen)
        if (not FREEZE_BACKBONE) and UNFREEZE_AFTER_EPOCH > 0 and backbone_unfrozen and initial_frozen:
            model.set_backbone_frozen(False)
            optimizer.add_param_group({
                'params': [p for p in model.backbone.parameters() if p.requires_grad],
                'lr': BACKBONE_LR,
            })

        scheduler, scheduler_needs_metric = build_scheduler(
            optimizer, SCHEDULER_KIND, EPOCHS, WARMUP_EPOCHS, MIN_LR_RATIO
        )
        scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_iou = ckpt['best_iou']
        print(f'Resumed from checkpoint at epoch {ckpt["epoch"]} (best_iou={best_val_iou:.4f}).')
    else:
        MEAN, STD = compute_mean_std(train_paths)
        save_mean_std(MEAN, STD)

        POS_WEIGHT = compute_pos_weight(train_paths) if AUTO_POS_WEIGHT else None

        initial_frozen = FREEZE_BACKBONE or (UNFREEZE_AFTER_EPOCH > 0)
        model = build_model(BACKBONE, PRETRAINED, IMAGE_SIZE, in_channels=4, num_labels=1,
                             head_channels=HEAD_CHANNELS, freeze_backbone=initial_frozen,
                             head_norm=HEAD_NORM)
        optimizer = build_optimizer(model, initial_frozen, BACKBONE_LR, HEAD_LR,
                                     WEIGHT_DECAY, LAYER_DECAY)
        scheduler, scheduler_needs_metric = build_scheduler(
            optimizer, SCHEDULER_KIND, EPOCHS, WARMUP_EPOCHS, MIN_LR_RATIO
        )
        scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                'backbone':               BACKBONE,
                'pretrained':             PRETRAINED,
                'in_channels':            4,
                'num_labels':             1,
                'image_size':             IMAGE_SIZE,
                'patch_size':             DINOV2_PATCH_SIZE,
                'head_channels':          HEAD_CHANNELS,
                'head_norm':              HEAD_NORM,
                'freeze_backbone':        FREEZE_BACKBONE,
                'unfreeze_after_epoch':   UNFREEZE_AFTER_EPOCH,
                'backbone_lr':            BACKBONE_LR,
                'head_lr':                HEAD_LR,
                'weight_decay':           WEIGHT_DECAY,
                'layer_decay':            LAYER_DECAY,
                'loss':                   LOSS_NAME,
                'pos_weight':             POS_WEIGHT,
                'scheduler':              SCHEDULER_KIND,
                'warmup_epochs':          WARMUP_EPOCHS,
                'min_lr_ratio':           MIN_LR_RATIO,
                'split_by':               SPLIT_BY,
                'gradient_checkpointing': GRADIENT_CHECKPOINTING,
            }, f, indent=2)
        print('New model initialised.')

    LOSS_FN = build_loss_fn(LOSS_NAME, POS_WEIGHT, DEVICE)

    # ---- Rebuild GEO_AUGMENTER against the FINAL IMAGE_SIZE ----
    # IMAGE_SIZE may have just been overwritten above by a resumed run's saved
    # dinov2_config.json (which can differ from the --image-size passed on this
    # invocation's command line). GEO_AUGMENTER was originally built at module
    # import time against the CLI's IMAGE_SIZE, so its RandomResizedCrop target
    # size would silently go stale on a resume that changes image size. Rebuild
    # it now, after IMAGE_SIZE is fully resolved and before any dataset/loader
    # reads it.
    GEO_AUGMENTER = build_geo_augmenter(IMAGE_SIZE, enable_elastic=ELASTIC_AUGMENT)

    # ---- Datasets & loaders ----
    train_dataset = SolarDataset(train_paths, MEAN, STD, is_train=True)
    val_dataset   = SolarDataset(val_paths,   MEAN, STD, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )

    # [OPT: torch.compile] Compile a *separate* callable for the forward pass —
    # `model` itself stays the plain nn.Module everywhere else (checkpointing,
    # optimizer construction, .train()/.eval(), grad clipping), so state_dict()
    # keys and the checkpoint format are byte-for-byte unchanged; inference
    # doesn't need to know or care that compile was used during training.
    # forward_model() tries the compiled graph and permanently falls back to
    # eager execution (for the rest of this run) the first time it errors —
    # covers unsupported ops, Windows/no-Triton environments, etc.
    _compile_state = {'enabled': hasattr(torch, 'compile') and DEVICE.type == 'cuda'}
    if _compile_state['enabled']:
        try:
            _compiled_model = torch.compile(model)
            print('[compile] torch.compile enabled for the training forward pass.')
        except Exception as e:
            print(f'[compile] torch.compile init failed ({e}); using eager mode.')
            _compiled_model = model
            _compile_state['enabled'] = False
    else:
        _compiled_model = model
        if DEVICE.type == 'cuda':
            print('[compile] torch.compile not available in this PyTorch build; using eager mode.')

    def forward_model(x):
        if _compile_state['enabled']:
            try:
                return _compiled_model(x)
            except Exception as e:
                print(f'[compile] compiled forward failed at runtime ({e}); disabling '
                      'torch.compile for the remainder of this run.')
                _compile_state['enabled'] = False
        return model(x)

    # ---- Training loop ----
    patience_counter    = 0
    early_stop_patience = EARLY_STOP_PATIENCE

    def make_checkpoint(epoch, best_iou_value):
        return {
            'epoch':              epoch,
            'model':              model.state_dict(),
            'optimizer':          optimizer.state_dict(),
            'scheduler':          scheduler.state_dict(),
            'scaler':             scaler.state_dict(),
            'best_iou':           best_iou_value,
            'mean':               MEAN,
            'std':                STD,
            'backbone_unfrozen':  backbone_unfrozen,
        }

    training_start = time.time()
    epoch_times    = []

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        # ---- Dynamic unfreeze (--unfreeze-after-epoch) ----
        if ((not FREEZE_BACKBONE) and UNFREEZE_AFTER_EPOCH > 0
                and not backbone_unfrozen and epoch > UNFREEZE_AFTER_EPOCH):
            model.set_backbone_frozen(False)
            optimizer.add_param_group({
                'params': [p for p in model.backbone.parameters() if p.requires_grad],
                'lr': BACKBONE_LR,
            })
            backbone_unfrozen = True
            print(f'  -> Backbone unfrozen at epoch {epoch} (added to optimizer at '
                  f'lr={BACKBONE_LR}).')

        train_loss, train_iou, train_f1, train_dice, train_prec, train_rec, \
            train_spec, train_bacc = train_epoch(
                model, forward_model, train_loader, optimizer, scaler, epoch, LOSS_FN
            )
        val_loss, val_iou, val_f1, val_dice, val_prec, val_rec, \
            val_spec, val_bacc = validate_epoch(model, forward_model, val_loader, LOSS_FN)

        duration = time.time() - t0
        epoch_times.append(duration)

        log_line = (
            f'Epoch:{epoch} | time:{duration:.1f}s | '
            f'loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | '
            f'iou:{train_iou:.4f} | val_iou:{val_iou:.4f} | '
            f'f1:{train_f1:.4f} | val_f1:{val_f1:.4f} | '
            f'dice:{train_dice:.4f} | val_dice:{val_dice:.4f}'
        )
        print(log_line)
        print(f'  val_precision:{val_prec:.4f}  val_recall:{val_rec:.4f}  '
              f'val_specificity:{val_spec:.4f}  val_balanced_acc:{val_bacc:.4f}')

        with open(TRAIN_LOG, 'a') as f:
            f.write(log_line + '\n')

        append_csv_row({
            'epoch': epoch, 'time_s': round(duration, 1),
            'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6),
            'train_iou': round(float(train_iou), 6),   'val_iou': round(float(val_iou), 6),
            'train_f1': round(float(train_f1), 6),     'val_f1': round(float(val_f1), 6),
            'train_dice': round(float(train_dice), 6), 'val_dice': round(float(val_dice), 6),
            'train_precision': round(float(train_prec), 6), 'val_precision': round(float(val_prec), 6),
            'train_recall': round(float(train_rec), 6),     'val_recall': round(float(val_rec), 6),
            'train_specificity': round(float(train_spec), 6), 'val_specificity': round(float(val_spec), 6),
            'train_balanced_acc': round(float(train_bacc), 6), 'val_balanced_acc': round(float(val_bacc), 6),
        })

        if scheduler_needs_metric:
            scheduler.step(val_iou)
        else:
            scheduler.step()

        # Always save "last" (self-contained: includes mean/std) so resume never loses progress
        torch.save(make_checkpoint(epoch, max(best_val_iou, val_iou)), LAST_MODEL)

        if val_iou > best_val_iou:
            best_val_iou     = val_iou
            patience_counter = 0
            torch.save(make_checkpoint(epoch, best_val_iou), BEST_MODEL)
            print(f'  -> Best model saved (val_iou: {val_iou:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f'Early stopping triggered after epoch {epoch}.')
            break

    print('Training finished.')

    # ======================================================
    # TRAINING TIMING SUMMARY
    # ======================================================
    total_training_time = time.time() - training_start
    avg_epoch_time       = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0

    hw_lines = print_hardware_info(header='FINAL GPU / HARDWARE INFO')

    print(f'\n{"=" * 60}')
    print(f'TRAINING TIMING SUMMARY')
    print(f'{"=" * 60}')
    print(f'  Model type       : DINOv2 (backbone={BACKBONE})')
    print(f'  Model name       : {model_name}')
    print(f'  Hardware         : {hardware_summary_line()}')
    print(f'  Epochs run       : {len(epoch_times)}')
    print(f'  Avg epoch time   : {avg_epoch_time:.2f}s')
    print(f'  Total train time : {total_training_time:.2f}s '
          f'({total_training_time / 60:.1f} min)')
    print(f'{"=" * 60}\n')

    with open(TRAIN_LOG, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('TRAINING SUMMARY\n')
        f.write('=' * 60 + '\n')
        f.write(f'Model type          : DINOv2 (backbone={BACKBONE})\n')
        f.write(f'Model name          : {model_name}\n')
        f.write(f'Hardware            : {hardware_summary_line()}\n')
        for line in hw_lines:
            f.write(f'  {line}\n')
        f.write(f'Epochs run          : {len(epoch_times)}\n')
        f.write(f'Average epoch time  : {avg_epoch_time:.2f}s\n')
        f.write(f'Total training time : {total_training_time:.2f}s\n')
        f.write('=' * 60 + '\n')

    # ======================================================
    # POST-TRAINING EXPORT
    # 1) Load the BEST checkpoint's weights into `model`, then save a
    #    clean weights_only=True-loadable state_dict as "<model_name>.pth"
    #    — this is the exact file inference_dinov2_solar.py looks for.
    # 2) Run the threshold search on the val set using that same best
    #    model, and write optimal_threshold.json + update
    #    dinov2_config.json so inference finds a real value instead of
    #    always falling back to 0.5.
    # ======================================================
    best_source = BEST_MODEL if os.path.exists(BEST_MODEL) else LAST_MODEL
    print(f'\nExporting inference artifacts from: {best_source}')

    best_ckpt = torch.load(best_source, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    model.eval()

    torch.save(model.state_dict(), INFERENCE_WEIGHTS_FILE)
    print(f'  Inference weights : {INFERENCE_WEIGHTS_FILE}')

    threshold_result = search_optimal_threshold(model, forward_model, val_loader, DEVICE, USE_AMP)
    threshold_result['source_checkpoint'] = os.path.basename(best_source)
    threshold_result['source_epoch']      = best_ckpt.get('epoch')

    with open(OPTIMAL_THRESHOLD_FILE, 'w') as f:
        json.dump(threshold_result, f, indent=2)
    print(f'  Optimal threshold : {threshold_result["optimal_threshold"]:.4f}  '
          f'(F1={threshold_result["f1"]:.4f}, '
          f'P={threshold_result["precision"]:.4f}, '
          f'R={threshold_result["recall"]:.4f})')
    print(f'  Threshold file    : {OPTIMAL_THRESHOLD_FILE}')

    # Also embed into dinov2_config.json for redundancy (inference checks this
    # as a fallback if the dedicated threshold files are missing).
    with open(CONFIG_FILE) as f:
        cfg_data = json.load(f)
    cfg_data['optimal_threshold'] = threshold_result['optimal_threshold']
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg_data, f, indent=2)

    print('\nAll set — MODEL_DIR for inference should point at:')
    print(f'  {OUTPUT_FOLDER}')
    print(f'MODEL_NAME for inference should be:')
    print(f'  {model_name}')