# =========================================================
# SOLAR FARM EXTRACTION - SEGFORMER VERSION (PyTorch / HuggingFace)
#
# USAGE:
#   python train_segformer_solar.py <DATA_PATH> <OUTPUT_FOLDER> <MODEL_NAME>
#       [--backbone D:\mit-b0-local] [--no-pretrained] [--epochs 200]
#       [--batch-size 8] [--image-size 128] [--no-amp] [--threshold 0.5]
# =========================================================
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
import os
import sys
import csv
import glob
import time
import json
import platform
import subprocess
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import rasterio
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# -------------------------------------------------
# CLI ARGS
# -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('data_path')
parser.add_argument('output_folder')
parser.add_argument('model_name')
parser.add_argument('--backbone', default=r'D:\mit-b0-local',
                    help='HuggingFace SegFormer backbone name or local path '
                         '(default: D:\\mit-b0-local). Pass --backbone to override, '
                         'e.g. nvidia/mit-b0 or another local path.')
parser.add_argument('--no-pretrained', action='store_true',
                    help='Randomly initialise the backbone (no ImageNet weights)')
parser.add_argument('--epochs',     type=int, default=200)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--image-size', type=int, default=128)
parser.add_argument('--no-amp', action='store_true',
                    help='Disable mixed precision (AMP) training')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Probability threshold used for IoU/F1/Dice/precision/recall during training')
args = parser.parse_args()

# [PERF #3] Detect whether --batch-size was explicitly supplied (vs just
# picking up argparse's default of 8), so autodetection never overrides
# a value the user actually asked for.
BATCH_SIZE_EXPLICIT = any(
    a == '--batch-size' or a.startswith('--batch-size=') for a in sys.argv[1:]
)

IMAGE_SIZE    = args.image_size
BATCH_SIZE    = args.batch_size
EPOCHS        = args.epochs
BACKBONE      = args.backbone
PRETRAINED    = not args.no_pretrained
USE_AMP       = not args.no_amp
THRESHOLD     = args.threshold
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PIN_MEMORY    = torch.cuda.is_available()

DATA_PATH     = args.data_path
OUTPUT_FOLDER = args.output_folder
model_name    = args.model_name
BEST_MODEL    = os.path.join(OUTPUT_FOLDER, model_name + '_best.pth')
LAST_MODEL    = os.path.join(OUTPUT_FOLDER, model_name + '_last.pth')
MEAN_STD_FILE = os.path.join(OUTPUT_FOLDER, 'normalization_values.txt')
TRAIN_LOG     = os.path.join(OUTPUT_FOLDER, 'training_log.txt')
TRAIN_LOG_CSV = os.path.join(OUTPUT_FOLDER, 'training_log.csv')
CONFIG_FILE   = os.path.join(OUTPUT_FOLDER, 'segformer_config.json')

# This is the exact filename inference_segformer_solar.py expects
# (MODEL_PATH = MODEL_FOLDER/<model_name>.pth). Written at the very end of
# training as a clean, weights_only-loadable state_dict.
INFERENCE_WEIGHTS_FILE = os.path.join(OUTPUT_FOLDER, model_name + '.pth')
# Threshold search output, read by inference's load_optimal_threshold().
OPTIMAL_THRESHOLD_FILE = os.path.join(OUTPUT_FOLDER, 'optimal_threshold.json')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

CSV_FIELDS = ['epoch', 'time_s',
              'train_loss', 'val_loss',
              'train_iou', 'val_iou',
              'train_f1', 'val_f1',
              'train_dice', 'val_dice']


def append_csv_row(row: dict):
    write_header = not os.path.exists(TRAIN_LOG_CSV)
    with open(TRAIN_LOG_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ==========================================================
# [PERF] HARDWARE / CAPABILITY DETECTION
# All checks are best-effort and never raise — anything unavailable
# just gets skipped and the corresponding v4 behavior is used instead.
# ==========================================================
def detect_capabilities(device):
    caps = {
        'device':        str(device),
        'cuda':          device.type == 'cuda',
        'torch_version': torch.__version__,
        'compile':       hasattr(torch, 'compile'),
        'fused_adamw':   False,
        'sdpa':          False,
        'bf16':          False,
    }
    if device.type == 'cuda':
        try:
            caps['gpu_name'] = torch.cuda.get_device_name(0)
            caps['vram_gb']  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass
        try:
            caps['bf16'] = torch.cuda.is_bf16_supported()
        except Exception:
            pass
        # fused AdamW support depends on PyTorch build + device; the only
        # reliable check is trying it, done later at optimizer construction.
        caps['fused_adamw'] = True
        try:
            import transformers  # noqa: F401
            caps['sdpa'] = True  # actual support is probed per-model at construction
        except Exception:
            pass
    return caps


def maybe_compile(model, mode='default'):
    # torch.compile requires Triton which is not supported on Windows
    # Disabled to use eager mode instead
    return model, False


def make_optimizer(params, lr, weight_decay, device):
    """[PERF #2] Try fused AdamW first, fall back to standard AdamW."""
    if device.type == 'cuda':
        try:
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
            return opt, True
        except (TypeError, RuntimeError) as e:
            print(f"WARNING: fused AdamW unavailable ({e}); using standard AdamW.")
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay), False


def autodetect_batch_size(forward_fn, loss_fn, device, image_size,
                           start_bs, max_bs=256, safety=0.85):
    """
    [PERF #3] Probe increasing batch sizes with a real forward+backward pass
    shaped like actual training batches (4-channel image in, 1-channel mask
    target), and return the largest power-of-two-scaled batch size that
    stays under `safety` fraction of total GPU memory. Falls back to
    `start_bs` on CPU or on any failure — never raises.
    """
    if device.type != 'cuda':
        return start_bs

    best = start_bs
    try:
        total_mem = torch.cuda.get_device_properties(device).total_memory
        bs = start_bs
        while bs <= max_bs:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

                dummy_img  = torch.randn(bs, 4, image_size, image_size, device=device)
                dummy_mask = torch.randint(0, 2, (bs, 1, image_size, image_size),
                                            device=device).float()

                with torch.autocast(device_type='cuda', enabled=USE_AMP):
                    out  = forward_fn(dummy_img)
                    loss = loss_fn(dummy_mask, out)
                loss.backward()

                peak = torch.cuda.max_memory_allocated(device)

                del dummy_img, dummy_mask, out, loss
                for p in [p for p in forward_fn.parameters()] if hasattr(forward_fn, 'parameters') else []:
                    if p.grad is not None:
                        p.grad = None
                torch.cuda.empty_cache()

                if peak > total_mem * safety:
                    break
                best = bs
                bs *= 2
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
    except Exception as e:
        print(f"WARNING: batch-size autodetection failed ({e}); using {start_bs}.")
        return start_bs

    return best


# -------------------------------------------------
# MEAN / STD
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
        if torch.rand(1) > 0.5:
            img  = torch.flip(img,  dims=[2])
            mask = torch.flip(mask, dims=[2])

        if torch.rand(1) > 0.5:
            img  = torch.flip(img,  dims=[1])
            mask = torch.flip(mask, dims=[1])

        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img  = torch.rot90(img,  k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        if torch.rand(1) > 0.5:
            delta = torch.rand(1).item() * 0.2 - 0.1
            img   = img + delta

        if torch.rand(1) > 0.5:
            factor = torch.rand(1).item() * 0.2 + 0.9
            img    = img * factor

        return img, mask


# -------------------------------------------------
# MODEL — returns raw LOGITS (B, 1, H, W), no sigmoid.
# -------------------------------------------------
class SegFormerSolar(nn.Module):
    def __init__(self, in_channels=4, num_labels=1, backbone='nvidia/mit-b0',
                 pretrained=True, image_size=128):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            # [PERF #5] Request SDPA / Flash-Attention backed attention when
            # the installed `transformers` version supports the kwarg; older
            # versions raise TypeError on the unknown kwarg, so we retry
            # without it and silently keep the default (eager) attention.
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    backbone, num_labels=num_labels, ignore_mismatched_sizes=True,
                    attn_implementation='sdpa'
                )
            except TypeError:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    backbone, num_labels=num_labels, ignore_mismatched_sizes=True
                )
            self._adapt_input_channels(in_channels)
        else:
            config = SegformerConfig.from_pretrained(backbone)
            config.num_channels = in_channels
            config.num_labels   = num_labels
            try:
                config._attn_implementation = 'sdpa'
            except Exception:
                pass
            try:
                self.model = SegformerForSemanticSegmentation(config)
            except Exception:
                # In case setting _attn_implementation directly ever breaks
                # construction on some transformers version, fall back clean.
                config._attn_implementation = 'eager'
                self.model = SegformerForSemanticSegmentation(config)

        self.image_size = image_size

    def _adapt_input_channels(self, in_channels):
        """
        Dynamically find the first Conv2d (the patch-embedding projection)
        and rebuild it for `in_channels` input channels.
        Works across all transformers versions — no hardcoded attribute paths.
        NOTE: inference_segformer_solar.py uses this exact same
        implementation, so the two scripts can't silently diverge.
        """
        target_name = None
        old_conv    = None

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                target_name = name
                old_conv    = module
                break

        if old_conv is None:
            raise RuntimeError('Could not find any Conv2d inside the SegFormer model.')

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
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

        print(f'  [channel adapt] {target_name}: {old_conv.in_channels}ch -> {in_channels}ch')

    def forward(self, x):
        logits = self.model(pixel_values=x).logits          # (B, 1, H/4, W/4)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode='bilinear', align_corners=False
        )
        return logits   # raw logits — NOT sigmoided


# -------------------------------------------------
# LOSS FUNCTIONS — operate on LOGITS.
# -------------------------------------------------
bce_logits      = nn.BCEWithLogitsLoss()
bce_logits_none = nn.BCEWithLogitsLoss(reduction='none')


def dice_loss_from_logits(y_true, logits, smooth=1e-6):
    y_pred = torch.sigmoid(logits)
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    inter  = torch.sum(y_true * y_pred)
    return 1 - (2 * inter + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)


def focal_loss_from_logits(y_true, logits, gamma=2, alpha=0.25):
    """
    alpha weights positives and negatives differently
    (alpha_t = alpha where y==1, else 1-alpha), which down-weights the
    abundant negative class for a sparse positive class like solar panels.
    """
    y_true_flat = y_true.view(-1)
    logits_flat = logits.view(-1)
    bce = bce_logits_none(logits_flat, y_true_flat)
    p   = torch.sigmoid(logits_flat)
    pt  = torch.where(y_true_flat == 1, p, 1 - p)
    alpha_t = torch.where(y_true_flat == 1,
                          torch.full_like(y_true_flat, alpha),
                          torch.full_like(y_true_flat, 1 - alpha))
    return torch.mean(alpha_t * (1 - pt) ** gamma * bce)


def combined_loss(y_true, logits):
    bce = bce_logits(logits, y_true)
    return dice_loss_from_logits(y_true, logits) + focal_loss_from_logits(y_true, logits) + 0.1 * bce


# -------------------------------------------------
# METRICS — confusion-count based, so callers can accumulate TP/FP/FN
# across an entire epoch and compute IoU/F1/Dice/precision/recall ONCE
# at the end, instead of averaging per-batch metrics (which biases the
# result for a sparse positive class).
# -------------------------------------------------
def confusion_counts(y_true, logits, threshold=THRESHOLD):
    y_pred = (torch.sigmoid(logits) > threshold).float()
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))
    return tp, fp, fn


def metrics_from_counts(tp, fp, fn, eps=1e-6):
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    iou       = (tp + eps) / (tp + fp + fn + eps)
    dice      = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return iou, f1, dice, precision, recall


# -------------------------------------------------
# TRAIN / VALIDATE  (AMP + gradient clipping)
# `model` here is the (possibly torch.compile-wrapped) forward-pass
# handle; callers keep a SEPARATE reference to the raw module for
# checkpointing. `use_channels_last` only takes effect on CUDA.
# -------------------------------------------------
def train_epoch(model, dataloader, optimizer, scaler, epoch, use_channels_last=False):
    model.train()
    total_loss = 0.0
    tp_sum = fp_sum = fn_sum = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [train]')
    for images, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)
        if use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            logits = model(images)
            loss   = combined_loss(masks, logits)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            tp, fp, fn = confusion_counts(masks, logits, THRESHOLD)
        tp_i, fp_i, fn_i = tp.item(), fp.item(), fn.item()
        tp_sum += tp_i
        fp_sum += fp_i
        fn_sum += fn_i
        total_loss += loss.item()

        # Per-batch numbers shown on the progress bar are just live feedback —
        # the numbers that actually get logged/checkpointed are the
        # epoch-accumulated ones computed below, after the loop.
        b_iou, b_f1, b_dice, _, _ = metrics_from_counts(tp_i, fp_i, fn_i)
        pbar.set_postfix(
            loss=f'{loss.item():.4f}',
            iou=f'{b_iou:.4f}',
            f1=f'{b_f1:.4f}',
            dice=f'{b_dice:.4f}'
        )

    n = len(dataloader)
    epoch_iou, epoch_f1, epoch_dice, _, _ = metrics_from_counts(tp_sum, fp_sum, fn_sum)
    return total_loss / n, epoch_iou, epoch_f1, epoch_dice


def validate_epoch(model, dataloader, use_channels_last=False):
    model.eval()
    total_loss = 0.0
    tp_sum = fp_sum = fn_sum = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                logits = model(images)
                loss   = combined_loss(masks, logits)

            total_loss += loss.item()
            tp, fp, fn = confusion_counts(masks, logits, THRESHOLD)
            tp_sum += tp.item()
            fp_sum += fp.item()
            fn_sum += fn.item()

    n = len(dataloader)
    epoch_iou, epoch_f1, epoch_dice, _, _ = metrics_from_counts(tp_sum, fp_sum, fn_sum)
    return total_loss / n, epoch_iou, epoch_f1, epoch_dice


# -------------------------------------------------
# OPTIMAL THRESHOLD SEARCH
# Sweeps a grid of thresholds over the validation set (reusing the
# already-computed sigmoid probabilities per batch — no extra forward
# passes beyond the grid comparisons) and picks the threshold that
# maximizes F1. Writes the result so inference can actually use it.
# -------------------------------------------------
def search_optimal_threshold(model, dataloader, device, use_amp,
                              thresholds=None, use_channels_last=False):
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.96, 0.02), 4)

    model.eval()
    tp_sums = np.zeros(len(thresholds))
    fp_sums = np.zeros(len(thresholds))
    fn_sums = np.zeros(len(thresholds))

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Threshold search'):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True).float()
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
            probs = torch.sigmoid(logits).float()

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

    script_start = time.time()

    CAPS = detect_capabilities(DEVICE)

    print_hardware_info(header="GPU / HARDWARE INFO")
    print(f'Model type   : SegFormer')
    print(f'Model name   : {model_name}')
    print(f'Backbone     : {BACKBONE}  (pretrained={PRETRAINED})')
    print(f'AMP          : {USE_AMP}')
    print(f'Threshold    : {THRESHOLD}  (training-time metric threshold; '
          f'a separate optimal threshold is searched after training)')
    print(f'torch.compile available: {CAPS["compile"]}')

    all_img_paths = sorted(glob.glob(f'{DATA_PATH}/images/*.tif'))
    if len(all_img_paths) == 0:
        raise FileNotFoundError(
            f'No .tif files found in {DATA_PATH}/images/  '
            'Check your DATA_PATH and folder structure.'
        )
    print(f'Total images : {len(all_img_paths)}')

    train_paths, val_paths = train_test_split(
        all_img_paths, test_size=0.2, random_state=SEED
    )
    print(f'Train: {len(train_paths)}  |  Val: {len(val_paths)}')

    # [PERF #4] More workers / deeper prefetch on GPU runs; CPU-path
    # defaults are unchanged from v4.
    if DEVICE.type == 'cuda':
        num_workers     = max(2, min((os.cpu_count() or 4) - 2, 16))
        prefetch_factor = 4
    else:
        num_workers     = max(1, (os.cpu_count() or 2) // 2)
        prefetch_factor = 2

    USE_CHANNELS_LAST = DEVICE.type == 'cuda'

    # ---- Resume logic ----
    start_epoch  = 1
    best_val_iou = 0.0
    resume_ckpt  = LAST_MODEL if os.path.exists(LAST_MODEL) else None

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
            BACKBONE   = saved_cfg.get('backbone',   BACKBONE)
            PRETRAINED = saved_cfg.get('pretrained', PRETRAINED)
            IMAGE_SIZE = saved_cfg.get('image_size', IMAGE_SIZE)

        model = SegFormerSolar(
            in_channels=4, num_labels=1, backbone=BACKBONE,
            pretrained=PRETRAINED, image_size=IMAGE_SIZE
        ).to(DEVICE)
        if USE_CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)

        model.load_state_dict(ckpt['model'])

        optimizer, fused_ok = make_optimizer(model.parameters(), lr=6e-5,
                                              weight_decay=1e-4, device=DEVICE)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)
        scaler    = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_iou = ckpt['best_iou']
        print(f'Resumed from checkpoint at epoch {ckpt["epoch"]} (best_iou={best_val_iou:.4f}).')
    else:
        MEAN, STD = compute_mean_std(train_paths)
        save_mean_std(MEAN, STD)
        model = SegFormerSolar(
            in_channels=4, num_labels=1, backbone=BACKBONE,
            pretrained=PRETRAINED, image_size=IMAGE_SIZE
        ).to(DEVICE)
        if USE_CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)

        optimizer, fused_ok = make_optimizer(model.parameters(), lr=6e-5,
                                              weight_decay=1e-4, device=DEVICE)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7)
        scaler    = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                'backbone':    BACKBONE,
                'pretrained':  PRETRAINED,
                'in_channels': 4,
                'num_labels':  1,
                'image_size':  IMAGE_SIZE
            }, f, indent=2)
        print('New model initialised.')

    # [PERF #1] `model` stays the RAW module for all checkpoint/export I/O.
    # `train_model` is the (possibly compiled) handle used for forward
    # passes in the loops below.
    train_model, compiled_ok = maybe_compile(model)

    # [PERF #3] Auto batch size — only if the user didn't explicitly set one.
    effective_batch_size = BATCH_SIZE
    if not BATCH_SIZE_EXPLICIT and DEVICE.type == 'cuda':
        print('Probing GPU memory to auto-select batch size '
              f'(starting from --batch-size default {BATCH_SIZE})...')
        effective_batch_size = autodetect_batch_size(
            train_model, combined_loss, DEVICE, IMAGE_SIZE, start_bs=BATCH_SIZE
        )
        model.zero_grad(set_to_none=True)
        print(f'Auto-selected batch size: {effective_batch_size}')
    elif BATCH_SIZE_EXPLICIT:
        print(f'--batch-size was explicitly set to {BATCH_SIZE}; skipping autodetection.')

    print(f'\n{"=" * 60}')
    print('Performance features')
    print(f'{"=" * 60}')
    print(f'  torch.compile   : {"ON" if compiled_ok else "OFF"}')
    print(f'  fused AdamW     : {"ON" if fused_ok else "OFF"}')
    print(f'  channels_last   : {"ON" if USE_CHANNELS_LAST else "OFF"}')
    print(f'  batch size      : {effective_batch_size}'
          f'{" (auto)" if (not BATCH_SIZE_EXPLICIT and DEVICE.type == "cuda") else " (explicit)"}')
    print(f'  DataLoader      : num_workers={num_workers}, prefetch_factor={prefetch_factor}')
    print(f'{"=" * 60}\n')

    # ---- Datasets & loaders ----
    train_dataset = SolarDataset(train_paths, MEAN, STD, is_train=True)
    val_dataset   = SolarDataset(val_paths,   MEAN, STD, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0), prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=PIN_MEMORY,
        persistent_workers=(num_workers > 0), prefetch_factor=prefetch_factor if num_workers > 0 else None
    )

    # ---- Training loop ----
    patience_counter    = 0
    early_stop_patience = 10
    epoch_times          = []   # per-epoch wall-clock durations, for the timing summary

    def make_checkpoint(epoch, best_iou_value):
        # Always checkpoints from the RAW module, never the compiled wrapper,
        # so LAST_MODEL / BEST_MODEL stay format-identical to v4.
        return {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler':    scaler.state_dict(),
            'best_iou':  best_iou_value,
            'mean':      MEAN,
            'std':       STD,
        }

    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_iou, train_f1, train_dice = train_epoch(
            train_model, train_loader, optimizer, scaler, epoch,
            use_channels_last=USE_CHANNELS_LAST
        )
        val_loss, val_iou, val_f1, val_dice = validate_epoch(
            train_model, val_loader, use_channels_last=USE_CHANNELS_LAST
        )

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

        with open(TRAIN_LOG, 'a') as f:
            f.write(log_line + '\n')

        append_csv_row({
            'epoch': epoch, 'time_s': round(duration, 1),
            'train_loss': round(train_loss, 6), 'val_loss': round(val_loss, 6),
            'train_iou': round(train_iou, 6),   'val_iou': round(val_iou, 6),
            'train_f1': round(train_f1, 6),     'val_f1': round(val_f1, 6),
            'train_dice': round(train_dice, 6), 'val_dice': round(val_dice, 6),
        })

        scheduler.step(val_iou)

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
    # POST-TRAINING EXPORT
    # 1) Load the BEST checkpoint's weights into `model` (the raw module),
    #    then save a clean weights_only=True-loadable state_dict as
    #    "<model_name>.pth" — this is the exact file
    #    inference_segformer_solar.py looks for.
    # 2) Run the threshold search on the val set using that same best
    #    model (through the compiled `train_model` handle for speed, since
    #    it wraps the same underlying module by reference), and write
    #    optimal_threshold.json + update segformer_config.json.
    # ======================================================
    best_source = BEST_MODEL if os.path.exists(BEST_MODEL) else LAST_MODEL
    print(f'\nExporting inference artifacts from: {best_source}')

    best_ckpt = torch.load(best_source, map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    model.eval()

    torch.save(model.state_dict(), INFERENCE_WEIGHTS_FILE)
    print(f'  Inference weights : {INFERENCE_WEIGHTS_FILE}')

    threshold_result = search_optimal_threshold(
        train_model, val_loader, DEVICE, USE_AMP, use_channels_last=USE_CHANNELS_LAST
    )
    threshold_result['source_checkpoint'] = os.path.basename(best_source)
    threshold_result['source_epoch']      = best_ckpt.get('epoch')

    with open(OPTIMAL_THRESHOLD_FILE, 'w') as f:
        json.dump(threshold_result, f, indent=2)
    print(f'  Optimal threshold : {threshold_result["optimal_threshold"]:.4f}  '
          f'(F1={threshold_result["f1"]:.4f}, '
          f'P={threshold_result["precision"]:.4f}, '
          f'R={threshold_result["recall"]:.4f})')
    print(f'  Threshold file    : {OPTIMAL_THRESHOLD_FILE}')

    # Also embed into segformer_config.json for redundancy (inference checks
    # this as a fallback if the dedicated threshold files are missing).
    with open(CONFIG_FILE) as f:
        cfg_data = json.load(f)
    cfg_data['optimal_threshold'] = threshold_result['optimal_threshold']
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg_data, f, indent=2)

    print('\nAll set — MODEL_DIR for inference should point at:')
    print(f'  {OUTPUT_FOLDER}')
    print(f'MODEL_NAME for inference should be:')
    print(f'  {model_name}')

    # ======================================================
    # TIMING SUMMARY
    # Per-epoch average time + total wall-clock time for the whole run
    # (data prep + all epochs + export), plus whether it ran on GPU or CPU.
    # ======================================================
    total_elapsed  = time.time() - script_start
    n_epochs_run   = len(epoch_times)
    avg_epoch_time = (sum(epoch_times) / n_epochs_run) if n_epochs_run else 0.0

    hw_lines = print_hardware_info(header="FINAL GPU / HARDWARE INFO")

    print(f'\n{"=" * 60}')
    print(f'ALL DONE — Timing Summary')
    print(f'{"=" * 60}')
    print(f'  MODEL TYPE       : SegFormer (backbone={BACKBONE})')
    print(f'  MODEL NAME       : {model_name}')
    print(f'  AVG TIME/EPOCH   : {avg_epoch_time:.2f}s  ({n_epochs_run} epoch{"s" if n_epochs_run != 1 else ""} run)')
    print(f'  TOTAL TIME       : {total_elapsed:.2f}s')
    print(f'  RUN BY           : {hardware_summary_line()}')
    print(f'  torch.compile    : {"ON" if compiled_ok else "OFF"}')
    print(f'  fused AdamW      : {"ON" if fused_ok else "OFF"}')
    print(f'{"=" * 60}')

    with open(TRAIN_LOG, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('TRAINING SUMMARY\n')
        f.write('=' * 60 + '\n')
        f.write(f'Model type          : SegFormer (backbone={BACKBONE})\n')
        f.write(f'Model name          : {model_name}\n')
        f.write(f'Hardware            : {hardware_summary_line()}\n')
        for line in hw_lines:
            f.write(f'  {line}\n')
        f.write(f'Epochs run          : {n_epochs_run}\n')
        f.write(f'Average epoch time  : {avg_epoch_time:.2f}s\n')
        f.write(f'Total training time : {total_elapsed:.2f}s\n')
        f.write(f'torch.compile       : {"ON" if compiled_ok else "OFF"}\n')
        f.write(f'fused AdamW         : {"ON" if fused_ok else "OFF"}\n')
        f.write('=' * 60 + '\n')
