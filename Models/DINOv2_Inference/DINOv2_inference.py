# =========================================================
# SOLAR FARM DETECTION - DINOv2 INFERENCE  (PyTorch / HuggingFace)
# Supports both GPU and CPU inference via command-line arg.

# USAGE:
#   GPU: python inference_dinov2_solar.py <INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> gpu
#   CPU: python inference_dinov2_solar.py <INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> cpu
# =========================================================

import sys
import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import cv2
import time
import platform
import traceback
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# MODEL IDENTITY (shown in the final timing summary)
# ==========================================================
MODEL_TYPE = "DINOv2"

# ==========================================================
# SETTINGS
# ==========================================================
# TILE_SIZE / OVERLAP / STEP / GAUSSIAN_WEIGHT are not module-level
# constants — they're derived from dinov2_config.json ("image_size")
# inside main(), after MODEL_FOLDER is known. See below.
# [OPT: auto batch size] BATCH_SIZE is now a *ceiling* the probe below won't
# exceed, not a fixed value — see find_safe_batch_size() in main(). Raise
# this if you know a card has more headroom than the probe will find safe by
# itself; lower it to cap throughput/VRAM use. On CPU (or if probing is
# skipped) this value is used as-is.
BATCH_SIZE      = 64

SOLAR_THRESHOLD = 0.5                   # fallback threshold if none was saved at training time
NDWI_THRESHOLD  = 0.2

MIN_AREA_HA     = 2.0
MW_PER_HECTARE  = 0.5

DINOV2_PATCH_SIZE = 14  # fixed for all official facebook/dinov2-* checkpoints

# ---- Inference-quality / speed toggles (script-level config, CLI args unchanged) ----
ENABLE_TTA               = True                   # identity / hflip / vflip / hflip+vflip
ENABLE_MULTISCALE        = False                  # optional — off by default to preserve baseline speed
MULTISCALE_SCALES        = [0.75, 1.0, 1.25]
USE_CHANNELS_LAST        = True
USE_CUDA_STREAMS         = True                   # async H2D prefetch, GPU only
EXPORT_PROBABILITY_RASTER = False                 # optional extra output, doesn't touch existing outputs
EXPORT_UNCERTAINTY_MAP    = False                 # optional extra output, doesn't touch existing outputs
USE_TORCH_COMPILE         = True                  # [OPT: torch.compile] set False to force eager mode


# ==========================================================
# GAUSSIAN WEIGHT MAP
# Plain function of tile_size, called once tile_size is known (from
# dinov2_config.json) instead of at import time with a hardcoded value.
# ==========================================================
def _make_gaussian_weight(tile_size: int) -> np.ndarray:
    sigma  = tile_size / 4.0
    coords = np.arange(tile_size, dtype=np.float32) - (tile_size - 1) / 2.0
    g1d    = np.exp(-0.5 * (coords / sigma) ** 2)
    g2d    = np.outer(g1d, g1d)
    return (g2d / g2d.max()).astype(np.float32)


# ==========================================================
# HARDWARE NAME HELPER (for the timing summary)
# ==========================================================
def get_hardware_name(device):
    if device.type == 'cuda':
        return f"GPU - {torch.cuda.get_device_name(0)}"
    try:
        if platform.system() == "Windows":
            name = platform.processor()
        elif platform.system() == "Darwin":
            import subprocess
            name = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
        elif platform.system() == "Linux":
            name = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        name = line.split(":", 1)[1].strip()
                        break
            if not name:
                name = platform.processor()
        else:
            name = platform.processor()
    except Exception:
        name = None
    return f"CPU - {name}" if name else "CPU - (name unavailable)"


# ==========================================================
# SEGMENTATION HEAD — identical to train_dinov2_solar.py's Dinov2SegHead.
# CHANGELOG: newly-trained models default to a GroupNorm head (more stable
# than BatchNorm at small batch sizes). Attribute names stay bn1/bn2 either
# way so state_dict keys are unaffected. Which norm type to build is read
# from dinov2_config.json's "head_norm" key at model-construction time below
# (main()), defaulting to 'batch' when that key is absent so OLD exported
# models (trained before this option existed) still load correctly.
# ==========================================================
def _make_norm2d(norm_type, channels, max_groups=32):
    if norm_type == 'group':
        groups = max_groups
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    return nn.BatchNorm2d(channels)


class Dinov2SegHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_labels=1, dropout=0.1,
                 norm_type='batch'):
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


# ==========================================================
# MODEL — DINOv2 wrapped to match the SolarModel I/O contract:
#   input  : (B, in_channels, H, W) normalized float32
#   output : (B, num_labels, H, W) sigmoid probabilities
#
# Must be built with the SAME backbone / in_channels / head_channels the
# checkpoint was trained with — those are loaded from dinov2_config.json
# at runtime.
# ==========================================================
class Dinov2Solar(nn.Module):
    def __init__(self, in_channels=4, num_labels=1, backbone='facebook/dinov2-small',
                 pretrained=True, image_size=126, head_channels=256, freeze_backbone=False,
                 head_norm='batch'):
        super().__init__()
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

        self.head = Dinov2SegHead(self.hidden_size, hidden_channels=head_channels,
                                   num_labels=num_labels, norm_type=head_norm)

    def _adapt_input_channels(self, in_channels):
        """
        Dynamically find the first Conv2d (the patch-embedding projection) and
        rebuild it for `in_channels` input channels. Identical implementation to
        train_dinov2_solar.py — no hardcoded attribute paths, so it can't silently
        diverge from training on a transformers version bump. (Note: at inference
        time this branch is never actually exercised since the model is always
        constructed with pretrained=False — see the header notes above — but it's
        kept identical to training for safety/parity.)
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
            return

        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None)
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

        # See train_dinov2_solar.py's identical comment: forward() checks a stored
        # self.num_channels attribute, not the conv's in_channels, so both must be updated.
        if hasattr(parent, 'num_channels'):
            parent.num_channels = in_channels
        self.backbone.config.num_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.shape

        try:
            outputs = self.backbone(pixel_values=x, interpolate_pos_encoding=True)
        except TypeError:
            outputs = self.backbone(pixel_values=x)

        tokens = outputs.last_hidden_state

        h_p = H // self.patch_size
        w_p = W // self.patch_size
        num_patches = h_p * w_p

        # Patch tokens are always the LAST `num_patches` tokens in the sequence,
        # regardless of how many prefix tokens (CLS, registers) come before them.
        patch_tokens = tokens[:, -num_patches:, :]
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, self.hidden_size, h_p, w_p)

        logits = self.head(patch_tokens)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
        return torch.sigmoid(logits)


# ==========================================================
# LOAD NORMALIZATION
# ==========================================================
def load_mean_std(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    mean = np.array(list(map(float, lines[1].strip().split(","))), dtype=np.float32)
    std  = np.array(list(map(float, lines[3].strip().split(","))), dtype=np.float32)
    return mean, std


# ==========================================================
# OPTIMAL THRESHOLD LOADER (falls back to SOLAR_THRESHOLD)
# train_dinov2_solar.py writes optimal_threshold.json (and embeds the
# value in dinov2_config.json) so this loader has a real value to find
# in the normal case.
# ==========================================================
def load_optimal_threshold(model_folder, seg_cfg, default=SOLAR_THRESHOLD):
    # 1) dedicated plain-text file
    txt_path = os.path.join(model_folder, 'optimal_threshold.txt')
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r') as f:
                val = float(f.read().strip().split(",")[0])
            return val, txt_path
        except Exception:
            pass

    # 2) dedicated json file (written by train_dinov2_solar.py)
    json_path = os.path.join(model_folder, 'optimal_threshold.json')
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key in ('optimal_threshold', 'threshold', 'best_threshold'):
                    if key in data:
                        return float(data[key]), json_path
            elif isinstance(data, (int, float)):
                return float(data), json_path
        except Exception:
            pass

    # 3) embedded directly in dinov2_config.json
    for key in ('optimal_threshold', 'threshold', 'best_threshold'):
        if key in seg_cfg:
            try:
                return float(seg_cfg[key]), 'dinov2_config.json'
            except Exception:
                pass

    # 4) fallback
    return default, None


# ==========================================================
# INDEX HELPERS
# ==========================================================
def compute_ndvi(r, n):
    return (n - r) / (n + r + 1e-6)

def compute_ndwi(g, n):
    return (g - n) / (g + n + 1e-6)


# ==========================================================
# PREPROCESS TILE
# ==========================================================
def preprocess_tile(g, r, n, mean, std):
    ndvi = compute_ndvi(r, n)
    img  = np.stack([g, r, n, ndvi], axis=-1)
    img  = (img - mean) / std
    return img


# ==========================================================
# UTM HELPER
# ==========================================================
def get_utm_epsg(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone


# ==========================================================
# TILE DATASET  (identical tiling / normalize-then-pad / NoData logic to
# the SegFormer inference script; raw bands + nodata mask are also
# padded to tile_size so the batch can be stacked and masking vectorized
# downstream instead of looped in Python per-tile)
# ==========================================================
class TiledSolarDataset(Dataset):
    def __init__(self, tif_path, mean, std, tile_size, step):
        self.tif_path  = tif_path
        self.mean      = mean
        self.std       = std
        self.tile_size = tile_size
        self.step      = step
        self.src       = None

        with rasterio.open(tif_path) as src:
            self.height = src.height
            self.width  = src.width
            self.nodata = src.nodata

        self.coords = []
        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                y2 = min(y + tile_size, self.height)
                x2 = min(x + tile_size, self.width)
                self.coords.append((y, y2, x, x2))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        if self.src is None:
            self.src = rasterio.open(self.tif_path)

        y, y2, x, x2 = self.coords[idx]
        window = ((y, y2), (x, x2))

        g      = self.src.read(1, window=window).astype(np.float32)
        r      = self.src.read(2, window=window).astype(np.float32)
        n_band = self.src.read(3, window=window).astype(np.float32)

        if self.nodata is not None:
            nodata_mask = (
                (g      == self.nodata) |
                (r      == self.nodata) |
                (n_band == self.nodata)
            )
        else:
            nodata_mask = (g == 0) & (r == 0) & (n_band == 0)

        g_raw      = g.copy()
        n_band_raw = n_band.copy()
        nodata_raw = nodata_mask.copy()

        tile = preprocess_tile(g, r, n_band, self.mean, self.std)  # (h, w, 4), float32
        tile_chw = torch.from_numpy(tile.transpose(2, 0, 1))       # (4, h, w)

        h_actual, w_actual = tile_chw.shape[1], tile_chw.shape[2]
        if h_actual != self.tile_size or w_actual != self.tile_size:
            pad_bottom = self.tile_size - h_actual
            pad_right  = self.tile_size - w_actual
            tile_chw = F.pad(tile_chw, (0, pad_right, 0, pad_bottom), value=0.0)

            # Pad the raw bands / nodata mask identically (zero / False fill)
            # so batches can be stacked into fixed-size tensors. The padded
            # region is never used — every consumer crops back down to
            # (h_actual, w_actual) before it matters, so this is a no-op
            # numerically, only enables vectorized batch ops downstream.
            g_pad = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            n_pad = np.zeros((self.tile_size, self.tile_size), dtype=np.float32)
            nodata_pad = np.zeros((self.tile_size, self.tile_size), dtype=bool)
            g_pad[:h_actual, :w_actual] = g_raw
            n_pad[:h_actual, :w_actual] = n_band_raw
            nodata_pad[:h_actual, :w_actual] = nodata_raw
            g_raw, n_band_raw, nodata_raw = g_pad, n_pad, nodata_pad

        return (
            tile_chw,
            torch.from_numpy(g_raw),
            torch.from_numpy(n_band_raw),
            torch.from_numpy(nodata_raw),
            y, y2, x, x2
        )


def collate_fn(batch):
    tiles, g_raws, n_raws, nodata_masks, ys, y2s, xs, x2s = zip(*batch)
    return (
        torch.stack(tiles),
        torch.stack(g_raws),
        torch.stack(n_raws),
        torch.stack(nodata_masks),
        list(zip(ys, y2s, xs, x2s))
    )


# ==========================================================
# ASYNC CUDA-STREAM PREFETCHER
# Overlaps host->device copy of the *next* batch with GPU compute of the
# *current* batch. On CPU it degrades gracefully to a plain iterator.
# Numerically a no-op — purely a throughput optimization.
# ==========================================================
class _BatchPrefetcher:
    def __init__(self, loader, device, use_stream=True):
        self.loader     = iter(loader)
        self.device     = device
        self.use_stream = use_stream and (device.type == 'cuda')
        self.stream     = torch.cuda.Stream() if self.use_stream else None
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            tiles, g_raws, n_raws, nodata_masks, coords = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        if self.use_stream:
            with torch.cuda.stream(self.stream):
                tiles_dev = tiles.to(self.device, non_blocking=True)
        else:
            tiles_dev = tiles.to(self.device, non_blocking=(self.device.type == 'cuda'))

        self.next_batch = (tiles_dev, g_raws, n_raws, nodata_masks, coords)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration
        if self.use_stream:
            torch.cuda.current_stream().wait_stream(self.stream)
        tiles_dev, g_raws, n_raws, nodata_masks, coords = self.next_batch
        if self.use_stream:
            tiles_dev.record_stream(torch.cuda.current_stream())
        result = (tiles_dev, g_raws, n_raws, nodata_masks, coords)
        self._preload()
        return result


# ==========================================================
# TTA / MULTI-SCALE INFERENCE
# Runs the model over every requested (scale, flip) combination and
# averages the resulting probabilities. Optionally also returns the
# variance across all combinations as an uncertainty estimate.
# AMP autocast is preserved exactly as before, per forward pass.
# ==========================================================
def _apply_flip(x, mode):
    if mode == 'none':
        return x
    elif mode == 'h':
        return torch.flip(x, dims=[3])
    elif mode == 'v':
        return torch.flip(x, dims=[2])
    elif mode == 'hv':
        return torch.flip(x, dims=[2, 3])
    raise ValueError(f"Unknown flip mode: {mode}")


def run_inference(model, tiles_dev, device, use_tta=True, scales=(1.0,),
                   use_channels_last=True, compute_uncertainty=False):
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    flip_modes = ['none', 'h', 'v', 'hv'] if use_tta else ['none']
    target_hw = tiles_dev.shape[-2:]

    all_preds = []
    for scale in scales:
        if scale != 1.0:
            new_h = max(8, int(round(target_hw[0] * scale)))
            new_w = max(8, int(round(target_hw[1] * scale)))
            scaled_in = F.interpolate(tiles_dev, size=(new_h, new_w),
                                       mode='bilinear', align_corners=False)
        else:
            scaled_in = tiles_dev

        for mode in flip_modes:
            aug_in = _apply_flip(scaled_in, mode)
            if use_channels_last and device.type == 'cuda':
                aug_in = aug_in.contiguous(memory_format=torch.channels_last)

            with torch.inference_mode():
                with torch.autocast(autocast_device):
                    out = model(aug_in)            # (B, 1, h, w) sigmoid probs

            out = out.float()
            out = _apply_flip(out, mode)            # flips are self-inverse

            if scale != 1.0:
                out = F.interpolate(out, size=target_hw, mode='bilinear', align_corners=False)

            all_preds.append(out)

    stacked   = torch.stack(all_preds, dim=0)       # (K, B, 1, H, W)
    mean_pred = stacked.mean(dim=0)
    var_pred  = stacked.var(dim=0, unbiased=False) if compute_uncertainty else None
    return mean_pred, var_pred


# ==========================================================
# [OPT: auto batch size] Probes decreasing batch sizes with a *real* forward
# pass through run_inference() — using the actual TTA/multi-scale view count,
# since that's what determines peak VRAM, not just batch_size alone — starting
# from `max_batch_size` and halving on CUDA OOM until one fits. Numerically a
# no-op (only ever calls inference_mode forward passes on throwaway zero
# tensors); purely a throughput/robustness tuning step. Skipped on CPU, where
# there's no OOM risk in the same sense and max_batch_size is returned as-is.
# ==========================================================
def find_safe_batch_size(model_fn, tile_size, in_channels, device, max_batch_size=128,
                          use_tta=True, scales=(1.0,), use_channels_last=True):
    if device.type != 'cuda':
        return max_batch_size

    candidate = max_batch_size
    while candidate >= 1:
        try:
            dummy = torch.zeros(candidate, in_channels, tile_size, tile_size, device=device)
            with torch.inference_mode():
                run_inference(model_fn, dummy, device, use_tta=use_tta, scales=scales,
                               use_channels_last=use_channels_last, compute_uncertainty=False)
            del dummy
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print(f'[batch-size] auto-selected batch_size={candidate} '
                  f'(probed with actual TTA/multi-scale view count; ceiling={max_batch_size}).')
            return candidate
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if 'out of memory' not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise  # a real bug, not an OOM -- don't swallow it
            torch.cuda.empty_cache()
            next_candidate = candidate // 2
            print(f'[batch-size] batch_size={candidate} ran out of memory; retrying at {next_candidate}.')
            candidate = next_candidate

    print('[batch-size] Could not fit even batch_size=1; falling back to 1.')
    return 1


# ==========================================================
# MAIN
# ==========================================================
if __name__ == '__main__':

    if len(sys.argv) < 5:
        print("Usage: python inference_dinov2_solar.py "
              "<INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> [gpu|cpu]")
        sys.exit(1)

    INPUT_FOLDER  = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    MODEL_FOLDER  = sys.argv[3]
    model_name    = sys.argv[4]
    device_arg    = sys.argv[5].lower() if len(sys.argv) > 5 else 'gpu'

    if device_arg not in ('gpu', 'cpu'):
        print(f"ERROR: DEVICE must be 'gpu' or 'cpu', got '{device_arg}'")
        sys.exit(1)

    # This matches the file train_dinov2_solar.py actually writes at the end of
    # training (a clean, weights_only=True-loadable state_dict named exactly
    # "<model_name>.pth").
    MODEL_PATH         = os.path.join(MODEL_FOLDER, model_name + '.pth')
    NORMALIZATION_FILE = os.path.join(MODEL_FOLDER, 'normalization_values.txt')
    CONFIG_FILE        = os.path.join(MODEL_FOLDER, 'dinov2_config.json')

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ---- Device selection ----
    if device_arg == 'gpu':
        if not torch.cuda.is_available():
            print("\n" + "!" * 60)
            print("ERROR: 'gpu' was requested but no CUDA GPU was detected.")
            print("\nTo diagnose, run in your conda environment:")
            print("  python -c \"import torch; print(torch.cuda.is_available())\"")
            print("  python -c \"import torch; print(torch.version.cuda)\"")
            print("\nLikely fixes:")
            print("  1. conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia")
            print("  2. Activate the correct conda environment")
            print("  3. Check GPU drivers: nvidia-smi")
            print("\nTo run on CPU: python inference_dinov2_solar.py ... cpu")
            print("!" * 60)
            sys.exit(1)
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    print(f"\n{'=' * 60}")
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"{'=' * 60}\n")

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark         = True
        torch.backends.cudnn.deterministic     = False
        torch.backends.cuda.matmul.allow_tf32  = True
        torch.backends.cudnn.allow_tf32        = True

    # [OPT: DataLoader] More workers to keep pace with the larger auto-selected
    # batch sizes below; capped at 16 for the same reason as the training script.
    NUM_WORKERS = min(16, os.cpu_count() or 8) if DEVICE.type == 'cuda' else 2
    PIN_MEMORY  = DEVICE.type == 'cuda'

    # ---- Load architecture config (written by the training script) ----
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: {CONFIG_FILE} not found. It is written automatically by "
              f"train_dinov2_solar.py next to the checkpoint — make sure "
              f"MODEL_DIR points at that training output folder.")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        seg_cfg = json.load(f)
    BACKBONE      = seg_cfg.get('backbone', 'facebook/dinov2-small')
    IN_CHANNELS   = seg_cfg.get('in_channels', 4)
    NUM_LABELS    = seg_cfg.get('num_labels', 1)
    HEAD_CHANNELS = seg_cfg.get('head_channels', 256)
    # 'batch' default preserves compatibility with models exported before this
    # option existed; models trained with the updated script write 'group' here.
    HEAD_NORM     = seg_cfg.get('head_norm', 'batch')

    # Tile size comes from the training config (already a multiple of the DINOv2
    # patch size). OVERLAP is kept at 50% of tile size, STEP and the Gaussian
    # weight map are derived from it — same invariant as the SegFormer script.
    TILE_SIZE = seg_cfg.get('image_size', 126)
    if TILE_SIZE % DINOV2_PATCH_SIZE != 0:
        adjusted = max(DINOV2_PATCH_SIZE, int(round(TILE_SIZE / DINOV2_PATCH_SIZE)) * DINOV2_PATCH_SIZE)
        print(f"WARNING: image_size={TILE_SIZE} in {CONFIG_FILE} is not a multiple of the "
              f"DINOv2 patch size ({DINOV2_PATCH_SIZE}); adjusting to {adjusted}. "
              f"(This should not happen with a config written by train_dinov2_solar.py.)")
        TILE_SIZE = adjusted

    OVERLAP         = TILE_SIZE // 2
    STEP            = TILE_SIZE - OVERLAP
    GAUSSIAN_WEIGHT = _make_gaussian_weight(TILE_SIZE)

    # Always pretrained=False at inference time — the checkpoint's state_dict
    # overwrites every weight immediately below, so downloading full pretrained
    # backbone weights here is wasted bandwidth/time (especially for DINOv2
    # base/large/giant, whose pretrained weights are multiple GB). Only a small
    # config JSON needs to be fetched (or read from local HF cache) to build the
    # right architecture shape.
    PRETRAINED = False

    # ---- Load model ----
    script_start     = time.time()
    model_load_start = time.time()
    print("Loading model...")
    model = Dinov2Solar(in_channels=IN_CHANNELS, num_labels=NUM_LABELS,
                         backbone=BACKBONE, pretrained=PRETRAINED,
                         image_size=TILE_SIZE, head_channels=HEAD_CHANNELS,
                         freeze_backbone=False, head_norm=HEAD_NORM).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if USE_CHANNELS_LAST and DEVICE.type == 'cuda':
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:
            print(f"WARNING: could not convert model to channels_last ({e}); continuing without it.")

    # [OPT: torch.compile] `model` stays the plain eager nn.Module (state_dict
    # already loaded above, so this doesn't touch weights/predictions at all).
    # A separate compiled callable is built for the actual forward pass;
    # model_call() below tries it and permanently falls back to the eager
    # model — for the rest of this process — the first time it errors
    # (unsupported op, no Triton on this platform, etc). Every prediction
    # this script produces still comes from the identical eager-equivalent
    # computation graph either way.
    _eager_model = model
    _compiled_model = model
    _compile_ok = USE_TORCH_COMPILE and hasattr(torch, 'compile') and DEVICE.type == 'cuda'
    if _compile_ok:
        try:
            _compiled_model = torch.compile(model)
            print("torch.compile   : enabled")
        except Exception as e:
            print(f"torch.compile   : init failed ({e}); using eager mode")
            _compiled_model = model
            _compile_ok = False
    else:
        print(f"torch.compile   : {'disabled' if not USE_TORCH_COMPILE else 'unavailable'} "
              f"(eager mode)")

    _compile_state = {'ok': _compile_ok}

    def model_call(x):
        if _compile_state['ok']:
            try:
                return _compiled_model(x)
            except Exception as e:
                print(f"[compile] compiled forward failed at runtime ({e}); disabling "
                      "torch.compile for the remainder of this run.")
                _compile_state['ok'] = False
        return _eager_model(x)

    model_load_time = time.time() - model_load_start
    print(f"Model loaded : {MODEL_PATH}  ({model_load_time:.2f}s)")
    print(f"Backbone     : {BACKBONE}  (in_channels={IN_CHANNELS}, head_norm={HEAD_NORM})")

    MEAN, STD = load_mean_std(NORMALIZATION_FILE)
    print(f"Normalization: mean={MEAN}, std={STD}  (dtype={MEAN.dtype})")

    EFFECTIVE_THRESHOLD, threshold_source = load_optimal_threshold(MODEL_FOLDER, seg_cfg, SOLAR_THRESHOLD)
    if threshold_source:
        print(f"Threshold    : {EFFECTIVE_THRESHOLD:.4f}  (loaded from {threshold_source})")
    else:
        print(f"Threshold    : {EFFECTIVE_THRESHOLD:.4f}  (fallback default — no saved threshold found)")

    SCALES = MULTISCALE_SCALES if ENABLE_MULTISCALE else [1.0]
    N_TTA_VIEWS = (4 if ENABLE_TTA else 1) * len(SCALES)

    # [OPT: auto batch size] BATCH_SIZE (the configurable ceiling set at the top
    # of the file, default 128) is now probed down to whatever actually fits in
    # VRAM for this exact tile size / TTA / multi-scale configuration, using the
    # already-compiled-or-eager model_call. Predictions/tiling/blending are
    # completely unaffected — this only changes how many tiles are batched per
    # forward pass.
    _batch_size_ceiling = BATCH_SIZE
    BATCH_SIZE = find_safe_batch_size(
        model_call, TILE_SIZE, IN_CHANNELS, DEVICE,
        max_batch_size=_batch_size_ceiling, use_tta=ENABLE_TTA, scales=SCALES,
        use_channels_last=USE_CHANNELS_LAST
    )

    tif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    print(f"Total images : {len(tif_files)}\n")
    print(f"Tiling config: TILE={TILE_SIZE}, STEP={STEP}, OVERLAP={OVERLAP}  "
          f"(from dinov2_config.json)")
    print(f"Weighting    : 2-D Gaussian (sigma={TILE_SIZE//4}px)")
    print(f"Padding      : normalized-space zero-fill")
    print(f"TTA          : {'ON (identity/h/v/hv)' if ENABLE_TTA else 'OFF'}")
    print(f"Multi-scale  : {'ON ' + str(SCALES) if ENABLE_MULTISCALE else 'OFF'}  -> {N_TTA_VIEWS} forward view(s)/tile")
    print(f"Batch size   : {BATCH_SIZE}  "
          f"(auto-selected, ceiling={_batch_size_ceiling if DEVICE.type == 'cuda' else 'n/a (CPU)'})")
    print(f"channels_last: {USE_CHANNELS_LAST and DEVICE.type == 'cuda'}")
    print(f"CUDA streams : {USE_CUDA_STREAMS and DEVICE.type == 'cuda'}")
    print(f"Prob. raster : {'export' if EXPORT_PROBABILITY_RASTER else 'skip'}")
    print(f"Uncertainty  : {'export' if EXPORT_UNCERTAINTY_MAP else 'skip'}\n")

    # ======================================================
    # PROCESS EACH IMAGE
    # (body wrapped in a function so a single failed/corrupt tif can be
    #  caught and skipped in the loop below, instead of crashing the
    #  whole batch — that's how images were silently getting "lost".)
    # ======================================================
    image_times    = []
    failed_files   = []   # (name, tif_path, error_message)      — crashed/raised an exception
    skipped_files  = []   # (name, reason)                       — finished OK but produced no .shp

    def _process_one_tif(tif):
        print(f"\n{'=' * 60}")
        print(f"Processing: {tif}")
        print(f"{'=' * 60}")
        scene_start = time.time()

        name     = os.path.splitext(os.path.basename(tif))[0]
        mask_out = os.path.join(OUTPUT_FOLDER, name + "_solar_mask.tif")
        shp_out  = os.path.join(OUTPUT_FOLDER, name + "_solar_clean.shp")
        prob_out = os.path.join(OUTPUT_FOLDER, name + "_solar_prob.tif")
        unc_out  = os.path.join(OUTPUT_FOLDER, name + "_solar_uncertainty.tif")

        with rasterio.open(tif) as src:
            profile   = src.profile
            transform = src.transform
            crs       = src.crs
            width     = src.width
            height    = src.height

        prediction = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)

        if EXPORT_UNCERTAINTY_MAP:
            uncertainty_accum  = np.zeros((height, width), dtype=np.float32)
            uncertainty_weight = np.zeros((height, width), dtype=np.float32)

        dataset = TiledSolarDataset(tif, MEAN, STD, TILE_SIZE, STEP)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=4 if NUM_WORKERS > 0 else None,
            collate_fn=collate_fn
        )

        print(f"Tiles: {len(dataset)} | Batches: {len(dataloader)} | "
              f"Workers: {NUM_WORKERS} | pin_memory: {PIN_MEMORY}")

        inference_start = time.time()

        batch_iter = _BatchPrefetcher(dataloader, DEVICE, use_stream=USE_CUDA_STREAMS)

        for batch_idx, (tiles_dev, g_raws, n_raws, nodata_masks, coords) in enumerate(batch_iter):

            batch_t_start = time.time()

            mean_pred, var_pred = run_inference(
                model_call, tiles_dev, DEVICE,   # [OPT: torch.compile] compiled-or-eager
                use_tta=ENABLE_TTA,
                scales=SCALES,
                use_channels_last=USE_CHANNELS_LAST,
                compute_uncertainty=EXPORT_UNCERTAINTY_MAP
            )   # (B, 1, H, W) each

            preds_np = mean_pred.squeeze(1).cpu().numpy()          # (B, TILE, TILE)
            var_np   = var_pred.squeeze(1).cpu().numpy() if var_pred is not None else None

            # ---- Vectorized (whole-batch) NDWI + NoData masking ----
            g_batch      = g_raws.numpy()                          # (B, TILE, TILE)
            n_batch      = n_raws.numpy()
            nodata_batch = nodata_masks.numpy()

            ndwi_batch = compute_ndwi(g_batch, n_batch)
            water_mask = ndwi_batch > NDWI_THRESHOLD
            preds_np[water_mask]   = 0.0
            preds_np[nodata_batch] = 0.0
            if var_np is not None:
                var_np[water_mask]   = 0.0
                var_np[nodata_batch] = 0.0

            # ---- Per-tile scatter-accumulate (unavoidable: irregular output regions) ----
            for i, (yy, yy2, xx, xx2) in enumerate(coords):
                h_crop = yy2 - yy
                w_crop = xx2 - xx

                pred_crop  = preds_np[i, :h_crop, :w_crop]
                w_crop_arr = GAUSSIAN_WEIGHT[:h_crop, :w_crop]

                prediction[yy:yy2, xx:xx2] += pred_crop * w_crop_arr
                weight_sum[yy:yy2, xx:xx2] += w_crop_arr

                if EXPORT_UNCERTAINTY_MAP:
                    var_crop = var_np[i, :h_crop, :w_crop]
                    uncertainty_accum[yy:yy2, xx:xx2]  += var_crop * w_crop_arr
                    uncertainty_weight[yy:yy2, xx:xx2] += w_crop_arr

            batch_elapsed = time.time() - batch_t_start
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  Batch {batch_idx + 1:4d}/{len(dataloader)}: {batch_elapsed:.3f}s")

        inference_elapsed = time.time() - inference_start
        print(f"Inference done: {inference_elapsed:.2f}s")

        prediction /= np.maximum(weight_sum, 1e-6)

        if EXPORT_PROBABILITY_RASTER:
            prob_profile = profile.copy()
            prob_profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(prob_out, "w", **prob_profile) as dst:
                dst.write(prediction.astype(np.float32), 1)

        if EXPORT_UNCERTAINTY_MAP:
            uncertainty_map = uncertainty_accum / np.maximum(uncertainty_weight, 1e-6)
            unc_profile = profile.copy()
            unc_profile.update(dtype=rasterio.float32, count=1)
            with rasterio.open(unc_out, "w", **unc_profile) as dst:
                dst.write(uncertainty_map.astype(np.float32), 1)

        solar_mask = (prediction > EFFECTIVE_THRESHOLD).astype(np.uint8)
        solar_mask = cv2.morphologyEx(solar_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        solar_mask = cv2.morphologyEx(solar_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(mask_out, "w", **profile) as dst:
            dst.write(solar_mask, 1)

        results = (
            {'properties': {'value': v}, 'geometry': s}
            for s, v in shapes(solar_mask, transform=transform)
        )
        geoms = [shape(r["geometry"]) for r in results if r["properties"]["value"] == 1]

        if len(geoms) == 0:
            print(f"  → NO SHP written: prediction never exceeded threshold "
                  f"({EFFECTIVE_THRESHOLD:.4f}) anywhere in this image.")
            scene_elapsed = time.time() - scene_start
            print(f"TOTAL TIME for {name}: {scene_elapsed:.2f}s")
            image_times.append((name, scene_elapsed))
            skipped_files.append((name, f"no pixels above threshold {EFFECTIVE_THRESHOLD:.4f}"))
            return name

        gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)

        if crs.is_geographic:
            centroid = gdf.geometry.unary_union.centroid
            utm_epsg = get_utm_epsg(centroid.x, centroid.y)
            gdf      = gdf.to_crs(epsg=utm_epsg)

        gdf["area_m2"]     = gdf.area
        gdf["area_ha"]     = gdf["area_m2"] / 10000
        gdf                = gdf[gdf["area_ha"] >= MIN_AREA_HA]

        if len(gdf) == 0:
            print(f"  → NO SHP written: solar detected ({len(geoms)} raw polygon(s)) but "
                  f"ALL were smaller than MIN_AREA_HA={MIN_AREA_HA} ha.")
            scene_elapsed = time.time() - scene_start
            print(f"TOTAL TIME for {name}: {scene_elapsed:.2f}s")
            image_times.append((name, scene_elapsed))
            skipped_files.append((name, f"{len(geoms)} polygon(s) found, all < {MIN_AREA_HA} ha"))
            return name

        gdf["capacity_mw"] = gdf["area_ha"] * MW_PER_HECTARE

        if crs.is_geographic:
            gdf = gdf.to_crs(crs)

        gdf.to_file(shp_out)

        scene_elapsed = time.time() - scene_start
        print(f"Features detected : {len(gdf)}")
        print(f"TOTAL TIME for {name}: {scene_elapsed:.2f}s")
        image_times.append((name, scene_elapsed))
        return name

    for tif in tif_files:
        _name = os.path.splitext(os.path.basename(tif))[0]
        try:
            _process_one_tif(tif)
        except Exception as e:
            print(f"\n{'!' * 60}")
            print(f"ERROR processing {tif}: {e}")
            traceback.print_exc()
            print(f"{'!' * 60}\n")
            failed_files.append((_name, tif, str(e)))
            continue

    # ======================================================
    # TIMING SUMMARY
    # ======================================================
    total_elapsed  = time.time() - script_start
    n_images       = len(image_times)
    avg_image_time = (sum(t for _, t in image_times) / n_images) if n_images else 0.0

    expected_names  = [os.path.splitext(os.path.basename(t))[0] for t in tif_files]
    completed_names = {n for n, _ in image_times}
    missing_names   = [n for n in expected_names if n not in completed_names]   # never finished (crashed)
    skipped_names   = {n for n, _ in skipped_files}                              # finished, no .shp
    n_shp_written   = n_images - len(skipped_names)

    hardware_label = get_hardware_name(DEVICE)
    device_label   = (f'GPU [{torch.cuda.get_device_name(0)}]'
                       if DEVICE.type == 'cuda' else 'CPU')

    print(f"\n{'=' * 60}")
    print(f"ALL DONE — Timing Summary  [{device_label}]")
    print(f"{'=' * 60}")
    print(f"  Model            : {MODEL_TYPE}")
    print(f"  Hardware         : {hardware_label}")
    print(f"  Model load       : {model_load_time:.2f}s")
    for img_name, t in image_times:
        flag = "  [no .shp]" if img_name in skipped_names else ""
        print(f"  {img_name}: {t:.2f}s{flag}")
    print(f"  {'─' * 40}")
    print(f"  Images found     : {len(tif_files)}")
    print(f"  Images processed : {n_images}")
    print(f"  Shapefiles (.shp): {n_shp_written}")
    print(f"  Avg time/image   : {avg_image_time:.2f}s  ({avg_image_time / 60:.2f} min)")
    print(f"  TOTAL time       : {total_elapsed:.2f}s  ({total_elapsed / 60:.2f} min)")
    print(f"{'=' * 60}")

    # ---- Crashed / never completed (exception raised mid-processing) ----
    if missing_names:
        print(f"\n{'!' * 60}")
        print(f"CRASHED / NEVER COMPLETED — {len(missing_names)} of {len(tif_files)} tiffs "
              f"raised an error and were skipped entirely:")
        print(f"{'!' * 60}")
        for m in missing_names:
            reason = next((err for n, _, err in failed_files if n == m), "unknown reason")
            print(f"  - {m}.tif   (reason: {reason})")
        print(f"{'!' * 60}")
    else:
        print(f"\nNo tiffs crashed — every input tiff was at least fully processed.")

    # ---- Processed successfully but produced NO shapefile ----
    # (this is the case that silently drops output count vs. input count —
    #  e.g. N tiffs in, fewer .shp out — without ever raising an error)
    if skipped_files:
        print(f"\n{'─' * 60}")
        print(f"PROCESSED BUT NO SHAPEFILE — {len(skipped_files)} of {len(tif_files)} tiffs "
              f"finished normally but produced no .shp (mask .tif was still written for each):")
        print(f"{'─' * 60}")
        for n, reason in skipped_files:
            print(f"  - {n}.tif   ({reason})")
        print(f"{'─' * 60}")
        print(f"Tip: lower SOLAR_THRESHOLD ({SOLAR_THRESHOLD}) and/or MIN_AREA_HA ({MIN_AREA_HA}) "
              f"if these should have been detected.")
    else:
        print(f"\nEvery processed tiff produced a shapefile — none skipped for being empty.")