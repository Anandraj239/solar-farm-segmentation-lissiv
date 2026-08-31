# =========================================================
# SOLAR FARM DETECTION - SEGFORMER INFERENCE  (PyTorch / HuggingFace)
#
# USAGE (unchanged):
#   GPU: python inference_segformer_solar.py <INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> gpu
#   CPU: python inference_segformer_solar.py <INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> cpu
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
MODEL_TYPE = "SegFormer"

# ==========================================================
# SETTINGS
# ==========================================================
# TILE_SIZE / OVERLAP / STEP / GAUSSIAN_WEIGHT are derived from
# segformer_config.json ("image_size") inside main(), after
# MODEL_FOLDER is known. See below.
BATCH_SIZE      = 64   # starting point for autodetection; never a CLI arg

SOLAR_THRESHOLD = 0.5                   # fallback threshold if none was saved at training time
NDWI_THRESHOLD  = 0.2

MIN_AREA_HA     = 2.0
MW_PER_HECTARE  = 0.5

# ---- Inference-quality / speed toggles (script-level config, CLI args unchanged) ----
ENABLE_TTA               = True                   # identity / hflip / vflip / hflip+vflip
ENABLE_MULTISCALE        = False                  # optional — off by default to preserve baseline speed
MULTISCALE_SCALES        = [0.75, 1.0, 1.25]
USE_CHANNELS_LAST        = True
USE_CUDA_STREAMS         = True                   # async H2D prefetch, GPU only
EXPORT_PROBABILITY_RASTER = False                 # optional extra output, doesn't touch existing outputs
EXPORT_UNCERTAINTY_MAP    = False                 # optional extra output, doesn't touch existing outputs

# ---- [PERF] toggles ----
ENABLE_TORCH_COMPILE      = True                  # falls back to eager automatically if unsupported
ENABLE_CUDA_GRAPHS        = True                  # via torch.compile(mode='reduce-overhead'); GPU only
AUTO_BATCH_SIZE           = True                  # probes GPU memory to raise BATCH_SIZE; GPU only


# ==========================================================
# GAUSSIAN WEIGHT MAP
# Plain function of tile_size, called once tile_size is known (from
# segformer_config.json) instead of at import time with a hardcoded 128.
# ==========================================================
def _make_gaussian_weight(tile_size: int) -> np.ndarray:
    sigma  = tile_size / 4.0
    coords = np.arange(tile_size, dtype=np.float32) - (tile_size - 1) / 2.0
    g1d    = np.exp(-0.5 * (coords / sigma) ** 2)
    g2d    = np.outer(g1d, g1d)
    return (g2d / g2d.max()).astype(np.float32)


# ==========================================================
# HARDWARE NAME HELPER (for the timing summary — separate from the
# detect_capabilities() checks below, which are about compute features,
# not the human-readable device name)
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
# [PERF] HARDWARE / CAPABILITY DETECTION + COMPILE / AUTOTUNE HELPERS
# All checks are best-effort and never raise — anything unavailable
# just gets skipped and the corresponding v2 behavior is used instead.
# ==========================================================
def detect_capabilities(device):
    caps = {
        'device':        str(device),
        'cuda':          device.type == 'cuda',
        'torch_version': torch.__version__,
        'compile':       hasattr(torch, 'compile'),
    }
    if device.type == 'cuda':
        try:
            caps['gpu_name'] = torch.cuda.get_device_name(0)
            caps['vram_gb']  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass
    return caps


def maybe_compile(model, device, prefer_cudagraphs=True):
    """
    [PERF #1] Wraps `model` with torch.compile() if supported. Prefers
    mode='reduce-overhead' (CUDA-Graph backed) on GPU when
    ENABLE_CUDA_GRAPHS/prefer_cudagraphs is set; falls back to default
    compile mode, then to plain eager `model`, on any failure at any
    stage. Compilation/graphing is purely a speed optimization and must
    never change prediction values beyond normal floating-point
    tolerance.
    """
    if not hasattr(torch, 'compile'):
        return model, 'eager (torch.compile unavailable)'

    if device.type == 'cuda' and prefer_cudagraphs:
        try:
            compiled = torch.compile(model, mode='reduce-overhead')
            return compiled, 'compiled (reduce-overhead / cudagraphs)'
        except Exception as e:
            print(f"WARNING: torch.compile(mode='reduce-overhead') failed ({e}); "
                  f"trying default compile mode.")

    try:
        compiled = torch.compile(model)
        return compiled, 'compiled (default)'
    except Exception as e:
        print(f"WARNING: torch.compile() unavailable/failed ({e}); continuing eager.")
        return model, 'eager (compile failed)'


def autodetect_inference_batch_size(model, device, in_channels, tile_size, n_views,
                                     use_channels_last, start_bs, max_bs=512, safety=0.85):
    """
    [PERF #2] Probe increasing batch sizes with real forward passes shaped
    like actual inference tiles, replaying the same number of forward calls
    per "batch" that TTA/multi-scale will actually issue (n_views), and
    return the largest batch size that stays under `safety` fraction of
    total GPU memory. Falls back to `start_bs` on CPU or on any failure.
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

                dummy = torch.randn(bs, in_channels, tile_size, tile_size, device=device)
                if use_channels_last:
                    dummy = dummy.contiguous(memory_format=torch.channels_last)

                with torch.inference_mode():
                    for _ in range(max(1, n_views)):
                        with torch.autocast(device_type='cuda'):
                            out = model(dummy)
                        out = out.float()

                peak = torch.cuda.max_memory_allocated(device)
                del dummy, out
                torch.cuda.empty_cache()

                if peak > total_mem * safety:
                    break
                best = bs
                bs *= 2
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                break
    except Exception as e:
        print(f"WARNING: inference batch-size autodetection failed ({e}); using {start_bs}.")
        return start_bs

    return best


# ==========================================================
# MODEL — SegFormer wrapped to match SolarModel's I/O contract:
#   input  : (B, in_channels, H, W) normalized float32
#   output : (B, num_labels, H, W) sigmoid probabilities
#
# Must be built with the SAME backbone / in_channels the checkpoint was
# trained with — those are loaded from segformer_config.json at runtime.
# ==========================================================
class SegFormerSolar(nn.Module):
    def __init__(self, in_channels=4, num_labels=1, backbone='nvidia/mit-b0',
                 pretrained=True, image_size=128):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
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
            # [PERF #4] Request SDPA / Flash-Attention backed attention when
            # the installed `transformers` version supports it; otherwise
            # this is silently ignored and the model uses eager attention,
            # exactly like v2.
            try:
                config._attn_implementation = 'sdpa'
            except Exception:
                pass
            try:
                self.model = SegformerForSemanticSegmentation(config)
            except Exception:
                config._attn_implementation = 'eager'
                self.model = SegformerForSemanticSegmentation(config)

        self.image_size = image_size

    def _adapt_input_channels(self, in_channels):
        """
        Dynamically find the first Conv2d (the patch-embedding
        projection) and rebuild it for `in_channels` input channels.
        Identical implementation to train_segformer_solar.py — no
        hardcoded attribute paths, so it can't silently diverge from
        training on a transformers version bump. (Note: at inference
        time this branch is never actually exercised since the model is
        always constructed with pretrained=False — see below — but it's
        kept identical to training for safety/parity.)
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
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits          # (B, 1, H/4, W/4)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode='bilinear', align_corners=False
        )
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
# train_segformer_solar.py writes optimal_threshold.json (and embeds the
# value in segformer_config.json) so this loader has a real value to
# find in the normal case.
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

    # 2) dedicated json file (written by train_segformer_solar.py)
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

    # 3) embedded directly in segformer_config.json
    for key in ('optimal_threshold', 'threshold', 'best_threshold'):
        if key in seg_cfg:
            try:
                return float(seg_cfg[key]), 'segformer_config.json'
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
# the ResNet-UNet inference script; raw bands + nodata mask are also
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

            # [PERF #1] This float() cast is load-bearing when the model is
            # torch.compile(mode='reduce-overhead')-wrapped: it forces a
            # fresh allocation, copying data out of the compiled callable's
            # internal (possibly CUDA-Graph-owned) static output buffer
            # before that buffer gets reused/overwritten by the next call
            # below. Do not remove or reorder this line relative to the
            # next `model(...)` invocation.
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
# MAIN
# ==========================================================
if __name__ == '__main__':

    if len(sys.argv) < 5:
        print("Usage: python inference_segformer_solar.py "
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

    # This matches the file train_segformer_solar.py actually writes at
    # the end of training (a clean, weights_only=True-loadable state_dict
    # named exactly "<model_name>.pth").
    MODEL_PATH         = os.path.join(MODEL_FOLDER, model_name + '.pth')
    NORMALIZATION_FILE = os.path.join(MODEL_FOLDER, 'normalization_values.txt')
    CONFIG_FILE        = os.path.join(MODEL_FOLDER, 'segformer_config.json')

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
            print("\nTo run on CPU: python inference_segformer_solar.py ... cpu")
            print("!" * 60)
            sys.exit(1)
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')

    CAPS = detect_capabilities(DEVICE)

    print(f"\n{'=' * 60}")
    print(f"Device : {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"torch  : {CAPS['torch_version']}  |  torch.compile available: {CAPS['compile']}")
    print(f"{'=' * 60}\n")

    if DEVICE.type == 'cuda':
        torch.backends.cudnn.benchmark         = True
        torch.backends.cudnn.deterministic     = False
        torch.backends.cuda.matmul.allow_tf32  = True
        torch.backends.cudnn.allow_tf32        = True

    # [PERF #3] More workers / deeper prefetch on GPU runs; CPU-path
    # defaults are unchanged from v2.
    if DEVICE.type == 'cuda':
        NUM_WORKERS = max(2, min((os.cpu_count() or 4) - 2, 16))
    else:
        NUM_WORKERS = 2
    PIN_MEMORY  = DEVICE.type == 'cuda'

    # ---- Load architecture config (written by the training script) ----
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: {CONFIG_FILE} not found. It is written automatically by "
              f"train_segformer_solar.py next to the checkpoint — make sure "
              f"MODEL_DIR points at that training output folder.")
        sys.exit(1)

    with open(CONFIG_FILE) as f:
        seg_cfg = json.load(f)
    BACKBONE    = seg_cfg.get('backbone', 'nvidia/mit-b0')
    IN_CHANNELS = seg_cfg.get('in_channels', 4)
    NUM_LABELS  = seg_cfg.get('num_labels', 1)

    # Tile size comes from the training config instead of a hardcoded 128.
    # OVERLAP is kept at 50% of tile size (same invariant as before), STEP
    # and the Gaussian weight map are derived from it.
    TILE_SIZE       = seg_cfg.get('image_size', 128)
    OVERLAP         = TILE_SIZE // 2
    STEP            = TILE_SIZE - OVERLAP
    GAUSSIAN_WEIGHT = _make_gaussian_weight(TILE_SIZE)

    # Always pretrained=False at inference time — the checkpoint's
    # state_dict overwrites every weight immediately below, so downloading
    # full pretrained backbone weights here is wasted bandwidth/time. Only
    # a small config JSON needs to be fetched (or read from local HF cache)
    # to build the right architecture shape.
    PRETRAINED = False

    # ---- Load model ----
    script_start     = time.time()
    model_load_start = time.time()
    print("Loading model...")
    model = SegFormerSolar(in_channels=IN_CHANNELS, num_labels=NUM_LABELS,
                            backbone=BACKBONE, pretrained=PRETRAINED,
                            image_size=TILE_SIZE).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    if USE_CHANNELS_LAST and DEVICE.type == 'cuda':
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:
            print(f"WARNING: could not convert model to channels_last ({e}); continuing without it.")

    # [PERF #1] Compile once, outside the per-image loop. The first batch
    # of each distinct shape pays a one-time (re)compilation cost; this is
    # reported separately below so it doesn't skew the per-batch timing
    # prints inside the main loop.
    compile_note = 'eager (disabled)'
    if ENABLE_TORCH_COMPILE:
        model, compile_note = maybe_compile(model, DEVICE, prefer_cudagraphs=ENABLE_CUDA_GRAPHS)

    model_load_time = time.time() - model_load_start
    print(f"Model loaded : {MODEL_PATH}  ({model_load_time:.2f}s)")
    print(f"Backbone     : {BACKBONE}  (in_channels={IN_CHANNELS})")
    print(f"Compile mode : {compile_note}")

    MEAN, STD = load_mean_std(NORMALIZATION_FILE)
    print(f"Normalization: mean={MEAN}, std={STD}  (dtype={MEAN.dtype})")

    EFFECTIVE_THRESHOLD, threshold_source = load_optimal_threshold(MODEL_FOLDER, seg_cfg, SOLAR_THRESHOLD)
    if threshold_source:
        print(f"Threshold    : {EFFECTIVE_THRESHOLD:.4f}  (loaded from {threshold_source})")
    else:
        print(f"Threshold    : {EFFECTIVE_THRESHOLD:.4f}  (fallback default — no saved threshold found)")

    SCALES = MULTISCALE_SCALES if ENABLE_MULTISCALE else [1.0]
    N_TTA_VIEWS = (4 if ENABLE_TTA else 1) * len(SCALES)

    # [PERF #2] Auto-select the inference batch size (never a CLI arg, so
    # this can't conflict with any user-supplied option).
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE
    if AUTO_BATCH_SIZE and DEVICE.type == 'cuda':
        print(f"Probing GPU memory to auto-select inference batch size "
              f"(starting from {BATCH_SIZE}, {N_TTA_VIEWS} view(s)/tile)...")
        EFFECTIVE_BATCH_SIZE = autodetect_inference_batch_size(
            model, DEVICE, IN_CHANNELS, TILE_SIZE, N_TTA_VIEWS,
            use_channels_last=(USE_CHANNELS_LAST and DEVICE.type == 'cuda'),
            start_bs=BATCH_SIZE
        )
        print(f"Auto-selected batch size: {EFFECTIVE_BATCH_SIZE}")

    tif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    print(f"Total images : {len(tif_files)}\n")
    print(f"Tiling config: TILE={TILE_SIZE}, STEP={STEP}, OVERLAP={OVERLAP}  "
          f"(from segformer_config.json)")
    print(f"Weighting    : 2-D Gaussian (sigma={TILE_SIZE//4}px)")
    print(f"Padding      : normalized-space zero-fill")
    print(f"TTA          : {'ON (identity/h/v/hv)' if ENABLE_TTA else 'OFF'}")
    print(f"Multi-scale  : {'ON ' + str(SCALES) if ENABLE_MULTISCALE else 'OFF'}  -> {N_TTA_VIEWS} forward view(s)/tile")
    print(f"channels_last: {USE_CHANNELS_LAST and DEVICE.type == 'cuda'}")
    print(f"CUDA streams : {USE_CUDA_STREAMS and DEVICE.type == 'cuda'}")
    print(f"torch.compile: {compile_note}")
    print(f"Batch size   : {EFFECTIVE_BATCH_SIZE}"
          f"{' (auto)' if (AUTO_BATCH_SIZE and DEVICE.type == 'cuda') else ' (default)'}")
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
            batch_size=EFFECTIVE_BATCH_SIZE,
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
                model, tiles_dev, DEVICE,
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

    print(f"\n{'=' * 60}")
    print(f"ALL DONE — Timing Summary  [{DEVICE}]")
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
    print(f"  Compile mode     : {compile_note}")
    print(f"  Batch size       : {EFFECTIVE_BATCH_SIZE}")
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
    #  e.g. 83 tiffs in, only 79 .shp out — without ever raising an error)
    if skipped_files:
        print(f"\n{'─' * 60}")
        print(f"PROCESSED BUT NO SHAPEFILE — {len(skipped_files)} of {len(tif_files)} tiffs "
              f"finished normally but produced no {'.shp'} (mask .tif was still written for each):")
        print(f"{'─' * 60}")
        for n, reason in skipped_files:
            print(f"  - {n}.tif   ({reason})")
        print(f"{'─' * 60}")
        print(f"Tip: lower SOLAR_THRESHOLD ({SOLAR_THRESHOLD}) and/or MIN_AREA_HA ({MIN_AREA_HA}) "
              f"if these should have been detected.")
    else:
        print(f"\nEvery processed tiff produced a shapefile — none skipped for being empty.")