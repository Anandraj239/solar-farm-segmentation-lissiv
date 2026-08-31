# =========================================================
# SOLAR FARM DETECTION - PYTORCH INFERENCE  (v11 — TF-MATCHED + ACCURACY FIXED)
# Supports both GPU and CPU inference via command-line arg.
# USAGE:
#   GPU: python inference_solar_v11_accuracy.py <INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> gpu
#   CPU: python inference_solar_v11_accuracy.py <INPUT> <OUTPUT> <MODEL_DIR> <MODEL_NAME> cpu
# =========================================================

import sys
import os
import glob
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
MODEL_TYPE = "UNet"   # SolarModel = ResNet50 encoder + attention decoder (U-Net family)

# ==========================================================
# SETTINGS
# ==========================================================
TILE_SIZE       = 128
OVERLAP         = 64                    # FIX-1: was 16 (12.5%) → now 64 (50%)
STEP            = TILE_SIZE - OVERLAP   # = 64
BATCH_SIZE      = 64

SOLAR_THRESHOLD = 0.5
NDWI_THRESHOLD  = 0.2

MIN_AREA_HA     = 2.0
MW_PER_HECTARE  = 0.5


# ==========================================================
# FIX-2: GAUSSIAN WEIGHT MAP
# Built once at module level; reused for every tile.
# Shape: (TILE_SIZE, TILE_SIZE), float32, values in (0, 1].
# sigma = TILE_SIZE / 4  gives ~0.98 at center, ~0.02 at edges.
# ==========================================================
def _make_gaussian_weight(tile_size: int) -> np.ndarray:
    """
    Returns a (tile_size, tile_size) float32 array whose values follow
    a 2-D Gaussian centred on the tile, normalised so the peak = 1.0.
    """
    sigma  = tile_size / 4.0
    coords = np.arange(tile_size, dtype=np.float32) - (tile_size - 1) / 2.0
    g1d    = np.exp(-0.5 * (coords / sigma) ** 2)
    g2d    = np.outer(g1d, g1d)
    return (g2d / g2d.max()).astype(np.float32)

GAUSSIAN_WEIGHT = _make_gaussian_weight(TILE_SIZE)   # (128, 128)


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
# MODEL COMPONENTS  — identical to training v11
# ==========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1      = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.conv2      = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2        = nn.BatchNorm2d(out_channels)
        self.dropout    = nn.Dropout2d(0.3)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.gap(x).view(b, c)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1, 1)
        return x * se


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, out_channels, 1)
        self.phi   = nn.Conv2d(in_channels, out_channels, 1)
        self.bn    = nn.BatchNorm2d(out_channels)
        self.psi   = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        theta = self.theta(x)
        phi   = self.phi(g)
        act   = self.bn(F.relu(theta + phi))   # ReLU -> BN matches TF/training
        psi   = self.psi(act)
        return x * psi


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.attention   = AttentionBlock(out_channels, out_channels)
        self.conv_block  = ConvBlock(out_channels * 2, out_channels)
        self.dropout_out = nn.Dropout2d(0.2)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            import warnings
            warnings.warn(
                f"DecoderBlock shape mismatch: upsampled x={x.shape}, "
                f"skip={skip.shape} — architecture mismatch with checkpoint.",
                stacklevel=2
            )
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        skip = self.skip_proj(skip)
        skip = self.attention(skip, x)
        x    = torch.cat([x, skip], dim=1)
        x    = self.conv_block(x)
        x    = self.dropout_out(x)
        return x


class SolarModel(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet50(weights=None)

        self.enc_conv1   = nn.Conv2d(in_channels, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)
        self.enc_bn1     = resnet.bn1
        self.enc_relu    = resnet.relu
        self.enc_maxpool = resnet.maxpool
        self.enc_layer1  = resnet.layer1
        self.enc_layer2  = resnet.layer2
        self.enc_layer3  = resnet.layer3
        self.enc_layer4  = resnet.layer4

        self.bridge_conv    = ConvBlock(2048, 512)
        self.bridge_dropout = nn.Dropout2d(0.5)
        self.bridge_se      = SEBlock(512)

        self.decoder1 = DecoderBlock(512,  1024, 512)
        self.decoder2 = DecoderBlock(512,   512, 256)
        self.decoder3 = DecoderBlock(256,   256, 128)
        self.decoder4 = DecoderBlock(128,    64,  64)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1)
        )
        self.final   = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x  = self.enc_conv1(x)
        x  = self.enc_bn1(x)
        s1 = self.enc_relu(x)
        x  = self.enc_maxpool(s1)
        s2 = self.enc_layer1(x)
        s3 = self.enc_layer2(s2)
        s4 = self.enc_layer3(s3)
        b  = self.enc_layer4(s4)

        bridge = self.bridge_conv(b)
        bridge = self.bridge_dropout(bridge)
        bridge = self.bridge_se(bridge)

        d1 = self.decoder1(bridge, s4)
        d2 = self.decoder2(d1,    s3)
        d3 = self.decoder3(d2,    s2)
        d4 = self.decoder4(d3,    s1)

        x = self.final_up(d4)
        x = self.final_conv(x)
        x = self.final(x)
        x = self.sigmoid(x)
        return x


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
# TILE DATASET
#
# FIX-3: Padding is applied AFTER normalization.
#   Old flow: pad raw bands with 0 → normalize
#             → pad pixels become (0 - mean) / std  (large negative)
#   New flow: normalize raw bands → pad normalized tensor with 0.0
#             → pad pixels are 0.0 = per-channel mean in normalized space
#
# The NoData mask is still built from raw bands before any padding,
# so border zeroing logic is unaffected.
# ==========================================================
class TiledSolarDataset(Dataset):
    def __init__(self, tif_path, mean, std, tile_size=128, step=64):
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

        # NoData mask — built from RAW bands before any padding (unchanged)
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

        # FIX-3: normalize FIRST on the actual (unpadded) data
        tile = preprocess_tile(g, r, n_band, self.mean, self.std)  # (h, w, 4), float32
        tile_chw = torch.from_numpy(tile.transpose(2, 0, 1))       # (4, h, w)

        # FIX-3: pad the NORMALIZED tensor with 0.0 (= per-channel mean)
        # instead of padding raw bands with 0 before normalization.
        h_actual, w_actual = tile_chw.shape[1], tile_chw.shape[2]
        if h_actual != self.tile_size or w_actual != self.tile_size:
            pad_bottom = self.tile_size - h_actual
            pad_right  = self.tile_size - w_actual
            # F.pad order: (left, right, top, bottom)
            tile_chw = F.pad(tile_chw, (0, pad_right, 0, pad_bottom), value=0.0)

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
        list(g_raws),
        list(n_raws),
        list(nodata_masks),
        list(zip(ys, y2s, xs, x2s))
    )


# ==========================================================
# MAIN
# ==========================================================
if __name__ == '__main__':

    if len(sys.argv) < 5:
        print("Usage: python inference_solar_v11_accuracy.py "
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

    MODEL_PATH         = os.path.join(MODEL_FOLDER, model_name + '.pth')
    NORMALIZATION_FILE = os.path.join(MODEL_FOLDER, 'normalization_values.txt')

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
            print("\nTo run on CPU: python inference_solar_v11_accuracy.py ... cpu")
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

    NUM_WORKERS = 4 if DEVICE.type == 'cuda' else 2
    PIN_MEMORY  = DEVICE.type == 'cuda'

    # ---- Load model ----
    script_start     = time.time()
    model_load_start = time.time()
    print("Loading model...")
    model = SolarModel().to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model_load_time = time.time() - model_load_start
    print(f"Model loaded : {MODEL_PATH}  ({model_load_time:.2f}s)")

    MEAN, STD = load_mean_std(NORMALIZATION_FILE)
    print(f"Normalization: mean={MEAN}, std={STD}  (dtype={MEAN.dtype})")

    tif_files = glob.glob(os.path.join(INPUT_FOLDER, "*.tif"))
    print(f"Total images : {len(tif_files)}\n")
    print(f"Tiling config: TILE={TILE_SIZE}, STEP={STEP}, OVERLAP={OVERLAP} "
          f"(FIX-1: was 16, now {OVERLAP})")
    print(f"Weighting    : 2-D Gaussian (sigma={TILE_SIZE//4}px)  (FIX-2)")
    print(f"Padding      : normalized-space zero-fill              (FIX-3)\n")

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

        with rasterio.open(tif) as src:
            profile   = src.profile
            transform = src.transform
            crs       = src.crs
            width     = src.width
            height    = src.height

        # FIX-2: use float32 accumulators for weighted average
        prediction = np.zeros((height, width), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)   # replaces counter

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

        for batch_idx, (tiles, g_raws, n_raws, nodata_masks, coords) in enumerate(dataloader):

            batch_t_start = time.time()
            tiles_dev = tiles.to(DEVICE, non_blocking=True)

            autocast_device = "cuda" if DEVICE.type == "cuda" else "cpu"
            with torch.inference_mode():
                with torch.autocast(autocast_device):
                    preds_dev = model(tiles_dev)   # (B, 1, H, W)

            preds_np = preds_dev.squeeze(1).float().cpu().numpy()   # (B, H, W)

            for i, (yy, yy2, xx, xx2) in enumerate(coords):
                h_crop = yy2 - yy
                w_crop = xx2 - xx

                pred = preds_np[i, :h_crop, :w_crop]

                # NDWI water mask (unchanged)
                g_crop      = g_raws[i].numpy()[:h_crop, :w_crop]
                n_band_crop = n_raws[i].numpy()[:h_crop, :w_crop]
                ndwi        = compute_ndwi(g_crop, n_band_crop)
                pred[ndwi > NDWI_THRESHOLD] = 0.0

                # NoData border mask (unchanged)
                nodata_crop = nodata_masks[i].numpy()[:h_crop, :w_crop]
                pred[nodata_crop] = 0.0

                # FIX-2: weighted accumulation with Gaussian map
                w_crop_arr = GAUSSIAN_WEIGHT[:h_crop, :w_crop]   # crop weight to tile size
                prediction[yy:yy2, xx:xx2] += pred * w_crop_arr
                weight_sum[yy:yy2, xx:xx2] += w_crop_arr         # replaces +=1

            batch_elapsed = time.time() - batch_t_start
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                print(f"  Batch {batch_idx + 1:4d}/{len(dataloader)}: {batch_elapsed:.3f}s")

        inference_elapsed = time.time() - inference_start
        print(f"Inference done: {inference_elapsed:.2f}s")

        # FIX-2: divide by Gaussian weight_sum instead of flat counter
        prediction /= np.maximum(weight_sum, 1e-6)

        # ---- Solar mask — threshold + morphological clean-up (unchanged) ----
        solar_mask = (prediction > SOLAR_THRESHOLD).astype(np.uint8)
        solar_mask = cv2.morphologyEx(solar_mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        solar_mask = cv2.morphologyEx(solar_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # ---- Save mask ----
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(mask_out, "w", **profile) as dst:
            dst.write(solar_mask, 1)

        # ---- Polygon extraction (unchanged) ----
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for s, v in shapes(solar_mask, transform=transform)
        )
        geoms = [shape(r["geometry"]) for r in results if r["properties"]["value"] == 1]

        if len(geoms) == 0:
            print(f"  → NO SHP written: prediction never exceeded threshold "
                  f"({SOLAR_THRESHOLD:.4f}) anywhere in this image.")
            scene_elapsed = time.time() - scene_start
            print(f"TOTAL TIME for {name}: {scene_elapsed:.2f}s")
            image_times.append((name, scene_elapsed))
            skipped_files.append((name, f"no pixels above threshold {SOLAR_THRESHOLD:.4f}"))
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