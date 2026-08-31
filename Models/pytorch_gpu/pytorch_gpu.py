# =========================================================
# SOLAR FARM EXTRACTION (v11) - PyTorch Version [TF-MATCHED]
# RESNET50 + ATTENTION UNET + SE BLOCK
# NORMALIZATION : DATASET MEAN STD
# ---- GPU-ONLY VERSION ----
# =========================================================

import os
import sys
import glob
import time
import platform
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
# CONFIG
# -------------------------------------------------
IMAGE_SIZE  = 128
BATCH_SIZE  = 8
EPOCHS      = 200

# ---- GPU-ONLY: hard requirement on CUDA, no CPU fallback ----
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. This script is the GPU-only version "
        "of the training pipeline and requires a CUDA-capable device."
    )
DEVICE = torch.device('cuda')

# ---- GPU-ONLY: let cuDNN pick the fastest conv algorithms.
# Safe here because input spatial size (IMAGE_SIZE=128) is fixed
# across every batch, so there's no shape-churn penalty. ----
torch.backends.cudnn.benchmark = True

DATA_PATH      = sys.argv[1]
OUTPUT_FOLDER  = sys.argv[2]
model_name     = sys.argv[3]
MODEL_SAVE     = os.path.join(OUTPUT_FOLDER, model_name + '.pth')
MEAN_STD_FILE  = os.path.join(OUTPUT_FOLDER, "normalization_values.txt")
TRAIN_LOG      = os.path.join(OUTPUT_FOLDER, "training_log.txt")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# -------------------------------------------------
# COMPUTE MEAN / STD  (with STD clamp)
# -------------------------------------------------
def compute_mean_std(img_paths):
    sum_    = np.zeros(4)
    sum_sq  = np.zeros(4)
    count   = 0

    for path in tqdm(img_paths, desc="Computing mean/std"):
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
    std  = np.where(std < 1e-6, 1e-6, std)   # STD clamp

    return mean, std


# -------------------------------------------------
# SAVE / LOAD NORMALIZATION
# -------------------------------------------------
def save_mean_std(mean, std):
    with open(MEAN_STD_FILE, "w") as f:
        f.write("Mean:\n")
        f.write(",".join(map(str, mean)) + "\n")
        f.write("Std:\n")
        f.write(",".join(map(str, std)) + "\n")


def load_mean_std_from_file(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        mean = np.array([float(x) for x in lines[1].strip().split(",")])
        std  = np.array([float(x) for x in lines[3].strip().split(",")])
    return mean, std


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
        mask_path = img_path.replace("images", "masks")

        # ---- Image ----
        with rasterio.open(img_path) as src:
            g = src.read(1).astype(np.float32)
            r = src.read(2).astype(np.float32)
            n = src.read(3).astype(np.float32)

        ndvi = (n - r) / (n + r + 1e-6)
        img  = np.stack([g, r, n, ndvi], axis=-1)          # HWC
        img  = (img - self.mean) / self.std

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)   # CHW
        img = F.interpolate(
            img.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # ---- Mask ----
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        mask = (mask > 0).astype(np.float32)               # binarize
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode='nearest'
        ).squeeze(0)

        # ---- Augmentation (train only) ----
        if self.is_train:
            img, mask = self._augment(img, mask)

        return img, mask

    # ------------------------------------------------------------------
    # Augmentations — identical to TF: flip_lr, flip_ud, rot90, 
    # random_brightness(0.1), random_contrast(0.9, 1.1)
    # ------------------------------------------------------------------
    def _augment(self, img, mask):
        # Horizontal flip
        if torch.rand(1) > 0.5:
            img  = torch.flip(img,  dims=[2])
            mask = torch.flip(mask, dims=[2])

        # Vertical flip
        if torch.rand(1) > 0.5:
            img  = torch.flip(img,  dims=[1])
            mask = torch.flip(mask, dims=[1])

        # Random 90° rotation (k in {0,1,2,3})
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img  = torch.rot90(img,  k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        # Random brightness  delta ~ Uniform(-0.1, 0.1)
        if torch.rand(1) > 0.5:
            delta = torch.rand(1).item() * 0.2 - 0.1
            img   = img + delta

        # Random contrast  factor ~ Uniform(0.9, 1.1)
        if torch.rand(1) > 0.5:
            factor = torch.rand(1).item() * 0.2 + 0.9
            img    = img * factor

        return img, mask


# -------------------------------------------------
# MODEL COMPONENTS
# -------------------------------------------------

class ConvBlock(nn.Module):
    """
    Two Conv-BN-LeakyReLU layers followed by Dropout(0.3).
    Matches TF conv_block() exactly, including the internal Dropout(0.3).
    Used in both bridge and decoder; the extra Dropout after decoder
    is handled in DecoderBlock, not here.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1      = nn.Conv2d(in_channels,  out_channels, 3, padding=1)
        self.bn1        = nn.BatchNorm2d(out_channels)
        self.conv2      = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2        = nn.BatchNorm2d(out_channels)
        self.dropout    = nn.Dropout2d(0.3)          # FIX: always 0.3, matching TF conv_block
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block — unchanged from both versions."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(channels, channels // reduction)
        self.fc2  = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.gap(x).view(b, c)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).view(b, c, 1, 1)
        return x * se


class AttentionBlock(nn.Module):
    """
    Attention gate matching TF attention_block().

    TF order:  Add -> ReLU -> BatchNorm -> Conv1x1(sigmoid) -> Multiply
    Previous PyTorch used BN -> ReLU (pre-activation). Fixed here.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, out_channels, 1)   # theta(x)
        self.phi   = nn.Conv2d(in_channels, out_channels, 1)   # phi(g)
        self.bn    = nn.BatchNorm2d(out_channels)
        self.psi   = nn.Sequential(
            nn.Conv2d(out_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, g):
        theta = self.theta(x)
        phi   = self.phi(g)
        # FIX: ReLU first, then BatchNorm — matches TF: relu -> BN
        act   = self.bn(F.relu(theta + phi))
        psi   = self.psi(act)
        return x * psi


class DecoderBlock(nn.Module):
    """
    Decoder stage matching TF decoder_block() exactly:

      ConvTranspose2d (upsample x2)
      -> BN -> LeakyReLU                         [matches TF Conv2DTranspose+BN+LeakyReLU]
      -> Conv1x1 on skip + BN                    [matches TF skip projection]
      -> AttentionBlock(skip, x)                 [matches TF attention_block]
      -> Concatenate                             [matches TF Concatenate]
      -> ConvBlock (Conv-BN-LReLU x2 + Drop 0.3)[matches TF conv_block inside decoder_block]
      -> Dropout(0.2)                            [FIX: extra dropout matching TF decoder_block]
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Upsample path
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

        # Skip projection (Conv1x1 + BN) — matches TF skip projection
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, 1, padding=0),
            nn.BatchNorm2d(out_channels)
        )

        self.attention   = AttentionBlock(out_channels, out_channels)

        # ConvBlock always uses internal Dropout(0.3) — see ConvBlock
        self.conv_block  = ConvBlock(out_channels * 2, out_channels)

        # FIX: extra Dropout(0.2) AFTER conv_block, matching TF decoder_block
        self.dropout_out = nn.Dropout2d(0.2)

    def forward(self, x, skip):
        x    = self.upconv(x)
        skip = self.skip_proj(skip)
        skip = self.attention(skip, x)
        x    = torch.cat([x, skip], dim=1)
        x    = self.conv_block(x)
        x    = self.dropout_out(x)   # FIX: stacked dropout matches TF
        return x


# -------------------------------------------------
# MAIN MODEL
# -------------------------------------------------
class SolarModel(nn.Module):
    """
    ResNet50 + Attention U-Net + SE Block — PyTorch port of TF v11.

    Skip connection mapping (input 128x128), matching TF exactly:

        TF layer name          PyTorch variable   Resolution   Channels
        conv1_relu             s1                 64x64  (1/2)   64
        conv2_block3_out       s2                 32x32  (1/4)  256
        conv3_block4_out       s3                 16x16  (1/8)  512
        conv4_block6_out       s4                  8x8  (1/16) 1024
        conv5_block3_out       bridge (raw)         4x4  (1/32) 2048

    Decoder (matches TF decoder_block call order):
        bridge -> ConvBlock(512) -> Dropout(0.5) -> SE   [bridge refined]
        d1 = decoder(bridge, s4)   ->  8x8,  512ch
        d2 = decoder(d1,    s3)   -> 16x16,  256ch
        d3 = decoder(d2,    s2)   -> 32x32,  128ch
        d4 = decoder(d3,    s1)   -> 64x64,   64ch
        final_up(d4)              -> 128x128,  32ch
        final_conv                -> 128x128,  32ch
        output Conv1x1 + Sigmoid  -> 128x128,   1ch
    """
    def __init__(self, in_channels=4, out_channels=1):
        super().__init__()

        import torchvision.models as models
        resnet = models.resnet50(weights=None)

        # ---- Encoder ----
        # Adapt first conv to 4-channel input (LISS-IV: G, R, N + NDVI)
        self.enc_conv1    = nn.Conv2d(in_channels, 64, kernel_size=7,
                                      stride=2, padding=3, bias=False)
        self.enc_bn1      = resnet.bn1
        self.enc_relu     = resnet.relu
        self.enc_maxpool  = resnet.maxpool

        self.enc_layer1   = resnet.layer1   # s2: 256ch, 1/4 res (32x32)
        self.enc_layer2   = resnet.layer2   # s3: 512ch, 1/8 res (16x16)
        self.enc_layer3   = resnet.layer3   # s4: 1024ch,1/16 res ( 8x8)
        self.enc_layer4   = resnet.layer4   # raw bridge: 2048ch, 1/32 res (4x4)

        # ---- Bridge ----
        # FIX: ConvBlock(Dropout 0.3) -> Dropout(0.5) -> SE
        # Matches TF:  conv_block[Dropout 0.3] -> Dropout(0.5) -> se_block
        self.bridge_conv    = ConvBlock(2048, 512)      # internal Dropout(0.3)
        self.bridge_dropout = nn.Dropout2d(0.5)         # FIX: extra 0.5 after conv_block
        self.bridge_se      = SEBlock(512)              # SE applied after dropout

        # ---- Decoder ----
        # FIX: channel dims and skip sources corrected to match TF
        #   d1: bridge(512) upsample -> fuse with s4(1024ch) -> out 512ch
        #   d2: d1(512)     upsample -> fuse with s3(512ch)  -> out 256ch
        #   d3: d2(256)     upsample -> fuse with s2(256ch)  -> out 128ch
        #   d4: d3(128)     upsample -> fuse with s1(64ch)   -> out  64ch
        self.decoder1 = DecoderBlock(512,  1024, 512)
        self.decoder2 = DecoderBlock(512,   512, 256)
        self.decoder3 = DecoderBlock(256,   256, 128)
        self.decoder4 = DecoderBlock(128,    64,  64)

        # ---- Final output head ----
        # Upsample 64x64 -> 128x128
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
        # ---- Encoder ----
        x  = self.enc_conv1(x)
        x  = self.enc_bn1(x)
        s1 = self.enc_relu(x)          # 64ch,  1/2 res (64x64)  <- TF conv1_relu
        x  = self.enc_maxpool(s1)

        s2 = self.enc_layer1(x)        # 256ch, 1/4 res (32x32)  <- TF conv2_block3_out
        s3 = self.enc_layer2(s2)       # 512ch, 1/8 res (16x16)  <- TF conv3_block4_out
        s4 = self.enc_layer3(s3)       # 1024ch,1/16 res  (8x8)  <- TF conv4_block6_out
        b  = self.enc_layer4(s4)       # 2048ch,1/32 res  (4x4)  <- TF conv5_block3_out

        # ---- Bridge ----
        # FIX: conv_block[drop 0.3] -> dropout(0.5) -> SE  (matches TF order)
        bridge = self.bridge_conv(b)
        bridge = self.bridge_dropout(bridge)
        bridge = self.bridge_se(bridge)

        # ---- Decoder (FIX: correct TF skip sources) ----
        d1 = self.decoder1(bridge, s4)  # bridge + s4(1024ch) ->  8x8
        d2 = self.decoder2(d1,    s3)   # d1    + s3(512ch)  -> 16x16
        d3 = self.decoder3(d2,    s2)   # d2    + s2(256ch)  -> 32x32
        d4 = self.decoder4(d3,    s1)   # d3    + s1(64ch)   -> 64x64

        # ---- Output head ----
        x = self.final_up(d4)           # 64x64 -> 128x128
        x = self.final_conv(x)
        x = self.final(x)
        x = self.sigmoid(x)

        return x


# -------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    inter  = torch.sum(y_true * y_pred)
    return 1 - (2 * inter + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)


def focal_loss(y_true, y_pred, gamma=2, alpha=0.25, smooth=1e-6):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1).clamp(smooth, 1 - smooth)
    bce    = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    pt     = torch.where(y_true == 1, y_pred, 1 - y_pred)
    return torch.mean(alpha * (1 - pt) ** gamma * bce)


def combined_loss(y_true, y_pred):
    bce = F.binary_cross_entropy(y_pred, y_true)
    return dice_loss(y_true, y_pred) + focal_loss(y_true, y_pred) + 0.1 * bce


# -------------------------------------------------
# METRICS
# -------------------------------------------------
def precision_metric(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    return (tp + 1e-6) / (tp + fp + 1e-6)


def recall_metric(y_true, y_pred, threshold=0.5):
    y_pred = (y_pred > threshold).float()
    tp = torch.sum(y_true * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))
    return (tp + 1e-6) / (tp + fn + 1e-6)


def iou_metric(y_true, y_pred, threshold=0.5):
    y_pred       = (y_pred > threshold).float()
    intersection = torch.sum(y_true * y_pred)
    union        = torch.sum(y_true) + torch.sum(y_pred) - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def f1_metric(y_true, y_pred, threshold=0.5):
    p = precision_metric(y_true, y_pred, threshold)
    r = recall_metric(y_true, y_pred, threshold)
    return 2 * (p * r) / (p + r + 1e-6)


# -------------------------------------------------
# TRAINING EPOCH
# -------------------------------------------------
def train_epoch(model, dataloader, optimizer, epoch, scaler):
    model.train()
    total_loss = total_iou = total_f1 = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [train]')
    for images, masks in pbar:
        # ---- GPU-ONLY: non_blocking async H2D copy (pairs with
        # pin_memory=True on the DataLoader below) ----
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        # ---- AMP: forward pass runs in mixed precision ----
        with torch.amp.autocast('cuda'):
            outputs = model(images)

        # ---- AMP FIX: binary_cross_entropy (used inside combined_loss)
        # is on autocast's blocklist -- it refuses to run on
        # already-sigmoided probabilities under fp16 because it's
        # numerically unstable there. So the loss is computed OUTSIDE
        # the autocast region, in normal fp32. .float() guards against
        # outputs having been left in fp16 by the autocast forward pass. ----
        loss = combined_loss(masks, outputs.float())

        # ---- AMP: scaled backward pass + optimizer step ----
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics computed on the same outputs as before AMP was added.
        iou = iou_metric(masks, outputs)
        f1  = f1_metric(masks, outputs)

        total_loss += loss.item()
        total_iou  += iou.item()
        total_f1   += f1.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}",
                         iou=f"{iou.item():.4f}",
                         f1=f"{f1.item():.4f}")

    n = len(dataloader)
    return total_loss / n, total_iou / n, total_f1 / n


# -------------------------------------------------
# VALIDATION EPOCH
# -------------------------------------------------
def validate_epoch(model, dataloader):
    model.eval()
    total_loss = total_iou = total_f1 = 0.0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            # ---- GPU-ONLY: non_blocking async H2D copy ----
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            # ---- AMP: mixed-precision forward pass (no scaler needed,
            # there's no backward pass during validation) ----
            with torch.amp.autocast('cuda'):
                outputs = model(images)

            # ---- AMP FIX: same reason as train_epoch -- BCE is
            # blocklisted under autocast, so compute loss outside it,
            # in fp32. ----
            loss = combined_loss(masks, outputs.float())

            total_loss += loss.item()
            total_iou  += iou_metric(masks, outputs).item()
            total_f1   += f1_metric(masks, outputs).item()

    n = len(dataloader)
    return total_loss / n, total_iou / n, total_f1 / n


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    print_hardware_info(header="GPU / HARDWARE INFO")
    print(f"Model type   : UNet (ResNet50 + Attention U-Net + SE Block)")
    print(f"Model name   : {model_name}")

    all_img_paths = sorted(glob.glob(f"{DATA_PATH}/images/*.tif"))
    np.random.shuffle(all_img_paths)

    split_idx   = int(0.8 * len(all_img_paths))
    train_paths = all_img_paths[:split_idx]
    val_paths   = all_img_paths[split_idx:]

    # ---- Normalization & model init ----
    if os.path.exists(MODEL_SAVE) and os.path.exists(MEAN_STD_FILE):
        MEAN, STD = load_mean_std_from_file(MEAN_STD_FILE)
        model = SolarModel().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
        print("Model loaded from checkpoint.")
    else:
        MEAN, STD = compute_mean_std(train_paths)
        save_mean_std(MEAN, STD)
        model = SolarModel().to(DEVICE)
        print("New model initialised.")

    # ---- Datasets & loaders ----
    train_dataset = SolarDataset(train_paths, MEAN, STD, is_train=True)
    val_dataset   = SolarDataset(val_paths,   MEAN, STD, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                  patience=5, min_lr=1e-6)

    # ---- AMP: gradient scaler, guards against fp16 underflow ----
    scaler = torch.amp.GradScaler('cuda')

    # ---- Training loop ----
    best_val_iou       = 0.0
    patience_counter   = 0
    early_stop_patience = 10

    # ---- TIMER: track per-epoch durations + overall training start ----
    epoch_durations = []
    training_start  = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_iou, train_f1 = train_epoch(model, train_loader, optimizer, epoch, scaler)
        val_loss,   val_iou,   val_f1   = validate_epoch(model, val_loader)

        duration = time.time() - t0
        epoch_durations.append(duration)
        avg_epoch_time = sum(epoch_durations) / len(epoch_durations)

        log_line = (f"Epoch:{epoch} | time:{duration:.1f}s | avg_epoch_time:{avg_epoch_time:.1f}s | "
                    f"loss:{train_loss:.4f} | val_loss:{val_loss:.4f} | "
                    f"iou:{train_iou:.4f} | val_iou:{val_iou:.4f} | "
                    f"f1:{train_f1:.4f} | val_f1:{val_f1:.4f}")
        print(log_line)

        with open(TRAIN_LOG, "a") as f:
            f.write(log_line + "\n")

        scheduler.step(val_loss)

        if val_iou > best_val_iou:
            best_val_iou    = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE)
            print(f"  -> Best model saved (val_iou: {val_iou:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    # ---- TIMER: final summary ----
    total_training_time = time.time() - training_start
    avg_epoch_time       = sum(epoch_durations) / len(epoch_durations)

    hrs, rem = divmod(total_training_time, 3600)
    mins, secs = divmod(rem, 60)

    print("Training finished.")

    hw_lines = print_hardware_info(header="FINAL GPU / HARDWARE INFO")
    print(f"Model type           : UNet (ResNet50 + Attention U-Net + SE Block)")
    print(f"Model name           : {model_name}")
    print(f"Epochs run           : {len(epoch_durations)}")
    print(f"Average epoch time   : {avg_epoch_time:.1f}s")
    print(f"Total training time  : {int(hrs)}h {int(mins)}m {secs:.1f}s "
          f"({total_training_time:.1f}s)")

    with open(TRAIN_LOG, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write("Model type          : UNet (ResNet50 + Attention U-Net + SE Block)\n")
        f.write(f"Model name          : {model_name}\n")
        f.write(f"Hardware            : {hardware_summary_line()}\n")
        for line in hw_lines:
            f.write(f"  {line}\n")
        f.write(f"Epochs run          : {len(epoch_durations)}\n")
        f.write(f"Average epoch time  : {avg_epoch_time:.1f}s\n")
        f.write(f"Total training time : {int(hrs)}h {int(mins)}m {secs:.1f}s "
                f"({total_training_time:.1f}s)\n")
        f.write("=" * 60 + "\n")