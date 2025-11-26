"""
Gaze Pipeline Trainer
======================
Single-file training application that builds a gaze-estimation pipeline:
- Face detection / crop using YOLOv8 (ultralytics)
- Head-pose estimation via 2D facial landmarks + solvePnP
- Gaze regression/classification model (AFF-Net style): ResNet18 backbones + feature fusion

Features:
- Supports regression (screen coordinates normalized to [0,1]) or region-classification (discrete screen zones)
- Uses pretrained backbones where available
- Training loop with validation, checkpointing

Dataset format (simple):
CSV with columns: image_path, x_norm, y_norm  OR image_path, region_id
- If using regression: set mode=regression and provide normalized x,y (0..1) coordinates on screen
- If using classification: set mode=classification and provide integer region ids (0..N-1)

Requirements (install):
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip install ultralytics==8.0.20  
pip install pandas
pip install scikit-learn
pip install face-alignment

Run example:
python gaze_pipeline_trainer.py --csv dataset.csv --mode regression --epochs 30 --batch 32 --outdir checkpoints

Notes:
- You must have a GPU for reasonable speed. YOLOv8 requires ultralytics package.
- You can replace the ResNet backbone with heavier models or swap the gaze head with a prebuilt GazeTR if you prefer.
"""

import os
import argparse
import math
import logging
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# ultralytics YOLOv8
from ultralytics import YOLO

# face landmarks (face_alignment) - lightweight and reliable
import face_alignment
print("All libraries loaded")


# main_logger = logging.getLogger("Main executions")
additiona_logger = logging.getLogger("Additional info")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
additiona_logger.addHandler(console_handler)

# ------------------------ Utilities ------------------------

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ------------------------ Head pose (solvePnP) ------------------------
# We'll estimate head pose using 2D facial landmarks (68 points) -> 3D model points
# Returns yaw, pitch, roll in degrees

FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # nose tip
    (0.0, -330.0, -65.0),        # chin
    (-225.0, 170.0, -135.0),     # left eye left corner
    (225.0, 170.0, -135.0),      # right eye right corner
    (-150.0, -150.0, -125.0),    # left mouth corner
    (150.0, -150.0, -125.0)      # right mouth corner
], dtype=np.float64)

LANDMARKS_INDEXES = [30, 8, 36, 45, 48, 54]  # correspond to model points


def estimate_head_pose(landmarks: np.ndarray, image_size: Tuple[int, int]) -> Tuple[float,float,float]:
    """Estimate yaw, pitch, roll from 2D landmarks using solvePnP."""
    img_h, img_w = image_size
    image_points = landmarks[LANDMARKS_INDEXES].astype(np.float64)

    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4,1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2,1], rmat[2,2])
        y = math.atan2(-rmat[2,0], sy)
        z = math.atan2(rmat[1,0], rmat[0,0])
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1])
        y = math.atan2(-rmat[2,0], sy)
        z = 0

    # convert to degrees
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)
    return yaw, pitch, roll

def crop_eye(img, pts, scale=1.8):
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx, cy = (x_min + x_max)/2, (y_min + y_max)/2
    w, h = (x_max - x_min)*scale, (y_max - y_min)*scale
    x1, y1 = int(cx - w/2), int(cy - h/2)
    x2, y2 = int(cx + w/2), int(cy + h/2)
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((32,32,3), dtype=np.uint8)
    return cv2.resize(crop, (64,64))

# ------------------------ Dataset ------------------------

class GazeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode='regression', transform=None, yolo_model=None, fa=None, crop_size=224):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.yolo = yolo_model
        self.fa = fa
        self.crop_size = crop_size

    def __len__(self):
        return len(self.df)

    def _detect_face_and_landmarks(self, img_rgb: np.ndarray):
        # face detection with YOLO model (expects BGR or RGB, ultralytics accepts numpy RGB)
        # returns face crop and landmarks
        res = self.yolo.predict(source=img_rgb, conf=0.35, max_det=1, imgsz=640, verbose=False)
        if len(res) == 0 or len(res[0].boxes) == 0:
            return None, None, None
        box = res[0].boxes[0].xyxy.cpu().numpy().astype(int).flatten()
        x1, y1, x2, y2 = box
        h, w, _ = img_rgb.shape
        # pad and crop
        pad = int(0.2 * max(y2-y1, x2-x1))
        x1p = max(0, x1-pad); y1p = max(0, y1-pad)
        x2p = min(w, x2+pad); y2p = min(h, y2+pad)
        face_crop = img_rgb[y1p:y2p, x1p:x2p].copy()

        # get landmarks on the whole image, then rebase to crop
        # face_alignment expects RGB
        try:
            lm_all = self.fa.get_landmarks_from_image(img_rgb)[0]  # (68,2)
        except Exception:
            lm_all = None
        if lm_all is None:
            return face_crop, None, (x1p, y1p, x2p, y2p)
        # shift landmarks to crop coordinates
        lm_crop = lm_all - np.array([x1p, y1p])
        return face_crop, lm_crop, (x1p, y1p, x2p, y2p)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row['image_name'])  # RGB
        face_crop, lm_crop, bbox = self._detect_face_and_landmarks(img)
        if face_crop is None:
            # fallback: use center crop
            print(f'Face crop is None for image {row["image_name"]}, using center crop')
            h,w,_ = img.shape
            s = min(h,w)
            cy, cx = h//2, w//2
            face_crop = img[cy-s//2:cy+s//2, cx-s//2:cx+s//2]
            lm_crop = None
            bbox = [0, 0, 1, 1]

        # compute head pose from landmarks if available
        if lm_crop is not None:
            yaw,pitch,roll = estimate_head_pose(lm_crop, (face_crop.shape[0], face_crop.shape[1]))
            left_eye_pts = lm_crop[36:42]
            right_eye_pts = lm_crop[42:48]
            left_eye = crop_eye(face_crop, left_eye_pts)
            right_eye = crop_eye(face_crop, right_eye_pts)
        else:
            yaw,pitch,roll = 0.0,0.0,0.0
            left_eye = np.zeros((64,64,3), dtype=np.uint8)
            right_eye = np.zeros((64,64,3), dtype=np.uint8)

        # prepare inputs: face image (resized) and head pose vector
        face_resized = cv2.resize(face_crop, (self.crop_size, self.crop_size))
        face_tensor = T.ToTensor()(face_resized).float()  # 0..1
        if self.transform is not None:
            face_tensor = self.transform(face_tensor)

        head_pose = np.array([yaw, pitch, roll], dtype=np.float32) / 180.0  # normalize to [-1,1]
        head_pose = torch.from_numpy(head_pose)

        h, w, _ = img.shape
        x1, y1, x2, y2 = bbox
        bbox_norm = np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)
        bbox_tensor = torch.from_numpy(bbox_norm)

        left_eye_t = T.ToTensor()(left_eye).float()
        right_eye_t = T.ToTensor()(right_eye).float()

        if self.mode == 'regression':
            y = torch.tensor([float(row['x_center_norm']), float(row['y_center_norm'])], dtype=torch.float32)
        else:
            y = torch.tensor(int(row['region_id']), dtype=torch.long)

        return face_tensor, left_eye_t, right_eye_t, head_pose, bbox_tensor, y

# ------------------------ Model (AFF-Net simplified) ------------------------

class AFFGazeNet(nn.Module):
    def __init__(self, mode='regression', n_regions=9, pretrained=True):
        super().__init__()
        self.mode = mode
        backbone = models.resnet18(pretrained=pretrained)
        # remove fc
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # output feat map
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        feat_dim = 512

        # head pose MLP
        self.pose_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.eye_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim + 64 + 128+64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        if mode == 'regression':
            self.out = nn.Linear(128, 2)  # x,y
        else:
            self.out = nn.Linear(128, n_regions)

    def forward(self, face_img, left_eye, right_eye, head_pose, bbox):
        # face_img: Bx3xHxW
        feats = self.backbone(face_img)  # B x C x h x w
        feats = self.avgpool(feats).view(feats.size(0), -1)
        le = self.eye_encoder(left_eye)
        re = self.eye_encoder(right_eye)
        eyes_feat = torch.cat([le.view(le.size(0), -1), re.view(re.size(0), -1)], dim=1)
        pose = self.pose_mlp(head_pose)
        bbox_feat = self.bbox_mlp(bbox)

        cat = torch.cat([feats, eyes_feat, pose, bbox_feat], dim=1)
        f = self.fusion(cat)
        out = self.out(f)
        if self.mode == 'regression':
            out = torch.sigmoid(out)  # normalized to 0..1 for screen coords
        return out

# ------------------------ Training utilities ------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        face,left_eye_t, right_eye_t, pose, bbox, y = [b.to(device) for b in batch]
        preds = model(face,left_eye_t, right_eye_t, pose, bbox)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * face.size(0)
        additiona_logger.debug(f'Batch loss: {loss.item():.4f}, total_loss: {total_loss:.4f}')
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, mode='regression'):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            face,left_eye, right_eye, pose, bbox,  y = [b.to(device) for b in batch]
            preds = model(face, left_eye, right_eye, pose, bbox)
            loss = criterion(preds, y)
            total_loss += loss.item() * face.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    import numpy as np
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets) if mode=='regression' else np.concatenate(all_targets)
    return total_loss / len(loader.dataset), all_preds, all_targets

# ------------------------ Main ------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='CSV file with dataset annotations')
    p.add_argument('--mode', choices=['regression','classification'], default='regression')
    p.add_argument('--train_dir', type=str, default=None, help='Path to directory with training images (if provided, CSV should contain only image filenames)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--outdir', default='checkpoints')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n_regions', type=int, default=9)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    main_logger = setup_main_logger(
        name="Main executions",
        log_file=os.path.join(args.outdir, "mainlogs.txt"),
        file_level=logging.DEBUG,
        console_level=logging.INFO
    )
    df = pd.read_csv(args.csv)
    if args.train_dir is not None:
        df['image_name'] = df['image_name'].apply(lambda x: os.path.join(args.train_dir, x))

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)

    # load YOLO face model (uses ultralytics)
    print('Loading YOLO face model...')
    yolo = YOLO('./models/yolov11n-face.pt')  # please download a face model or use yolov11n.pt and filter by class

    # face landmarks
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    transform = T.Compose([
        # you can add augmentations here (color jitter, random crop, etc.)
    ])

    train_ds = GazeDataset(train_df, mode=args.mode, transform=transform, yolo_model=yolo, fa=fa)
    val_ds = GazeDataset(val_df, mode=args.mode, transform=None, yolo_model=yolo, fa=fa)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Using device', device)
    if torch.cuda.is_available():
        print('Using GPU')
    else: os.exit('No GPU found.')

    model = AFFGazeNet(mode=args.mode, n_regions=args.n_regions, pretrained=True).to(device)

    if args.mode == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, preds, targets = eval_epoch(model, val_loader, criterion, device, mode=args.mode)

        main_logger.info(f"Epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # save checkpoint
        ckpt_path = os.path.join(args.outdir, f'epoch_{epoch:02d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
        }, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, 'best_model.pth'))

    main_logger.info('Training finished. Best val loss: %.4f', best_val)




def setup_main_logger(
    name="main",
    log_file="main.txt",
    file_level=logging.DEBUG,
    console_level=logging.INFO
):
    # создаём логер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # общий уровень

    # === формат вывода ===
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # === обработчик для файла ===
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # === обработчик для консоли ===
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # === добавляем обработчики ===
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    main()
