import os
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
import h5py
import matplotlib.pyplot as plt
from io import BytesIO
import torch.nn.functional as F
from joblib import Parallel, delayed
from models.AttriMIL import AttriMIL
import base64

# --- Config ---
PATCH_SIZE = 256
MAX_DIM = 65000
JPEG_QUALITY = 80
BLACK_THRESHOLD = 0.95
WHITE_THRESHOLD = 0.99
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Image.MAX_IMAGE_PIXELS = None

# --- 모델 로드 ---
model = AttriMIL(n_classes=5, dim=512).to(device)
ckpt_path = "./save_weights/attrimil_final.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

# --- ResNet feature extractor ---
resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- 전처리 함수들 ---
def resize_and_save(image_path, resized_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w > MAX_DIM or h > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img.save(resized_path, quality=JPEG_QUALITY)

def is_black(patch):
    return np.mean(patch < 10) > BLACK_THRESHOLD

def is_white(patch):
    return np.mean(patch > 245) > WHITE_THRESHOLD

def create_patches(image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape
    count = 0
    for y in range(0, h, PATCH_SIZE):
        for x in range(0, w, PATCH_SIZE):
            patch = img_np[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            if patch.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                continue
            if is_black(patch) or is_white(patch):
                continue
            Image.fromarray(patch).save(os.path.join(save_dir, f"patch_{x}_{y}.png"))
            count += 1
    return count

def extract_features(patch_dir, h5_path):
    feats, coords = [], []
    for fname in sorted(os.listdir(patch_dir)):
        if not fname.endswith(".png"):
            continue
        x, y = map(int, fname.replace(".png", "").split("_")[1:])
        img = Image.open(os.path.join(patch_dir, fname)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(img_tensor).squeeze().cpu().numpy()
        feats.append(feat)
        coords.append([x, y])
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('features', data=np.array(feats))
        f.create_dataset('coords', data=np.array(coords))

def add_nearest(h5_path):
    with h5py.File(h5_path, 'r') as f:
        coords = np.array(f['coords'])

    nearest = []
    for idx, p in enumerate(coords):
        neighbors = [idx]
        for dx, dy in [(0,-PATCH_SIZE),(0,PATCH_SIZE),(-PATCH_SIZE,0),(PATCH_SIZE,0),
                       (-PATCH_SIZE,-PATCH_SIZE),(PATCH_SIZE,-PATCH_SIZE),
                       (-PATCH_SIZE,PATCH_SIZE),(PATCH_SIZE,PATCH_SIZE)]:
            neighbor = p + np.array([dx, dy])
            loc = np.where(np.all(coords == neighbor, axis=1))[0]
            neighbors.append(loc[0] if len(loc) else idx)
    with h5py.File(h5_path, 'a') as f:
        f.create_dataset('nearest', data=np.array(nearest))

def merge_h5(input_h5, output_h5):
    with h5py.File(input_h5, 'r') as f:
        coords = np.array(f['coords'])
        features = np.array(f['features'])
        nearest = np.array(f['nearest'])
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('coords', data=coords)
        f.create_dataset('features', data=features)
        f.create_dataset('nearest', data=nearest)

def infer_and_get_attention(final_h5_path, resized_img_path):
    with h5py.File(final_h5_path, 'r') as f:
        features = torch.tensor(np.array(f['features']), dtype=torch.float32).to(device)
        coords_np = np.array(f['coords'])

    with torch.no_grad():
        logits, _, _, attribute_score, _ = model(features)

    pred_class = logits.argmax(dim=1).item()
    softmax_probs = F.softmax(logits, dim=1).cpu().numpy().flatten()

    attention_scores = attribute_score[0, pred_class].cpu().numpy().flatten()
    attention_norm = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
    attention_norm = np.power(attention_norm, 0.5)

    wsi_img = Image.open(resized_img_path).convert("RGB")
    base_width, h_size = wsi_img.size
    x_min, x_max = coords_np[:, 0].min(), coords_np[:, 0].max()
    y_min, y_max = coords_np[:, 1].min(), coords_np[:, 1].max()

    scaled_x = (coords_np[:, 0] - x_min) / (x_max - x_min + 1e-8) * base_width
    scaled_y = (coords_np[:, 1] - y_min) / (y_max - y_min + 1e-8) * h_size

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(wsi_img, alpha=0.5)
    scatter = ax.scatter(scaled_x, scaled_y, c=attention_norm, cmap='plasma', s=15, alpha=0.8, edgecolors='none')
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label='Attention Weight')
    plt.axis('off')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    attention_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return pred_class, softmax_probs.tolist(), attention_base64

