import os
import torch
import argparse
import yaml
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.io import read_image
import torchvision.utils as vutils  # 用于保存 SR 图像
from source.models.get_model import get_model


class SimpleImageFolder(torch.utils.data.Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = read_image(img_path).float() / 255.0  # [0, 1]
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            img = torch.zeros(3, 64, 64)  # fallback dummy
        return img, img_path


# -------------------- 参数解析 --------------------
parser = argparse.ArgumentParser(description='Inference Performance Only')
parser.add_argument('--config', type=str, required=True, help='YAML config file')
parser.add_argument('--checkpoint', type=str, required=True, help='Pretrained model path')
parser.add_argument('--input-dir', type=str, required=True, help='Directory of LR images')
parser.add_argument('--gpu_ids', type=int, default=0, help='GPU ID')
parser.add_argument('--fp16', type=bool, default=True, help='Use FP16')
parser.add_argument('--save-dir', type=str, default=None, help='If set, save SR outputs to this directory')
args = parser.parse_args()

# -------------------- 合并 YAML 配置 --------------------
if args.config:
    opt = vars(args)
    yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(yaml_args)

# -------------------- 设置设备 --------------------
device = torch.device(f"cuda:{args.gpu_ids}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# -------------------- 加载数据 --------------------
dataset = SimpleImageFolder(args.input_dir)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

# -------------------- 加载模型 --------------------
model = get_model(args, device)
print(f"Loading pretrained model from {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
else:
    model.load_state_dict(ckpt, strict=False)
model = model.to(device).eval()
if args.fp16:
    model.half()

# -------------------- 推理计时 --------------------
print("Starting inference...")
times = []

with torch.no_grad():
    for idx, (lr_patch, img_path) in enumerate(tqdm(test_dataloader, desc="Running inference")):
        lr_patch = lr_patch.to(device)
        if args.fp16:
            lr_patch = lr_patch.half()

        start_time = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.fp16):
            sr_output = model(lr_patch)
        torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)

        img_name = os.path.basename(img_path[0])
        print(f"[Image] {img_name} - Time: {elapsed:.4f}s")

        # ✅ 可选保存
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, img_name)
            vutils.save_image(sr_output.clamp(0, 1), save_path)

# -------------------- 总结统计 --------------------
num_images = len(times)
total_sec = sum(times)
avg_time = total_sec / num_images
fps = num_images / total_sec
min_time = min(times)
max_time = max(times)

print(f"\nTotal images: {num_images}")
print(f"Total inference time: {total_sec:.4f} s")
print(f"Average time per image: {avg_time:.4f} s")
print(f"FPS (frames per second): {fps:.2f}")
print(f"Min / Max time per image: {min_time:.4f}s / {max_time:.4f}s")
