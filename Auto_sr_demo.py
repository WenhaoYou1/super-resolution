import os
import torch
import pathlib
import logging
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import source.utils.dataset as dd
from source.utils import util_logger
from source.utils import util_image as util
from source.utils.model_summary import get_model_flops
from source.models.get_model import get_model
import yaml

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from source.losses import LPIPSLoss

def main(args):
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    util_logger.logger_info("NTIRE2023-RTSR", log_path=os.path.join(args.save_dir, f"Submission.txt"))
    logger = logging.getLogger("NTIRE2023-RTSR")

    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:{}'.format(0))

    if not args.bicubic:
        if args.config:
            opt = vars(args)
            yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
            opt.update(yaml_args)
        model = get_model(args, device, mode='Deploy')
        if args.checkpoint is not None:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)
        number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
        logger.info('Params number: {}'.format(number_parameters))
        print(model)

    lpips_fn = LPIPSLoss(net='vgg', device=device)

    base_dir = args.lr_base_dir
    scale = f"X{args.scale}"
    sets = ['Set14', 'DIV2K100', 'Urban100']

    for dataset_name in sets:
        lr_dir = os.path.join(base_dir, dataset_name, 'LR_bicubic', scale)
        hr_dir = os.path.join(base_dir, dataset_name, 'HR')
        save_dir = os.path.join(base_dir, dataset_name, f"HR_{scale}_{args.model}")
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

        dataset = dd.SRDataset(lr_images_dir=lr_dir, n_channels=3, transform=None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

        avg_psnr, avg_ssim, avg_lpips = 0.0, 0.0, 0.0
        count = 0

        with torch.no_grad():
            for img_L, img_path in tqdm(dataloader):
                img_name_ext = os.path.basename(img_path[0])
                img_name = os.path.splitext(img_name_ext)[0]

                img_L = img_L.to(device, non_blocking=True)

                if args.bicubic:
                    img_E = F.interpolate(img_L, scale_factor=args.scale, mode="bicubic", align_corners=False)
                else:
                    if args.fp16:
                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            img_E = model(img_L)
                    else:
                        img_E = model(img_L)

                img_E_np = util.tensor2uint(img_E)

                if args.save_sr:
                    util.imsave(img_E_np, os.path.join(save_dir, img_name + ".png"))

                hr_img_path = os.path.join(hr_dir, img_name + '.png')
                if os.path.exists(hr_img_path):
                    hr_img = util.imread_uint(hr_img_path, n_channels=3)
                    psnr = compare_psnr(hr_img, img_E_np, data_range=255)
                    ssim = compare_ssim(hr_img, img_E_np, multichannel=True)
                    lpips_val = lpips_fn(
                        torch.tensor(img_E_np / 255.0).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1,
                        torch.tensor(hr_img / 255.0).permute(2, 0, 1).unsqueeze(0).to(device).float() * 2 - 1
                    ).item()
                    avg_psnr += psnr
                    avg_ssim += ssim
                    avg_lpips += lpips_val
                    count += 1

        if count > 0:
            print(f"[{dataset_name}] Avg PSNR: {avg_psnr/count:.2f}, Avg SSIM: {avg_ssim/count:.4f}, Avg LPIPS: {avg_lpips/count:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="./results/")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--lr-base-dir", type=str, default='./dataset/test')
    parser.add_argument("--save-sr", action="store_true", default=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--crop-size", type=int, nargs="+", default=[720, 1280])
    parser.add_argument("--bicubic", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--model", type=str, required=True, help="Name of the model to tag output folder")
    args = parser.parse_args()

    main(args)
