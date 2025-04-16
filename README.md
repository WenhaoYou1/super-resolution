
# Improved Lightweight Real-Time Image Super-Resolution Network for 4K Images with ECA layer

### Dependencies & Installation

Download dependencies:
```bash
pip install -r requirements.txt
```

### Model Configuration

Model configuration files are located in the `configs/` directory.
- Config files with `ECA` in the name include the ECA module.
- Config files with `unknown` indicate training on the DIV2K **unknown** dataset.
- Others default to the DIV2K **bicubic** dataset.

### Training

Initial training command:
```bash
python train.py --config "configs/x3_final/l1_x3_200_div2k.yml" --gpu_ids 0
```

If training is interrupted, resume training with:
```bash
python train.py --config "configs/x3_final/l1_x3_200_div2k.yml" --gpu_ids 0 --resume /root/autodl-tmp/super-resolution/LRSRN/experiments/Val_X3_Best/PlainRepConv_x3
```

To resume with **wandb** tracking:
```bash
python train.py --config "configs/x3_final/l1_x3_200_div2k.yml" --gpu_ids 0 --resume_wandb 51fm1awe --resume /root/autodl-tmp/super-resolution/LRSRN/experiments/Val_X3_Best/PlainRepConv_x3
```

### Evaluation

To evaluate super-resolved images and compute PSNR/SSIM/LPIPS:
```bash
python Auto_sr_demo.py --model _X3 --config "configs/x3_final/ECA_l1_x3_200_div2k.yml" --checkpoint /root/autodl-tmp/super-resolution/LRSRN/experiments/Val_X3_Best/ECAPlainRepConv_x3/models/model_x3_best_submission_deploy.pt
```

In this example, a deploy-phase model is used. `Auto_sr_demo.py` performs super-resolution on LR images and evaluates multiple benchmark datasets. By default, it evaluates the following datasets:
```python
["set14", "Urban100", "DIV2K100"]
```
You can modify this list based on your dataset setup. The script performs SR on LR X3 images and compares them with their original HR counterparts.

### Model inference time

You can profile inference performance by measuring per-image time and FPS with scripts like `inference_time_test.py`.

### Dataset of SR

You can download Div2k dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
You can download Benchmark dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

Combined test dataset from Drive [Link](https://drive.google.com/file/d/1feZltvT0COZ87SjMxJpGMsrWTk1uLibD/view?usp=sharing)
 - Combined dataset includes: 
    - Train: DIV2K train set (full 800), Flickr train set (2650 full), GTA (train seq 00 ~ 19 seq) sample 361, LSDIR (first 1000)
    - Val: DIV2K val set (full 100), Flickr val (100), GTA (90), LSDIR(100)

Path of Dataset must be set in `./config/*name_of_yaml*.yaml`

### Reference:
```
@InProceedings{Gankhuyag_2023_CVPR,
    author    = {Gankhuyag, Ganzorig and Yoon, Kihwan and Park, Jinman and Son, Haeng Seon and Min, Kyoungwon},
    title     = {Lightweight Real-Time Image Super-Resolution Network for 4K Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {1746-1755}
}
```

### Our Team's other works
Our New RTSR Network Achieve 1st Place in CVPR2024 Workshop 🎉

[CASR : Efficient Cascade Network Structure with Channel Aligned method for 4K Real-Time Single Image Super-Resolution](https://github.com/rlghksdbs/CASR)
