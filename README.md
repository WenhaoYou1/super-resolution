
# Adaptive Lightweight Real-Time Image Super-Resolution Network for 4K Images with ECA layer

### Dataset

You can download Div2k dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
You can download Benchmark dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

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

Initial training:
```bash
python train.py \
  --config "configs/x3_final/l1_x3_200_div2k.yml" \
  --gpu_ids 0
```

If training is interrupted, resume training with:
```bash
python train.py \
  --config "configs/x3_final/l1_x3_200_div2k.yml" \
  --gpu_ids 0 \
  --resume /root/.../PlainRepConv_x3
```

To resume with **wandb** tracking:
```bash
python train.py \
  --config "configs/x3_final/l1_x3_200_div2k.yml" \
  --gpu_ids 0 \
  --resume_wandb 51fm1awe \
  --resume /root/.../PlainRepConv_x3
```

### Evaluation

To evaluate super-resolved images and compute PSNR/SSIM/LPIPS:
```bash
python Auto_sr_demo.py \
  --model _X3 \
  --config "configs/x3_final/ECA_l1_x3_200_div2k.yml" \
  --checkpoint /root/.../ECAPlainRepConv_x3/models/model_x3_best_submission_deploy.pt
```

In this example, a deploy-phase model is used. `Auto_sr_demo.py` performs super-resolution on LR images and evaluates multiple benchmark datasets. By default, it evaluates the following datasets:
```python
["set14", "Urban100", "DIV2K100"]
```
You can modify this list based on your dataset setup. The example command performs SR on LR X3 images and compares them with their original HR counterparts.

### Model inference time
To measure the inference time:
```bash
python inference_time_test.py \
  --config "configs/x4_final/l1_x4_200_div2k.yml" \
  --checkpoint "experiments/Val_X4_Best/PlainRepConv_x4/models/model_x4_best_submission_deploy.pt" \
  --input-dir "dataset/DIV2K/val_phase_X4"
```
Optional command. If you add this, it will save the SR images while inferring.
```bash
  --save-dir /Your_path
```
