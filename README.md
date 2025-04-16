<div align="center">

# Lightweight Real-Time Image Super-Resolution Network for 4K Images

### Page & Presentation Link
<div>
    <h4 align="center">
        <a href="https://rlghksdbs.github.io/lrsrn_page/" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Gankhuyag_Lightweight_Real-Time_Image_Super-Resolution_Network_for_4K_Images_CVPRW_2023_paper.pdf" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg">
        </a>
    </h4>
</div>
</div>

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

### Docker Setting
```
docker build --tag ntire2023 .
nvidia-docker run --name ntire2023 -it --gpus all --ipc=host --pid=host -v /your/data/path/:/dataset -v /your/sorce_code/:/source_code --shm-size=64g ntire2023:latest
pip install -U numpy

##If you use RTX A6000
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
```

### Dataset of SR

You can download Div2k dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar)
You can download Benchmark dataset from Web [Link](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

Combined test dataset from Drive [Link](https://drive.google.com/file/d/1feZltvT0COZ87SjMxJpGMsrWTk1uLibD/view?usp=sharing)
 - Combined dataset includes: 
    - Train: DIV2K train set (full 800), Flickr train set (2650 full), GTA (train seq 00 ~ 19 seq) sample 361, LSDIR (first 1000)
    - Val: DIV2K val set (full 100), Flickr val (100), GTA (90), LSDIR(100)

Path of Dataset must be set in `./config/*name_of_yaml*.yaml`

### Dataset preparation for Noised LR images
You can generate LR images with compression noise:
```bash
## LR path & HR path must be set by manually
python source/data/prepare_data.py 
```

### Testing
Set parameters in `./config/config_base_test.yaml`
```bash
## For test your model use sr_demo to check inference time.
python sr_demo.py
```

### Check Result
Validation result image, Test result image, Best weight, Last weight and log files saved in `./output/{DATE_of_TODAY}/{Last_folder}` folder.
Wandb result [WANDB](https://wandb.ai/iilab/ECCV_MAI2020_SR)

### Profilling model inference time
You can check ideal model inference time by pytorch profiling tool.
```bash
## If you set all settings correct
python profiller_model.py
```

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
