@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
python train.py --gpu_ids 0 --config ./configs/x2_final/repConv_x2_m4c32_relu_div2k_warmup_lr5e-4_b8_p384_normalize.yml
pause