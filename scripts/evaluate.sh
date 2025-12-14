#!/bin/bash

CONFIG_VALUES=$(python -c "
from config import dataset_root, predictor_path
print(f'--dataset_root={dataset_root}')
print(f'--predictor_path={predictor_path}')
" | tr '\n' ' ')


CUDA_VISIBLE_DEVICES=0 python train.py \
 --exp_root exp \
 --runner_name OWDFA_CAL \
 --exp_name Protocol1_evaluate \
 --teacher_temp 0.1 \
 --student_temp 1.0 \
 --ccr_weight 0.2 \
 --pseudo_epohcs 5 \
 --gamma 0.9 \
 --model_num_classes 41 \
 --protocol 1 \
 --eval_only \
 --eval_ckpt "your_path_models.ckpt" \
 $CONFIG_VALUES \