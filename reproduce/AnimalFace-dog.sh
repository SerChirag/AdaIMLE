#!/bin/bash

python train.py --hps fewshot \
    --data_root ./datasets/AnimalFace-dog \
    --change_coef 0.01 \
    --force_factor 100 \
    --imle_staleness 5 \
    --imle_force_resample 15  \
    --lr 0.0001 \