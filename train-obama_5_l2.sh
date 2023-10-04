python train.py --hps fewshot \
    --save_dir ./results/obama-splatter-5:l2/ \
    --data_root ./datasets/100-shot-obama/ \
    --change_coef 0.01 \
    --force_factor 50 \
    --imle_staleness 5 \
    --imle_force_resample 15  \
    --lr 0.0002 \
    --iters_per_ckpt 25000 --iters_per_images 5000 --iters_per_save 1000 \
    --comet_api_key '2SDNAxxWevz4p6SThRTEM2KlD' \
    --comet_name 'obama-splatter-5:l2' \
    --num_images_visualize 10 \
    --num_rows_visualize 5 \
    --imle_db_size 128 \
    --use_comet True\
    --use_splatter True \
    --search_type 'l2' \
    --angle 5.0 \
    --fid_freq 100 \
    --restore_path ./results/obama-splatter-5:l2/train/latest-model.th \
    --restore_ema_path ./results/obama-splatter-5:l2/train/latest-model-ema.th \
    --restore_optimizer_path ./results/obama-splatter-5:l2/train/latest-opt.th \
    --restore_threshold_path ./results/obama-splatter-5:l2/train/latent/0-threshold_latest.npy \
    --restore_log_path ./results/obama-splatter-5:l2/train/log.jsonl \
    --comet_experiment_key 'd79661020cf44aa5bf95011bbfaec7f2' 

    # --restore_optimizer_path ./results/obama/train/latest-opt.th \
    # --restore_threshold_path ./results/obama/train/latent/0-threshold_latest.npy 

    # --restore_path ./pretrained/100-shot-obama.th \
    # --restore_ema_path ./pretrained/100-shot-obama.th
