python train.py --hps fewshot \
    --save_dir ./new-vanilla-results/ffhq-og-no-snoise-3ff-90/ \
    --data_root ./datasets/ffhq/ \
    --change_coef 0.02 \
    --force_factor 3 \
    --imle_staleness 5 \
    --imle_force_resample 15  \
    --lr 0.0002 \
    --iters_per_ckpt 50000 --iters_per_images 5000 --iters_per_save 1000 \
    --comet_api_key '2SDNAxxWevz4p6SThRTEM2KlD' \
    --comet_name 'ffhq-og-no-snoise-3ff-90' \
    --num_images_visualize 10 \
    --num_rows_visualize 5 \
    --imle_db_size 256 \
    --use_comet True \
    --search_type 'l2' \
    --fid_freq 100 \
    --use_adaptive False \
    --n_batch 12 \
    --num_epochs 6000 \
    --use_snoise False \
    --use_angular_resample True \
    --max_sample_angle 90.0 

    # --restore_optimizer_path ./results/obama/train/latest-opt.th \
    # --restore_threshold_path ./results/obama/train/latent/0-threshold_latest.npy 

    # --restore_path ./pretrained/100-shot-obama.th \
    # --restore_ema_path ./pretrained/100-shot-obama.th
