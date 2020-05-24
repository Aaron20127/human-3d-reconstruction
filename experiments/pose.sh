cd src

## kp3d with big pose
python main.py --task pose --subtask pose_70 \
    --hm_weight 1 --wh_weight 1 --cd_weight 1 --pose_weight 50 --shape_weight 20 --kp2d_weight 2 --kp3d_weight 2\
    --batch_size_coco 9 --batch_size_lsp 2 --batch_size_hum36m 9 \
    --save_iter_interval 4000 --save_epoch_interval 1 \
    --val_iter_interval 4000 --val_epoch_interval 1 --val_batch_size_coco 1\
    --num_iters -1 --log_iters 20 --num_epochs 2 \
    --gpus 0 \
    --num_workers 4 \
    --min_vis_kps 6 --load_min_vis_kps 6\
    --lr 1.25e-5 \
    --lr_scheduler_factor 0.978 \
    --lr_scheduler_patience 200  \
    --lr_scheduler_threshold 0.001 \
    --eval_average_precision \
    --hide_data_time
cd ..
