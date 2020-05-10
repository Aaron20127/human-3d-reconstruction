cd src
# wh_weight train test
python main.py --exp_id weight_1 --wh_weight 1 --kp2d_weight 1\
                --batch_size_coco 15 \
               --batch_size_lsp  15 \
               --num_epochs 1 \
               --num_iters 1 \
               --save_intervals 1 \
               --gpus 0 \
               --num_workers 2


python main.py  --exp_id weight_2 --wh_weight 0.1 --kp2d_weight 1\
               --batch_size_coco 15 \
               --batch_size_lsp  15 \
               --num_epochs 1 \
               --num_iters 1 \
               --save_intervals 1 \
               --gpus 0 \
               --num_workers 2

python main.py  --exp_id weight_2 --wh_weight 1 --kp2d_weight 0.1\
               --batch_size_coco 15 \
               --batch_size_lsp  15 \
               --num_epochs 1 \
               --num_iters 1 \
               --save_intervals 1 \
               --gpus 0 \
               --num_workers 2 \

python main.py  --exp_id weight_2 --wh_weight 0.1 --kp2d_weight 0.1\
               --batch_size_coco 15 \
               --batch_size_lsp  15 \
               --num_epochs 1 \
               --num_iters 1 \
               --save_intervals 1 \
               --gpus 0 \
               --num_workers 2 \

cd ..
