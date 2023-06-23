python train.py \
    --device=cuda \
    --config=./configs/default_ed.json

python test.py \
     --device=cuda \
     --config=./configs/default_ed.json \
     --best_ckpt
