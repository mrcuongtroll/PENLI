python train_rl.py \
     --device=cuda \
     --config=./configs/default_ed.json \
     --seed=123 \
     --best_ckpt \
     --use_finetuned_critic

python test.py \
     --device=cuda \
     --config=./configs/default_ed.json \
     --best_ckpt \
     --use_rl_ckpt
