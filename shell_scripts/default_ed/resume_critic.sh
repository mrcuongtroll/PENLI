python train.py \
  --device=cuda \
  --config=./configs/default_ed.json \
  --finetune_critic \
  --resume

python test.py \
     --device=cuda \
     --config=./configs/default_ed.json \
     --best_ckpt \
     --use_explanation \
     --use_critic_ckpt
