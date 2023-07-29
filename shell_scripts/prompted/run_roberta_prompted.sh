python train.py \
   --device=cuda \
   --config=./configs/default_roberta.json

python test.py \
   --device=cuda \
   --config=./configs/default_roberta.json \
   --best_ckpt \
   --test_dataset_path=./datasets/e-SNLI/esnli_test.csv

python test.py \
   --device=cuda \
   --config=./configs/default_roberta.json \
   --best_ckpt \
   --test_dataset_path=./datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl

python test.py \
   --device=cuda \
   --config=./configs/default_roberta.json \
   --best_ckpt \
   --test_dataset_path=./datasets/multinli_1.0/multinli_1.0_dev_mismatched.jsonl