python train_baseline.py \
    --device=cuda \
    --config=./configs/baseline_roberta.json

python test_baseline.py \
    --device=cuda \
    --config=./configs/baseline_roberta.json \
    --best_ckpt \
    --test_dataset_path=./datasets/e-SNLI/esnli_test.csv

python test_baseline.py \
    --device=cuda \
    --config=./configs/baseline_roberta.json \
    --best_ckpt \
    --test_dataset_path=./datasets/multinli_1.0/multinli_1.0_dev_matched.jsonl

python test_baseline.py \
    --device=cuda \
    --config=./configs/baseline_roberta.json \
    --best_ckpt \
    --test_dataset_path=./datasets/multinli_1.0/multinli_1.0_dev_mismatched.jsonl