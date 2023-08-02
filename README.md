# PENLI
Pattern-Exploiting Natural Language Inference.

(My graduation thesis)
___________________________________________
## Requirements
To install all frameworks and libraries used in this project:
```
pip install -r requirements.txt
```
___________________________________________
## Datasets
This project uses two Natural Language Inference Datasets: [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI), which 
is an extension of the [SNLI](https://nlp.stanford.edu/projects/snli/) dataset,
and [MNLI](https://cims.nyu.edu/~sbowman/multinli/).
These two datasets should be stored in directories similar to this:  
```
PromptCLED
│   README.md
│   requirements.txt    
│   test.py
│   ...
└───datasets
│   └───e-SNLI
│   │   │esnli_dev.csv
│   │   │esnli_test.csv
│   │   │esnli_train_1.csv
│   │   │esnli_train_2.csv
│   │   
│   └───multinli_1.0
│   │   │multinli_1.0_dev_matched.jsonl
│   │   │multinli_1.0_dev_matched.txt
│   │   │multinli_1.0_dev_mismatched.jsonl
│   │   │multinli_1.0_dev_mismatched.txt   
│   │   │multinli_1.0_train.jsonl   
│   │   │multinli_1.0_train.txt  
│   
└───...
    │   ...
    │   ...
...
```  
These two datasets are publicly available and can be downloaded from the provided links.
Please download and place them inside the folders following the structure above.

___________________________________________
## Training
To train a model, run `train.py`. For example:
```
python train.py --config=./configs/default_ed.json --device=cuda
```
For more information about the arguments of this python script:
```
python train.py --help
```
___________________________________________
## Evaluating
To evaluate a trained model, run `test.py`. For example:
```
python test.py --config=./configs/default_ed.json --device=cuda --best_ckpt
```
For more information about the arguments of this python script:
```
python test.py --help
```
___________________________________________
## Reinforcement Learning
To further fine-tune a trained model using RL, run `train_rl.py`. For example:
```
python train.py --config=./configs/default_ed.json --device=cuda
```
The supervised fine-tuned checkpoint of the corresponding config must exist before RL training.
For more information about the arguments of this python script:
```
python train.py --help
```
