# COMP5212 Visual Emtailment
Part of the code is modified from Github repo [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task).

## Video Introduction

[![GivBERT](https://i.imgur.com/hnx74Ms.png)](https://youtu.be/dY5pY9iRLMg "GivBERT") 

## Setup
1. Download the repository from Github
```
git clone git@github.com:NoOneUST/COMP5212Project.git
cd  COMP5212Project
```
2. Install the requirements
```
pip install -r requirements.txt
```
3. Install pytorch, please check your CUDA version
```
conda install pytorch==1.4 torchvision cudatoolkit=10.1 -c pytorch
```

## Data Setup
To setup the data, you can either download the data provided by [vilbert-multi-task](https://github.com/facebookresearch/vilbert-multi-task/tree/master/data), or download from Google Drive which is a pruned-version especially for this project.
```text
TBC
```

## Model Setup
To get move on, you need to download the pre-trained VilBERT models for [12-in-1: Multi-Task Vision and Language Representation Learning](https://arxiv.org/abs/1912.02315). Please put the models under **model** folder. The download links are listed below:
### VilBERT
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin)
### VilBERT-MT 
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin)

## Working directory
### GivBERT
cd ./GivBERT
### VilBERT
cd ./

## Command lines for experiments
```
python main.py --bert_model bert-base-uncased --from_pretrained model/<model_name> --config_file config/bert_base_6layer_6conect.json --lr_scheduler 'warmup_linear' --train_iter_gap 4 --save_name <finetune_from_multi_task_model>
```
