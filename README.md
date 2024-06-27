# OOD Knowledge Distillation

## Setup
* Install the requirements from `requirements.txt`
* Setup `.env`file for your project as in `.env.example`

## Running experiments
* Training and distillation experiments can be run through `src/main.py`, passing the experiment config from `configs`direct`o`ry, e. g.:
```bash
python src/main.py --cfg_path configs/train/... 
```
* To override the config arguments from commandline, use hydra syntax:
```bash
python src/main.py --cfg_path configs/train/... seed=52 maxepochs=10 
```