# HMD-AMP


This repository contains code for the paper ***HMD-AMP***.





## Local Environment Setup

First, clone the repository and create an environment with conda.<br>

```bash
git clone https://github.com/Yu979/HMD-AMP.git
cd ./HMD-AMP
conda env create -f ./environment.yaml
```
<!--
## Data
The training data and test data can be obtained from our OSF project.
-->

## Training 
You can find the training script in `scripts`

## Prediction
`prediction.py` contains the script for AMP and its target groups prediction.

### Trained models
Besides training the model by yourself, we also provide the fine-tuned protein language model and trained classifiers for direct usage. 
#### [AMP/non-AMP prediction task](https://drive.google.com/file/d/1Z4IeD0rUfBtN4OwSh7S-2fJUCbk07qiA/view?usp=sharing)
Fine-tuned protein language model: `ft_parts.pth`

Trained classifier: `clsmodel/`

First, in `prediction.py`, assign `sequences_file_path` with your path of *FASTA file*, then download and decompress the above model files: assign `ftmodel_save_path` with your path of *Fine-tuned protein language model*
and `clsmodel_save_path` with your path of *Trained classifier* folder.


#### [AMP target groups prediction task](https://drive.google.com/file/d/199S59bh9KO9IPTmzOYOhd4t1NHN_zdcg/view?usp=sharing)

Fine-tuned protein language model: `ft_parts.pth`

Trained classifier: `clsmodel/`



First, download and decompress the above model files and it is suggested to organize them in the following format of directory:
```
Model
├── Gram+
│   ├── Fine-tuned model
│   └── Trained classifier folder
│       └── ...
├── Gram-
│   ├── Fine-tuned model
│   └── Trained classifier folder
│       └── ...
├── Mammalian_Cell
│   ├── Fine-tuned model
│   └── Trained classifier folder
│       └── ...
├── Virus
│   ├── Fine-tuned model
│   └── Trained classifier folder
│       └── ...
├── Fungus
│   ├── Fine-tuned model
│   └── Trained classifier folder
│       └── ...
└── Cancer
    ├── Fine-tuned model
    └── Trained classifier folder
        └── ...  
```
and you then could modify the corresponding path in `prediction.py`:
```python
# specify the path of Fine-tuned model
target_ftmodel_save_path = f'model/{target}/model_checkpoint.pth'
# specify the path of Trained classifier folder
target_clsmodel_save_path = f'model/{target}/clsmodel'
```

At last, run:
```
python prediction.py
```
you could get the prediction result of the sequences.
