# HMD-AMP


This repository contains code for the paper ***HMD-AMP***.





## Local Environment Setup

First, clone the repository and create an environment with conda.<br>

```bash
git clone https://github.com/ml4bio/HMD-AMP.git
conda create -n HMD-AMP python=3.8 -y
conda activate HMD-AMP
```

Pytorch installation is different for different Operating systems. For Linux, please use the following commands.<br>

```bash
pip install torch torchvision torchaudio
```

Then, install the following packages.

```bash
pip install deep-forest
pip install scikit-learn==1.3.0
pip install pandas==1.2.0
pip install biopython==1.83
pip install fair-esm==2.0.0
pip install numpy==1.19.5
```

<!--
## Data
The training data can be obtained from [Zenodo](https://doi.org/10.5281/zenodo.15583284).
-->

## Training 
You can find the training script in `script`.

The training data can be obtained from [Zenodo](https://doi.org/10.5281/zenodo.15583284).

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
target_ftmodel_save_path = f'model/{target}/ft_parts.pth'
# specify the path of Trained classifier folder
target_clsmodel_save_path = f'model/{target}/clsmodel'
```

At last, run:
```
python prediction.py
```
you could get the prediction result of the sequences.
