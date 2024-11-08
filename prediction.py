from sklearn.utils import shuffle
import esm
# import joblib
from Bio import SeqIO
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from deepforest import CascadeForestClassifier
from src.utils import *
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"

'''
AMP/non-AMP prediction
specify AMP/non-AMP model path
'''
ftmodel_save_path = ''
clsmodel_save_path = ''


'''
prepare your data (in fasta format)
and labels (if any)
'''
sequences_file_path = ''
labels = []


seqs = []
for record in SeqIO.parse(sequences_file_path, "fasta"):
    seqs.append(str(record.seq))
    
seqs = np.array(seqs)
# generate sequence features
seq_embeddings, _, seq_ids = amp_feature_extraction(ftmodel_save_path, device, sequences_file_path)
# start prediction
cls_model = CascadeForestClassifier()
cls_model.load(clsmodel_save_path)

# binary classification
binary_pred = cls_model.predict(seq_embeddings)

# filter predicted amps
amp_index = np.where(binary_pred == 1)[0]
predicted_amps_seqs = seqs[amp_index]
predicted_amps_ids = np.array(seq_ids)[amp_index]
# predicted_amps_labels = seq_labels[amp_index]


# save AMP/non-AMP prediction results
result_df = pd.DataFrame(
    {
        'ID': predicted_amps_ids,
        'Sequence': predicted_amps_seqs,
    }
)

# AMP targets prediction
for target in ['Gram+', 'Gram-', 'Mammalian_Cell', 'Virus', 'Fungus', 'Cancer']:
    '''
    specify target model path
    both fine-tuned feature extractor and trained classifier
    '''
    target_ftmodel_save_path = ''
    target_clsmodel_save_path = ''
    
    
    # generate target features
    target_ids, target_embeddings = targets_feature_extraction(target_ftmodel_save_path, device, predicted_amps_ids, predicted_amps_seqs)
    
    # start prediction
    target_cls_model = CascadeForestClassifier()
    target_cls_model.load(target_clsmodel_save_path)
    
    # AMP target annotation
    target_pred = target_cls_model.predict(target_embeddings)
    result_df[target] = target_pred
    

# show all target predictions
print(result_df)
# result_df.to_csv('prediction_result.csv', index=False)

