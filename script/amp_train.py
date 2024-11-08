import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.utils import shuffle
import argparse
from sklearn.metrics import classification_report
import sys
# use sys.path.append() to add src path
sys.path.append('/data/yuqinze/project/HMD-AMP')
from src.utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AMP/non-AMP prediction training')
    
    parser.add_argument(
        "--training_emb",
        type=str,
        default=None,
        help='directory of the training embeddings'
    )

    parser.add_argument(
        "--clsmodel_save_path",
        type=str,
        default=None,
        help='The trained prediction model saving path, you should not create it in advance'
    )
    
    args = parser.parse_args()
    
    # data process
    train_data = np.load(f'{args.training_emb}/embeddings.npy')
    train_label = np.load(f'{args.training_emb}/labels.npy')
    train_label = train_label.astype("int")
    num_samples = train_data.shape[0]
    train_index = shuffle_index_raw(num_samples)
    train_data = train_data[train_index]
    train_label = train_label[train_index]

    model = CascadeForestClassifier(n_jobs=6, n_estimators=4, n_trees=1000, predictor='xgboost')
    model.fit(train_data, train_label)
    
    model.save(args.clsmodel_save_path)