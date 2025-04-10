import numpy as np
from deepforest import CascadeForestClassifier
from sklearn.utils import shuffle
import argparse
from sklearn.metrics import classification_report
import sys
# use sys.path.append() to add src path
sys.path.append('../')
from src.utils import *



if __name__ == '__main__':    
    name = ['Gram+', 'Gram-', 'Mammalian_Cell', 'Virus', 'Fungus', 
            'Cancer']
    order = [0, 1, 2, 3, 4, 5]
    vec = dict(zip(order, name))

    
    
    parser = argparse.ArgumentParser(description='AMP/non-AMP prediction training')
    
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help='specify the functional activity to predict: Gram+, Gram-, Mammalian_Cell, Virus, Fungus, or Cancer'
    )
    
    parser.add_argument(
        "--emb",
        type=str,
        default=None,
        help='directory of the embeddings'
    )

    parser.add_argument(
        "--clsmodel_save_path",
        type=str,
        default=None,
        help='The trained prediction model saving path, you should not create it in advance'
    )
    
    args = parser.parse_args()
    
    # data process
    # specify the fold number to reach the data
    fold_number = 0
    # for i in range (0, 4)
    train_data = np.load(f'{args.emb_path}/{args.target}_fold{fold_number}/train_embeddings.npy')
    train_label = np.load(f'{args.emb_path}/{args.target}_fold{fold_number}/train_labels.npy')
    
    val_data = np.load(f'{args.emb_path}/{args.target}_fold{fold_number}/val_embeddings.npy')
    val_label = np.load(f'{args.emb_path}/{args.target}_fold{fold_number}/val_labels.npy')
    
    train_label = train_label.astype("int")
    val_label = val_label.astype("int")
    num_samples = train_data.shape[0]
    
    train_index = shuffle_index_raw(num_samples)
    train_data = train_data[train_index]
    train_label = train_label[train_index]

    model = CascadeForestClassifier(n_jobs=6, n_estimators=4, n_trees=1000, predictor='xgboost')
    model.fit(train_data, train_label)
    
    # y_prob = model.predict_proba(val_data)
    model.save(f'{args.clsmodel_save_path}/{args.target}_fold{fold_number}/')
