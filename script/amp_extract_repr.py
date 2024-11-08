from sklearn.utils import shuffle
import esm
# import joblib
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import sys
# use sys.path.append() to add src path
sys.path.append('/data/yuqinze/project/HMD-AMP')
from src.loss import *
from src.Net import *
from src.utils import *



def data_process(data_path, labels_path):
    # generate dataset for training
    labels_array = np.load(labels_path)
    data_prep = Pre_process(data_path)
    data_prep.generate_data()
    data = data_prep.data
    data = np.array(data)
    num_samples = data.shape[0]
    print(f'The amount of training data: {num_samples}')
    train_index, val_index = shuffle_index(num_samples)
    train_data = data[train_index]
    train_label = labels_array[train_index]
    val_data = data[val_index]
    val_label = labels_array[val_index]
    
    train_dataset = MyDataset(train_data, train_label)
    val_dataset = MyDataset(val_data, val_label)
    
    return train_dataset, val_dataset
    
    
def fine_tuning(args, train_dataset, val_dataset):
    early_stopping = EarlyStopping(patience=10)
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", 'esm2_t33_650M_UR50D')
    esm_model = nn.DataParallel(esm_model, device_ids=[0, 1])
    esm_model = esm_model.to(device)
    num_classes = 2
    model = ampPredictor(esm_model)
    batch_converter = alphabet.get_batch_converter()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
        
    
    # specify optimizer, loss, and hyperparameters        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001, eps=1e-8)
    batch_size = args.batch_size
    epochs = args.epochs
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    # record the performance
    train_epochs_loss = []
    best_test_acc = 0
    best_epoch = 0
    
    # start training 
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
    
    
        for step, (data, label, id) in enumerate(train_loader):
            esm_format_data = list(zip(id, data))
            batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

            batch_tokens = batch_tokens.cuda()
            label = convert_labels(label)
            label = torch.tensor(label).cuda()
            optimizer.zero_grad()
            output, _ = model.module(batch_tokens, batch_lens)
            # output = output.squeeze(1)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            
            # train_epoch_acc.append(get_acc(output, label.float()))
            train_epoch_loss.append(loss.item())

            if step % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, step * len(data), len(train_loader.dataset),
          100. * step / len(train_loader), loss.item()))
        
        train_epochs_loss.append(np.average(train_epoch_loss))
        print(train_epochs_loss)
        
        # on validation set
        val_probs, _, val_labels = valsets_loss(val_loader, model, batch_converter, alphabet)
        val_loss = criterion(val_probs, val_labels.float())
        
        early_stopping(val_loss, model, args.ftmodel_save_path)
        if early_stopping.early_stop:
            
            print(f'Early stop at epoch {epoch}, model is saved.')
            break
        elif epoch==49:
            torch.save({
                'predictor': model.predictor.state_dict(),
                'outlinear': model.outlinear.state_dict()
            }, args.ftmodel_save_path+'/'+'model_checkpoint.pth')
            
    del model
    del esm_model
        
        

def feature_extraction(ftmodel_save_path, data_path, labels_path):
    # extract representations of proteins from finetuned model
    # load ESM model
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", 'esm2_t33_650M_UR50D')
    print('pretrain model downloded')
    batch_converter = alphabet.get_batch_converter()
    esm_model = nn.DataParallel(esm_model, device_ids=[0])
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    # load feature_extraction model
    model_test = ampPredictor(esm_model)
    
    checkpoint = torch.load(ftmodel_save_path+'/'+'model_checkpoint.pth')
    # Load the parameters into the corresponding parts of the model
    model_test.predictor.load_state_dict(checkpoint['predictor'])
    model_test.outlinear.load_state_dict(checkpoint['outlinear'])
    
    model_test = nn.DataParallel(model_test, device_ids=[0])
    model_test = model_test.module.to(device)
    model_test.eval()
    
    data_prep = Pre_process(data_path)
    data_prep.generate_data()
    data = data_prep.data
    data = np.array(data)
    labels_array = np.load(labels_path)
    
    train_dataset = MyDataset(data, labels_array)
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        count = 0
        for data, label, id in train_loader:
            esm_format_data = list(zip(id, data))
            batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

            batch_tokens = batch_tokens.cuda()
            # label = convert_labels(label)
            _, emb = model_test(batch_tokens, batch_lens)
            
            emb = emb.cpu().detach().numpy()
            if count == 0: 
                embeddings = emb
                labels = label
                ids = id
                count = count+1
            else:
                embeddings = np.concatenate((embeddings, emb), axis=0)
                labels = np.concatenate((labels, label), axis=0)
                ids = np.concatenate((ids, np.array(id)), axis=0)
                
    return embeddings, labels, ids



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    parser = argparse.ArgumentParser(description='AMP/non-AMP prediction feature extraction')
    
    parser.add_argument(
        "--training_data",
        type=str,
        default=None,
        help='path of the training data fasta file'
    )
    parser.add_argument(
        "--training_label",
        type=str,
        default=None,
        help='path of the training labels, in .npy format'
    )
    parser.add_argument(
        "--ftmodel_save_path",
        type=str,
        default=None,
        help='fine-tuned model saving directory'
    )
    parser.add_argument(
        "--emb_path",
        type=str,
        default=None,
        help='directory to save the embeddings and labels'
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="upper epoch limit")
    parser.add_argument(
        "--batch_size", type=int, default=32, metavar="N", help="batch size"
    )
    
    
    args = parser.parse_args()
    
    # if no available fine-tuned model
    if not os.path.exists(args.ftmodel_save_path+'/'+'model_checkpoint.pth'):
        train_dataset, val_dataset = data_process(args.training_data, args.training_label)
        fine_tuning(args, train_dataset, val_dataset)
    embeddings, labels, _ = feature_extraction(args.ftmodel_save_path, args.training_data, args.training_label)
    np.save(f'{args.emb_path}/embeddings.npy', embeddings)
    np.save(f'{args.emb_path}/labels.npy', labels)
    print('embeddings extracted and saved.')