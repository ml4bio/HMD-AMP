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





def split_5folds(args, fold_number):
    '''
    target: which functional activity to predict
    data_path: path of sequence fasta file
    label_path: path of label file (numpy array)
    fold_number: specify the fold
    '''
    
    
    train_sampler = None
    idx = args.epochs // 30
    betas = [0, 0.999]
    
    # prepare data
    data_prep = Pre_process(args.training_data)
    data_prep.generate_data()
    data = data_prep.data
    data = np.array(data)
    
    label = np.load(args.label_path)

    index_list = {0:[], 1:[]}
    # test_split = 0.2
    for i in range(2):
        index_list[i] = np.where(np.array(label) == i)[0]
        
    index_list[0] = shuffle(index_list[0], random_state=5)
    index_list[1] = shuffle(index_list[1], random_state=5)
    
    split_index_neg = np.array_split(index_list[0], 5)
    split_index_pos = np.array_split(index_list[1], 5)
    

    # split data into 5 fold
   
    train_pos_index, val_pos_index = split_equal_distribution(fold_number, split_index_pos)
    train_neg_index, val_neg_index = split_equal_distribution(fold_number, split_index_neg)

    train_index = shuffle(np.concatenate((train_neg_index, train_pos_index)), random_state=5)
    val_index = shuffle(np.concatenate((val_neg_index, val_pos_index)), random_state=5)

    X_train = data[train_index]
    y_train = label[train_index]

    X_val = data[val_index]
    y_val = label[val_index]

    cls_num_list = [np.sum(y_train == 0), np.sum(y_train == 1)]
    print(cls_num_list)
    
    effective_num = 1.0 - np.power(betas[idx], cls_num_list)
    per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
    
    criterion_ldam = LDAMLoss(cls_num_list=cls_num_list, max_m=0.3, s=150, weight=per_cls_weights)
    
    print(f'data splited, now start fine tune the model on {args.target}')

    fine_tuning_target(args, X_train, y_train, X_val, y_val, criterion_ldam, fold_number)


def fine_tuning_target(args, train_data, train_label, val_data, val_label, loss_function, fold_number):
    early_stopping = EarlyStopping(patience=5)
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", 'esm2_t33_650M_UR50D')
    esm_model = nn.DataParallel(esm_model, device_ids=[0, 1])
    esm_model = esm_model.to(device)
    num_classes = 2
    model = ampPredictor(esm_model)
    batch_converter = alphabet.get_batch_converter()
    model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001, eps=1e-8)
    batch_size = args.batch_size
    epochs = args.epochs
    
    train_dataset = MyDataset(train_data, train_label)
    val_dataset = MyDataset(val_data, val_label)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    train_epochs_loss = []
    
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []
        
        for step, (data, label, id) in enumerate(train_loader):
            esm_format_data = list(zip(id, data))
            batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

            batch_tokens = batch_tokens.cuda()
            # label = convert_labels(label)
            label = torch.as_tensor(label).cuda()
            optimizer.zero_grad()
            # print(id)
            output, _ = model.module(batch_tokens, batch_lens)
            # output = output.squeeze(1)
            # loss = criterion(output, label.float())
            loss = loss_function(output, label.float())
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
        
        
        with torch.no_grad():
            model.eval()
            count = 0
            for data, label, id in val_loader:
                esm_format_data = list(zip(id, data))
                batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

                batch_tokens = batch_tokens.cuda()
                # label = convert_labels(label)
                label = torch.tensor(label).cuda()
                output, _ = model.module(batch_tokens, batch_lens)
                output2 = output.argmax(dim=1)
                # label = label.argmax(dim=1)
                if count == 0:
                    probs = output
                    preds = output2
                    labels = label
                    count = 1
                else:
                    preds = torch.cat((preds, output2), -1)
                    labels = torch.cat((labels, label), 0)
                    probs = torch.cat((probs, output), 0)

        valid_loss = loss_function(probs, labels.float())
            
        early_stopping(valid_loss, model, args.ftmodel_save_path+f'/{args.target}_fold{fold_number}')
            # 
        if early_stopping.early_stop:
            print(f'Early stop at epoch {epoch}, model is saved.')
            break
        elif epoch==49:
            torch.save({
                'predictor': model.predictor.state_dict(),
                'outlinear': model.outlinear.state_dict()
            }, args.ftmodel_save_path+f'/{args.target}_fold{fold_number}'+'/'+'model_checkpoint.pth')
                      
    del model
    del esm_model
    

def target_feature_extraction(args, fold_number):
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
    checkpoint = torch.load(args.ftmodel_save_path+f'/{args.target}_fold{fold_number}'+'/'+'model_checkpoint_fold.pth')
    # Load the parameters into the corresponding parts of the model
    model_test.predictor.load_state_dict(checkpoint['predictor'])
    model_test.outlinear.load_state_dict(checkpoint['outlinear'])
    
    model_test = nn.DataParallel(model_test, device_ids=[0])
    model_test = model_test.module.to(device)
    model_test.eval()
    
    data_prep = Pre_process(args.data_path)
    data_prep.generate_data()
    data = data_prep.data
    data = np.array(data)
    
    
    label = np.load(args.label_path)

    index_list = {0:[], 1:[]}
    # test_split = 0.2
    for i in range(2):
        index_list[i] = np.where(np.array(label) == i)[0]
    
    # random_state should be the same with split_5folds function
    index_list[0] = shuffle(index_list[0], random_state=5)
    index_list[1] = shuffle(index_list[1], random_state=5)
    
    split_index_neg = np.array_split(index_list[0], 5)
    split_index_pos = np.array_split(index_list[1], 5)
    

    # split data into 5 fold
   
    train_pos_index, val_pos_index = split_equal_distribution(fold_number, split_index_pos)
    train_neg_index, val_neg_index = split_equal_distribution(fold_number, split_index_neg)

    val_pos_index = split_index_pos[0]
    val_neg_index = split_index_neg[0]

    train_index = shuffle(np.concatenate((train_neg_index, train_pos_index)), random_state=5)
    val_index = shuffle(np.concatenate((val_neg_index, val_pos_index)), random_state=5)
    
    
    
    train_dataset = MyDataset(data[train_index], label[train_index])
    val_dataset = MyDataset(data[val_index], label[val_index])

    batch_size = args.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    train_ids, train_embeddings, train_labels = target_fine_tuned_emb_extraction(train_loader, model_test, batch_converter, alphabet)
    val_ids, val_embeddings, val_labels = target_fine_tuned_emb_extraction(val_loader, model_test, batch_converter, alphabet)
    
    return train_embeddings, train_labels, train_ids, val_embeddings, val_labels, val_ids




if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = ['Gram+', 'Gram-', 'Mammalian_Cell', 'Virus', 'Fungus', 
            'Cancer']
    order = [0, 1, 2, 3, 4, 5]
    vec = dict(zip(order, name))


    parser = argparse.ArgumentParser(description='AMP/non-AMP prediction feature extraction')
    
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help='specify the functional activity to predict: Gram+, Gram-, Mammalian_Cell, Virus, Fungus, or Cancer'
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help='path of the training data fasta file'
    )
    parser.add_argument(
        "--label_path",
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
    
    # if no available fintuned model
    if not os.path.exists(args.ftmodel_save_path+'/'+'model_checkpoint.pth'):
        # 5-fold fine-tuning, will produce 5 models corresponding to 5 folds.
        # remove for loop if only need to train once
        for i in range(0, 4):
            split_5folds(args, i)
    
    # extract embeddings of 5 folds
    # remove for loop and specify the fold number if only need to extract once
    for i in range(0, 4):
        train_embeddings, train_labels,_ , val_embeddings, val_labels, _ = target_feature_extraction(args, i)
        np.save(f'{args.emb_path}/{args.target}_fold{i}/train_embeddings.npy', train_embeddings)
        np.save(f'{args.emb_path}/{args.target}_fold{i}/train_labels.npy', train_labels)
        np.save(f'{args.emb_path}/{args.target}_fold{i}/val_embeddings.npy', val_embeddings)
        np.save(f'{args.emb_path}/{args.target}_fold{i}/val_labels.npy', val_labels)
        
    
    
    
    
    