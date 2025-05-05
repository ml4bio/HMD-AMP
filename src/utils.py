from sklearn.utils import shuffle
import esm
# import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from Bio import SeqIO
from src.Net import ampPredictor

class Pre_process():
    def __init__(self, filepath):
        self.data = []
        self.filepath = filepath
        self.gen = SeqIO.parse(open(self.filepath),'fasta')
    
    def gen_tuples(self):
        while True:
            try:
                protein = next(self.gen)
                yield (protein.id, str(protein.seq.upper()))
            except StopIteration:
                break
        
     # generate batches      
    def generate_data(self):
        for i, seq in enumerate(self.gen_tuples()):
                  # change places to remove BATCH-1
            self.data.append(seq)
    # clears memory by deleting processed proteins
    def annul_data(self):
        self.data = []   
    


def shuffle_index(num_samples):
    a = range(0, num_samples)
    a = shuffle(a)
    length = int(((num_samples + 1) / 5) * 4)
    train_index = a[:length]
    val_index = a[length:]
    return [train_index, val_index]


def shuffle_index_raw(num_samples):
    a = range(0, num_samples)
    a = shuffle(a, random_state=5)

    return a

    
    
class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'predictor': model.predictor.state_dict(),
            'outlinear': model.outlinear.state_dict()
        }, path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss
        
        
        
def convert_labels(labels):
    onehot_label = np.eye(2)[labels]
    
    return onehot_label



class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data= data[:,1]
        self.labels = labels
        self.id = data[:,0]  

    def __getitem__(self, index):    
        sequence, label, id = self.data[index], self.labels[index], self.id[index]
        # if torch.cuda.is_available():
        #     sequence = sequence.cuda()
        #     label = label.cuda()
        return sequence, label, id

    def __len__(self):
        return len(self.data)



def valsets_loss(val_loader, model, batch_converter, alphabet):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        # preds = torch.tensor([]).cuda()
        # labels = torch.tensor([]).cuda()
        count = 0
        for data, label, id in val_loader:
            esm_format_data = list(zip(id, data))
            batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

            batch_tokens = batch_tokens.cuda()
            label = convert_labels(label)
            label = torch.tensor(label).cuda()
            output, _ = model.module(batch_tokens, batch_lens)
            output2 = output.argmax(dim=1)
            # label = label.argmax(dim=1)
            if count == 0:
                preds = output2
                labels = label
                probs = output
                count = 1
            else:
                probs = torch.cat((probs, output), 0)
                preds = torch.cat((preds, output2), -1)
                labels = torch.cat((labels, label), 0)
            
            total += label.size(0)
            correct += (output == label).sum().item()
        
        # labels = labels.detach().cpu().numpy()
        # preds = preds.detach().cpu().numpy()
        # probs = probs.detach().cpu().numpy()
        
    return probs, preds, labels
    



def split_equal_distribution(fold_number, index_groups):
    val_index = index_groups[fold_number]
    train_index = np.concatenate(index_groups[:fold_number] + index_groups[fold_number+1:])
    return train_index, val_index




def target_fine_tuned_emb_extraction(data_loader, model, batch_converter, alphabet):
    with torch.no_grad():
        # embeddings = []
        # labels = []
        # ids = []
        # embeddings = np.array(embeddings)
        # labels = np.array(labels)
        # ids = np.array(ids)
        count = 0
        for data, label, id in data_loader:
            esm_format_data = list(zip(id, data))
            batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

            batch_tokens = batch_tokens.cuda()
            # label = convert_labels(label)
            _, emb = model(batch_tokens, batch_lens)
            
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
    
    return ids, embeddings, labels


def amp_feature_extraction(ftmodel_save_path, device, data_path):
    # extract representations of proteins from finetuned model
    # load ESM model
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    print('pretrain model downloded')
    batch_converter = alphabet.get_batch_converter()
    esm_model = nn.DataParallel(esm_model, device_ids=[0])
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    # load feature_extraction model
    model_test = ampPredictor(esm_model)
    checkpoint = torch.load(ftmodel_save_path)
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
    psudo_labels = np.full((data.shape[0],),0)
    train_dataset = MyDataset(data, psudo_labels)
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
                
    return embeddings, _, ids


def targets_feature_extraction(ftmodel_save_path, device, id, sequence):
    # load ESM model
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    print('pretrain model downloded')
    batch_converter = alphabet.get_batch_converter()
    esm_model = nn.DataParallel(esm_model, device_ids=[0])
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    # load feature_extraction model
    model_test = ampPredictor(esm_model)
    checkpoint = torch.load(ftmodel_save_path)
    # Load the parameters into the corresponding parts of the model
    model_test.predictor.load_state_dict(checkpoint['predictor'])
    model_test.outlinear.load_state_dict(checkpoint['outlinear'])
    model_test = nn.DataParallel(model_test, device_ids=[0])
    
    model_test = model_test.module.to(device)
    model_test.eval()
    
    esm_format_data = list(zip(id, sequence))
    batch_labels, batch_strs, batch_tokens = batch_converter(esm_format_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1).cuda()

    batch_tokens = batch_tokens.cuda()
    # label = convert_labels(label)
    _, emb = model_test(batch_tokens, batch_lens)
            
    embedding = emb.cpu().detach().numpy()
    # if count == 0: 
    #     embeddings = emb
    #     # labels = label
    #     ids = id
    #     count = count+1
    # else:
    #     embeddings = np.concatenate((embeddings, emb), axis=0)
    #     # labels = np.concatenate((labels, label), axis=0)
    #     ids = np.concatenate((ids, np.array(id)), axis=0)
        
    return id, embedding
