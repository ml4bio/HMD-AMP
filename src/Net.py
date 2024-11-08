import torch
import torch.nn as nn


class ampPredictor(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.encoder = esm_model
        self.predictor = nn.Sequential(
            nn.Linear(1280,640),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(640,320),
            nn.ReLU(),
            nn.Dropout(0.2),
            
        )
        self.outlinear = nn.Linear(320, 2)
    def forward(self, seq, batch_len):
        res = self.encoder(seq, repr_layers=[33], return_contacts=False)
        rep = res['representations'][33]
        # rep = rep.mean(1)
        reps = []
        for i, tokens_len in enumerate(batch_len):
            reps.append(rep[i, 1 : tokens_len - 1].mean(0))
        # rep= torch.tensor([item.cpu().detach().numpy() for item in reps]).cuda()
        rep= torch.tensor([item.cpu().detach().numpy() for item in reps]).cuda()
        # rep = torch.Tensor(reps)
        emb = self.predictor(rep)
        y = self.outlinear(emb)
        
        return y, emb