import torch
from torch import nn
import torch.optim as optim


def tensorized(data):
    return torch.tensor(data)
    



class LSTM(nn.Module):
    
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.h0 = torch.randn(2, 3, 20)
        self.c0 = torch.randn(2, 3, 20)
        input_feat_dim=1
        hidden_feat_dim=32
        hidden_layer_dim=1
        self.model = nn.LSTM(input_feat_dim, hidden_feat_dim, batch_first=True) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)   
        self.loss_fnc = nn.CrossEntropyLoss()
    
    
    def forward(self, input_data):
        rnn_out, (h_n, c_n) = self.model(input_data)
        print(rnn_out.size())
        return rnn_out


    def train(self, x_train, y_train):
        self.model.train()
        x_train = tensorized(x_train).to(torch.float32) # [B, len, emd_size]
        y_train = tensorized(y_train).to(torch.float32)
        # print(x_train.size())
        
        for epoch in range(1, 2):
            epoch_loss = 0
            for i in range(1, 2):
                self.model.zero_grad()
                
                output = self.forward(x_train)
                
                loss = self.loss_fnc(output,y_train)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
                
            print(f" Epoch {epoch} | loss: {epoch_loss}")