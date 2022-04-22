import torch
from torch import nn
import torch.optim as optim


def tensorized(data):
    return torch.tensor(data)
    
def post_process(input_x):
    Batch, seq, hidden = input_x.size()
    indices = torch.tensor([seq-1])
    temp = torch.index_select(input_x, 1, indices)
    # print("temp.size(): ", temp.size())
    # print(input_x[240,49,:])
    # print(temp[240,:])
    return temp.squeeze(1)




class Model (object):
    
    def __init__(self):
        self.h0 = torch.randn(2, 3, 20)
        self.c0 = torch.randn(2, 3, 20)
        input_feat_dim = 4
        hidden_feat_dim = 32
        num_layers = 2
        self.model = LSTM(input_feat_dim, hidden_feat_dim, num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fnc = nn.CrossEntropyLoss()


    def train(self, x_train, y_train):
        self.model.train()
        x_train = tensorized(x_train).to(torch.float32) # [B, len, emd_size]
        y_train = tensorized(y_train).to(torch.float32)
        # print(x_train.size())
        for epoch in range(1, 6):
            epoch_loss = 0
            for i in range(1, 2):
                self.model.zero_grad()
                # print("x_train.shape: ", x_train.shape)
                output = self.model(x_train)
                # print("pred.shape:", pred.shape)
                # output = post_process(output)
                # exit()
                # print("output.shape: ", output.shape)
                # print("y_train.shape: ", y_train.shape)
                
                loss = self.loss_fnc(output,y_train)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
                
            print(f" Epoch {epoch} | loss: {epoch_loss}")




    def predict(self, x_test):
        self.model.eval()
        x_test = tensorized(x_test).to(torch.float32)

        # print("x_test.shape: ", x_test.shape)
        output = self.model(x_test)
        # print("output.shape: ", output.shape)
        print("output: ", output)
        index = torch.argmax(output, dim=1)
        print("index: ", index)




class LSTM (nn.Module):
    
    def __init__(self, input_feat_dim, hidden_feat_dim, num_layers):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(input_feat_dim, hidden_feat_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_feat_dim, 3)
        self.num_layers = num_layers
        self.hidden_feat_dim = hidden_feat_dim
        self.hidden_dim = hidden_feat_dim

        
    
    def forward(self, input_x):
        # h0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_dim).requires_grad_()
        # c0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_dim).requires_grad_()
        rnn_out, (h_n, c_n) = self.model(input_x)
        # print("rnn_out.shape: ", rnn_out.shape)
        rnn_out = post_process(rnn_out)
        # print("rnn_out.shape: ", rnn_out.shape)
        prediction = self.linear(rnn_out)
        # print("prediction.shape: ", prediction.shape)
        # print(rnn_out.size())
        
        return prediction










