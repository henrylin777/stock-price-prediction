import torch
from torch import nn
import torch.optim as optim
import pandas as pd
from copy import deepcopy

from utils import trend2action, calculate_profit
import math





def tensorized(data):
    return torch.tensor(data)
    
def post_process(input_x):
    Batch, seq, hidden = input_x.size()
    # indices = torch.tensor([seq-1]).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    indices = torch.tensor([seq-1]).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    temp = torch.index_select(input_x, 1, indices)
    # print("temp.size(): ", temp.size())
    # print(input_x[240,49,:])
    # print(temp[240,:])
    return temp.squeeze(1)




class Model (object):
    
    def __init__(self, feat_dim, prev_best_profit):
        input_feat_dim = feat_dim
        hidden_feat_dim = 32
        num_layers = 2
        self.model = LSTM(input_feat_dim, hidden_feat_dim, num_layers)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # self.loss_fnc = nn.MSELoss()
        self.loss_fnc = nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device: ", self.device)
        if torch.cuda.is_available():
            self.model.cuda()
        self.best_profit = prev_best_profit
        self.model_name = "lstm_model"


    def print_structure(self, input_vector):
        with torch.no_grad():
            import pytorch_model_summary as pms
            result = pms.summary(self.model, input_vector, print_summary=True)
            # print(result) 

        return




    def train(self, x_train, y_train, x_test, y_test, raw_test_data):
        self.model.train()
        x_train = tensorized(x_train).to(device=self.device, dtype=torch.float32) # [B, len, emd_size]
        y_train = tensorized(y_train).to(device=self.device, dtype=torch.float32)
        x_test = tensorized(x_test).to(device=self.device, dtype=torch.float32)
        y_test = tensorized(y_test).to(device=self.device, dtype=torch.float32)
        # print(x_train.size())
        print(x_train.shape)
        self.print_structure(x_train[-1,:,:].unsqueeze(0))

        for epoch in range(1, 71):
            epoch_loss = 0
            assert x_train.size(0) == y_train.size(0)
            batch_size = 32

            # print("x_train.shape: ", x_train.shape)
            # print("y_train.shape: ", y_train.shape)
            # exit()
            for idx in range(0,x_train .size(0), batch_size):
                accu_list = []
                batch_x = x_train[ idx:idx+batch_size, :, :]
                batch_y = y_train[ idx:idx+batch_size,:]
                self.model.zero_grad()

                output = self.model(batch_x)
                
                loss = self.loss_fnc(output,batch_y)
                accu_rate = self.compute_accuracy(output, batch_y)

                accu_list.append(accu_rate)
                epoch_loss += loss
                loss.backward()
                self.optimizer.step()
            
            if epoch % 1 == 0:
                print(f" Epoch {epoch} | accu: {math.floor(sum(accu_list)/len(accu_list)*10000)/10000} | loss: {math.floor(epoch_loss.item()*10000)/10000} ")

            if epoch % 10 == 0:
                self.validate(x_test, y_test, raw_test_data)

        return self.best_profit


    def validate(self, x_val, y_val, df):
        with torch.no_grad():
            assert x_val.size(0) == y_val.size(0)
            batch_size = 32
            loss = 0
            for idx in range(0,x_val .size(0), batch_size):
                accu_list = []
                batch_x = x_val[ idx:idx+batch_size, :, :]
                batch_y = y_val[ idx:idx+batch_size,:]
                output = self.model(batch_x)
                loss += self.loss_fnc(output,batch_y)
                accu_rate = self.compute_accuracy(output,batch_y)
                accu_list.append(accu_rate)
        
            profit = self.calculate_profit(output, df)

        print(f"Validation | accu: {math.floor(sum(accu_list)/len(accu_list)*10000)/10000} | profit: {round(profit,4)} | best profit: {round(self.best_profit,4)}")
        if profit > self.best_profit:
            print("Save model...")
            self.save_model(deepcopy(self.model))
            self.best_profit = profit


    def save_model(self, model):
        torch.save(model.state_dict(), self.model_name)
        return

    def load_model(self, name=None):
        if name == None:
            name = self.model_name
        
        ckpt = torch.load(name, map_location=self.device)
        self.model.load_state_dict(ckpt)



    def compute_accuracy(self, predictions, labels) -> float:
        
        with torch.no_grad():
            batch_size = predictions.size(0)
            _, pred = torch.topk(predictions,1, dim=1)
            _, labels = torch.topk(labels,1, dim=1)
            # pred = torch.squeeze(pred, 1)
            correct = torch.eq(pred, labels)
            result = correct[:].view(-1).float().sum(0, keepdim=True)
            return result.item() / batch_size


    def predict(self, x_test):
        self.model.eval()
        x_test = tensorized(x_test).to(device=self.device, dtype=torch.float32)

        # print("x_test.shape: ", x_test.shape)
        output = self.model(x_test)
        # print("output.shape: ", output.shape)
        # print("output: ", output)
        index = torch.argmax(output, dim=1)
        # print("index: ", index)
        return index.tolist()


    def calculate_profit(self, pred, df: pd.DataFrame):
        index2action = {0:-2, 1:-1, 2:0, 3:1, 4:2}
        pred = torch.argmax(pred, dim=1)
        result = []
        for index in pred.tolist():
            result.append(index2action.get(index, None))
            action_list = trend2action(result)

        profit = calculate_profit(df, action_list)
        return profit 
        




class LSTM (nn.Module):
    
    def __init__(self, input_feat_dim, hidden_feat_dim, num_layers):
        super(LSTM, self).__init__()
        self.model = nn.LSTM(input_feat_dim, hidden_feat_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_feat_dim, 5)

    
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










