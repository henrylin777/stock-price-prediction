from json import load
import re
from tempfile import tempdir
import pandas as pd



def load_data(file_path):
    col_name = ('open', 'high', 'low', 'close')
    df = pd.read_csv(file_path, names=col_name)

    return df



class LabelGenerator():

    def __init__(self):
        self.cur_stock = None
        self.cur_trend = None
        self.buy_price = 0


    def generate_trend_list(self, stocks_data, close_price):

        length = len(stocks_data)
        result = []

        for idx in range(0, length-2):
            cur_price = stocks_data[idx]
            next_price = stocks_data[idx+1]
            nnext_price = stocks_data[idx+2]

            if cur_price > next_price:
                if next_price > nnext_price:
                    result.append(-2)
                elif next_price <= nnext_price:
                    result.append(-1)
            
            elif cur_price < next_price:
                if next_price >= nnext_price:
                    result.append(1)
                elif next_price < nnext_price:
                    result.append(2)
            
            else:
                result.append(0)
        

        # Deal with the action of day[-2] and day[-1]    
        if stocks_data[length-2] > stocks_data[length-1]:
            if stocks_data[length-1] > close_price:
                result.append(-2)
                result.append(-2)
            elif stocks_data[length-1] <= close_price:
                result.append(-1)
                result.append(-1)
        
        elif stocks_data[length-2] < stocks_data[length-1]:
            if stocks_data[length-1] >= close_price:
                result.append(1)
                result.append(1)
            elif stocks_data[length-1] < close_price:
                result.append(2)
                result.append(2)
        
        else:
            result.append(0)


        temp = []
        for value in result:
            if value == 2:
                temp.append([0, 0, 0, 0, 1])
            elif value == 1:
                temp.append([0, 0, 0, 1, 0])
            elif value == -1:
                temp.append([0, 1, 0, 0, 0])
            elif value == -2:
                temp.append([1, 0, 0, 0, 0])        
            else:
                temp.append([0, 0, 0, 0, 0])
                
        return temp



def write_output(actions):
    with open("g_truth.txt", "w", encoding="utf8") as f:
        for action in actions:
            f.write(f"{action}\n")




def main():
    stocks_df = load_data("training.csv")
    temp = load_data("testing.csv")
    stocks_df = pd.concat([stocks_df, temp], ignore_index=True)
    print(stocks_df)
    stocks_data = stocks_df["open"]
    close_price = stocks_df["close"].iloc[-1]

    worker = LabelGenerator()
    action_list = worker.generate_trend_list(stocks_data, close_price)
    for action in action_list:
        print(f"{action}")
    print("len(action_list): ", len(action_list))

    write_output(action_list)



if __name__ == "__main__":
    main()
