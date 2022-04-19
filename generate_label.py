from json import load
import re
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
        self.index2action = {0: "hold", 1: "buy", -1: "sell"}


    def generate_action_list(self, stocks_data, close_price):

        length = len(stocks_data)
        action_list = []

        # Initialization
        cur_price = stocks_data[0]
        next_price = stocks_data[1]
        nnext_price = stocks_data[2]

        if cur_price > next_price:
            self.cur_trend = -1
        elif cur_price < next_price:
            self.cur_trend = 1
        else:
            self.cur_trend = 0


        # Day 1: We do nothing
        action_list.append(None)


        # Day2: Special case
        if next_price > nnext_price :
            if self.cur_trend == 1:
                action_list.append(0)
                self.cur_stock = 0
            else:
                action_list.append(-1)
                self.cur_stock = -1

            self.cur_trend = -1

        elif next_price < nnext_price:
            action_list.append(1)
            self.buy_price = stocks_data[1]
            self.cur_stock = 1
            self.cur_trend = 1

        else:
            action_list.append(0)
            self.cur_stock = 0
            self.cur_trend = 0

        # =================================
        # Day3 to DayN
        for idx in range(2, length-1):
            cur_price = stocks_data[idx]
            next_price = stocks_data[idx+1]

            if cur_price > next_price:
                if self.cur_trend != -1:
                    if self.cur_stock != -1:
                        action_list.append(-1)
                        self.cur_stock = -1
                        self.buy_price = 0
                    else:
                        action_list.append(0)
                else:
                    action_list.append(0)
                
                self.cur_trend = -1
            
            elif cur_price < next_price:
                if self.cur_trend != 1:
                    if self.cur_stock != 1:
                        action_list.append(1)
                        self.buy_price = cur_price
                        self.cur_stock = 1
                    else:
                        action_list.append(0)

                else:
                    action_list.append(0)
                
                self.cur_trend = 1
            
            else:
                action_list.append(0)
                self.cur_trend = 0
        

        # Deal with the action of last day
        if self.cur_stock == -1:
            action_list.append(1)

        elif self.cur_stock == 1:
            if self.buy_price < close_price:
                action_list.append(-1)
            else:
                action_list.append(0)
        else:
             action_list.append(0)


        return action_list



def write_output(actions):
    with open("output.txt", "w", encoding="utf8") as f:
        for action in actions:
            f.write(f"{action}\n")





def main():
    stocks_df = load_data("training.csv")
    stocks_data = stocks_df["open"]
    close_price = stocks_df["close"].iloc[-1]

    worker = LabelGenerator()
    action_list = worker.generate_action_list(stocks_data, close_price)
    for action in action_list:
        print(f"{action}")
    print("len(action_list): ", len(action_list))

    write_output(action_list)



if __name__ == "__main__":
    main()
