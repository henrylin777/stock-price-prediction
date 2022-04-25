import numpy as np
import pandas as pd
from LSTM import Model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from data_augment import DataAugmentation
from utils import trend2action
from label_generator import LabelGenerator

def load_g_truth(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        temp = f.read().splitlines()
    
    temp = list(map(int, temp))

    result = []
    for value in temp:

        if value == 2:
            result.append([0, 0, 0, 0, 1])
        elif value == 1:
            result.append([0, 0, 0, 1, 0])
        elif value == -1:
            result.append([0, 1, 0, 0, 0])
        elif value == -2:
            result.append([1, 0, 0, 0, 0])        
        else:
            result.append([0, 0, 0, 0, 0])

    return result


def load_data(file_name):
    df = pd.read_csv(file_name, names=('open', 'high', 'low', 'close'))
    # df.columns = ['open', 'high', 'low', 'close'] # DO NOT use this.
    return df



def data_transform(df) -> list:
    '''
    Transform dataframe into 2D array for StandardScaler()
    '''
    result = []
    for index, row in df.iterrows():
        result.append(row.tolist())
        # result.append([row['open']])
    return result

## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(x_data:list, y_data:list, window_size:int, offset:int) -> tuple:
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data : (Dataframe) dataset 
        window_size : window size, e.g., 50 for 50 days of historical stock prices
        offset : position to start the split
    """
    X, y = [], []
    # print(len(y_data))
    # print(len(x_data))
    assert len(x_data) == len(y_data)
    
    for i in range(offset, len(x_data)):
        X.append( x_data[i-window_size : i] )
        y.append( y_data[i])

    return np.array(X), np.array(y)



def split_train_test_set(x_data, y_data, offset):
    # x_train, y_train, x_test, y_test = [], [], [], []
    split = len(x_data) - offset
    x_train = x_data[0:split]
    y_train = y_data[0:split]
    x_test = x_data[split:]
    y_test = y_data[split:]

    return x_train, y_train, x_test, y_test


def write_output(output, actions):
    with open(output, 'w', encoding='utf8') as outfile:
        for action in actions:
            outfile.write(f"{action}\n")





def main(training, testing, output):

    raw_train_data = load_data(training)
    raw_test_data = load_data(testing)

    raw_data = pd.concat([raw_train_data, raw_test_data], ignore_index=True)

    # ========== Prepare label data ==========
    Worker = LabelGenerator()
    g_truth = Worker.generate_trend_list(raw_data['open'], raw_data['close'].iloc[-1])

    # ========== Data Augmentation ==========
    data_augmenter = DataAugmentation(raw_data)
    for i in range(5, 65, 5):
        for dtype in ['open','close']:
            data_augmenter.data_augment(i, dtype)
    # for dtype in ['open','close']:
    #    data_augmenter.data_augment(3, dtype)

    # ========== Normalization ==========
    dataframe = data_transform(data_augmenter.df)
    scaler = StandardScaler()
    dataframe = scaler.fit_transform(dataframe)

    # ========== Prepare train & test set ==========
    x_data, y_data = extract_seqX_outcomeY(dataframe, g_truth, 60, 120)
    x_train, y_train, x_test, y_test = split_train_test_set(x_data, y_data, 19)

    # ========== Training  ==========
    input_dim = len(data_augmenter.df.columns)

    model = Model(input_dim, -100000)
    profit = model.train(x_train, y_train, x_test, y_test, raw_test_data)
    print(f"Round 1 | Best Profit: {profit}")

    for i in range(2, 5):
        model = Model(input_dim, profit)
        profit = model.train(x_train, y_train, x_test, y_test, raw_test_data)
        print(f"Round {i} | Best Profit: {profit}")


    # ================ Prediction Stage ================
    model = Model(len(data_augmenter.df.columns), profit)
    model.load_model()
    y_test_pred = model.predict(x_test)
    index2trend = {0:-2, 1:-1, 2:0, 3:1, 4:2}
    result = []
    for index in y_test_pred:
        result.append(index2trend.get(index, None))
    action_list = trend2action(result)

    print("Action: ", action_list)


    # ================ Calculate Profit ================
    from profit_calculator import calculate_profit
    profit = calculate_profit(raw_test_data, action_list)
    print("profit: ", profit)


    # =============== Write output ===============
    write_output(output, action_list)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    main(args.training, args.testing, args.output)
