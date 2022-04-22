import numpy as np
import pandas as pd
from LSTM import Model
from sklearn.preprocessing import StandardScaler


def load_g_truth(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        temp = f.read().splitlines()
    
    temp = list(map(int, temp))

    result = []
    for value in temp:
        if value == 1:
            result.append([1, 0, 0])
        elif value == 0:
            result.append([0, 1, 0])
        else:
            result.append([0, 0, 1])

    return result


def load_data(file_name):
    df = pd.read_csv(file_name, names=('open', 'high', 'low', 'close'))
    # df.columns = ['open', 'high', 'low', 'close'] # DO NOT use this.

    return df



def data_trans(df) -> list:
    '''
    Transform dataframe into 2D array for StandardScaler()
    '''
    result = []
    for index, row in df.iterrows():
        result.append([row['open'], row["high"], row["low"], row["close"]])

    return result

## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(x_data, y_data, window_size=50, offset=50):
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
        y.append(y_data[i])

    return np.array(X), np.array(y)




def split_train_test_set(x_data, y_data, split):
    # x_train, y_train, x_test, y_test = [], [], [], []
    split = len(x_data) - split
    x_train = x_data[0:split]
    y_train = y_data[0:split]
    x_test = x_data[split:]
    y_test = y_data[split:]

    return x_train, y_train, x_test, y_test


def main():
    
    raw_train_data = load_data("training.csv")
    split = len(raw_train_data.index)
    raw_test_data = load_data("testing.csv")
    raw_data = pd.concat([raw_train_data, raw_test_data], axis=0)
    raw_data = data_trans(raw_data)
    scaler = StandardScaler()
    raw_data = scaler.fit_transform(raw_data)
    g_truth = load_g_truth("g_truth.txt")
    x_data, y_data = extract_seqX_outcomeY(raw_data, g_truth, 180, 180)
    x_train, y_train, x_test, y_test = split_train_test_set(x_data, y_data, 20)
    
    model = Model()
    model.train(x_train, y_train)
    # ================ Prediction stage ================
    model.predict(x_test)

    








if __name__ == "__main__":
    main()
