import numpy as np
import pandas as pd
from LSTM import LSTM
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
    df = pd.read_csv(file_name)
    df.columns = ['open', 'high', 'low', 'close']

    return df



def data_trans(df) -> list:
    '''
    Transform dataframe into 2D array for StandardScaler()
    '''
    result = []
    # col = df.iloc[:,3]
    # for value in col:
        # result.append([value])
    for index, row in df.iterrows():
        result.append([row['open'], row["high"], row["low"], row["close"]])

    return result

## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, window_size=50, offset=50):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data : (Dataframe) dataset 
        window_size : window size, e.g., 50 for 50 days of historical stock prices
        offset : position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append( data[i-window_size : i] )
        y.append(data[i])

    return np.array(X), np.array(y)


def main():
    
    train_data = load_data("training.csv")
    train_data = data_trans(train_data)
    scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # print(scaled_data)
    
    x_train, _ = extract_seqX_outcomeY(train_data, window_size=50, offset=50)

    y_train = load_g_truth("output.txt")
    y_train = y_train[50:]
    model = LSTM()
    model.train(x_train, y_train)

    # ================ Prediction stage ================
    temp = x_train[len(x_train)-1]
    temp = temp[2:]
    print(temp)
    temp = np.append(temp, [[153.65, 154.41, 153.08, 153.97]], axis=0)
    temp = np.append(temp, [[154.4, 155.02, 152.91, 154.76]],  axis=0)
    print(len(temp))
    # exit()
    # x_test = x_train[-1,:,:]
    model.predict(temp)
    # print(len(x_test))

    








if __name__ == "__main__":
    main()
