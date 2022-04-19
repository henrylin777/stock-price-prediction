import numpy as np
import pandas as pd
from LSTM import LSTM
from sklearn.preprocessing import StandardScaler

def load_data(file_name):
    df = pd.read_csv(file_name)
    df.columns = ['open', 'high', 'low', 'close']

    return df



def data_trans(df) -> list:
    result = []
    col = df.iloc[:,3]
    for value in col:
        result.append([value])

    return result

## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, window_size=50, offset=5):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data : dataset 
        window_size : window size, e.g., 50 for 50 days of historical stock prices
        offset : position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append( data[i-offset : i] )
        y.append(data[i])

    return np.array(X), np.array(y)


def main():
    
    train_data = load_data("training.csv")
    data_close = data_trans(train_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_close)
    # print(scaled_data)
    
    X_train, y_train = extract_seqX_outcomeY(scaled_data, window_size=50, offset=10)
    model = LSTM()
    model.train(X_train, y_train)

    
    








if __name__ == "__main__":
    main()
