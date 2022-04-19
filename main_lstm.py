from re import L
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_name):
    df = pd.read_csv(file_name)
    print(df.iloc[:,0])
    return df


## Split the time-series data into training seq X and output value Y
def extract_seqX_outcomeY(data, N, offset):
    """
    Split time-series into training sequence X and outcome value Y
    Args:
        data - dataset 
        N - window size, e.g., 50 for 50 days of historical stock prices
        offset - position to start the split
    """
    X, y = [], []

    for i in range(offset, len(data)):
        X.append(data[i-N:i])
        y.append(data[i])

    return np.array(X), np.array(y)


def main():
    train_data = load_data("training.csv")
    exit()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_data)
    scaled_data_train = scaled_data[:]
    


if __name__ == "__main__":
    main()
