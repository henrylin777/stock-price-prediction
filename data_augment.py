

class DataAugmentation(object):

    def __init__(self, df):
        self.df = df
    
    def data_augment(self, N, data_type):
        data = self.df[data_type]

        maN = [ 0 for i in range(0, N)]
        maN_sub = [ 0 for i in range(0, N)]
        maxN_sub = [ 0 for i in range(0, N)]
        minN_sub = [ 0 for i in range(0, N)]

        for i in range(N, len(data)):
            value = sum(data[i-N:i])/N
            maN.append(value)
            maN_sub.append(data.iloc[i] - value)
            maxN_sub.append( max( data[i-N:i] ) - value)
            minN_sub.append( min( data[i-N:i] ) - value)


        # self.df.insert(len(self.df.columns), f"{data_type}_ma{N}", maN, True)
        # self.df.insert(len(self.df.columns), f"{data_type}_ma{N}_sub", maN_sub, True)
        # self.df.insert(len(self.df.columns), f"{data_type}_max{N}_sub", maxN_sub, True)
        # self.df.insert(len(self.df.columns), f"{data_type}_min{N}_sub", minN_sub, True)

        self.df.insert(len(self.df.columns), "_ma", maN, True)
        self.df.insert(len(self.df.columns), "_ma_sub", maN_sub, True)
        self.df.insert(len(self.df.columns), "_max_sub", maxN_sub, True)
        self.df.insert(len(self.df.columns), "_min_sub", minN_sub, True)

        return

