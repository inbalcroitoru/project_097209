import pandas as pd
import numpy as np
from sklearn.utils import shuffle



def over_sampling(data):
    ## over sampling
    num_0 = len(data[data['song_popularity'] == 0])
    num_1 = len(data[data['song_popularity'] == 1])
    ratio = num_1 / num_0
    new_data = pd.DataFrame(columns=data.columns)
    cnt = 0
    for idx,point in data.iterrows():
        # reshaped_point = point.reshape(1, -1)
        if point['song_popularity'] ==0:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            continue
        p = np.random.rand()
        if p > 0.5:
            cnt+=1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
        else:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
    print('cnt ', cnt)
    print('over sampling ',len(new_data))
    new_data = shuffle(new_data)
    new_data.to_csv('over_sampling.csv')
    return new_data

def under_sampling(data):
    ## under sampling
    # num_0 = len(data[data['song_popularity'] == 0])
    # num_1 = len(data[data['song_popularity'] == 1])
    # ratio = num_1 / num_0
    new_data = pd.DataFrame(columns=data.columns)
    cnt = 0
    for idx,point in data.iterrows():
        # reshaped_point = point.reshape(1, -1)
        if point['song_popularity'] == 1:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            continue
        p = np.random.rand()
        if p <= 0.5:
            cnt += 1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
    print('cnt ', cnt)
    print('under sampling ', len(new_data))
    new_data = shuffle(new_data)
    new_data.to_csv('under_sampling.csv')
    return new_data


