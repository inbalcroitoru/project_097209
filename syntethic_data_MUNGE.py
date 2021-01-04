import pandas as pd
# from textblob import TextBlob
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# data=pd.read_csv("song_data.csv")
# info=pd.read_csv("song_info.csv")
# info = info.drop(['Unnamed: 0','Unnamed: 0.1'], axis=1)
# song_data=pd.merge(data,info)
# song_data.drop_duplicates(subset=["song_name"],inplace=True)
# # song_data = pd.read_csv("song_data_info_no_duplicate_have_date.csv")
# #song_data['sentiment'] = song_data['song_name'].map(lambda text: TextBlob(text).sentiment.polarity)
# # median_ = song_data.song_popularity.median()+1
# song_data["day"].fillna(1, inplace=True)
# song_data["song_popularity"] = [1 if i >= 70 else 0 for i in song_data.song_popularity]
# song_data = song_data.drop(columns=['song_name', 'album_names','artist_name','playlist'])
#
#
#
#
#
# y = song_data["song_popularity"].values
# x_data = song_data.drop(["song_popularity"], axis=1)
# # normalization
# # scaler = MinMaxScaler()
# # x = scaler.fit_transform(x_data)
# # all_idx = list(range(0,len(x_data)))
# # idc = np.random.choice(all_idx, replace = False, size = int(len(x_data)*0.8))
# # not_idc = [item for item in all_idx if item not in idc]
# x = pd.DataFrame(x_data)
# y = pd.DataFrame(y)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# # x_train, x_test, y_train, y_test = x_data[idc], x_data[not_idc], y[idc], y[not_idc]
#
# # y_train = not_idp.array([y_train]).T
# # data = np.concatenate((x_train, y_train), axis=1)
# new_data=x_train.copy()
# new_data['popularity']=y_train
# new_data=new_data.dropna()
def Knn_sampling(data):
    droped_data=data.drop(columns=['song_name', 'album_names','artist_name','playlist'])
    data.reset_index(drop=True,inplace=True)
    droped_data.reset_index(drop=True,inplace=True)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(droped_data)
    extra_data = pd.DataFrame(columns=data.columns) #was song_data

    for idx,point in droped_data.iterrows():
        # reshaped_point = point.reshape(1, -1)
        neigh = nbrs.kneighbors([point], return_distance=False)
        neigh_idx = neigh[0][1]
        original_point = data.loc[idx] #was song_data
        original_neigh = data.loc[neigh_idx] #was song_data
        new_point = original_neigh.copy()
        for name,value in original_neigh.iteritems():
            if name=='day':
                new_point[name] = min(original_neigh[name],original_point[name])
                continue
            p = np.random.rand()
            if type(value) == np.float64 and name not in ['month','year']:
                if p < 0.5:
                    mean = new_point[name]
                    std = abs(original_neigh[name]-original_point[name])/len(data)
                    norm = np.random.normal(mean,std)
                    new_point[name] = norm
            else:
                if p < 0.5:
                    new_point[name] = original_point[name]
        extra_data.loc[-1] = new_point
        extra_data.index = extra_data.index + 1
    extra_data.to_csv('new_dataset.csv')
    return extra_data

# Knn_sampling(new_data)

def advanced_knn(data):
    data = data.reset_index(drop=True)
    data = data.drop(['artist_name','date'], axis=1)
    song_data = data.copy()
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(song_data)
    extra_data = pd.DataFrame(columns=data.columns)

    for idx, point in song_data.iterrows():
        neigh = nbrs.kneighbors([point], return_distance=False)
        neigh_idx = neigh[0][1]
        original_point = data.loc[idx]
        original_neigh = data.loc[neigh_idx]
        new_point = original_neigh.copy()
        duplicate = False
        not_popular = False
        for name, value in original_neigh.iteritems():
            p = np.random.rand()
            if name in ('day', 'year', 'month'):
                new_point[name] = min(original_neigh[name], original_point[name])
                continue
            if type(value) == np.float64 or type(value) == np.float or name == 'song_popularity':
                norm = new_point[name]
                if p < 0.5:
                    mean = new_point[name]
                    std = abs(original_neigh[name] - original_point[name]) / len(data)
                    norm = np.random.normal(mean, std)
                    new_point[name] = norm
                if name == 'song_popularity' and norm >= 70:
                    duplicate = True
                if name == 'song_popularity' and norm < 70:
                    not_popular = True
            else:
                if p < 0.5:
                    new_point[name] = original_point[name]
        if not_popular:
            p = np.random.rand()
            if p < 0.2:
                extra_data.loc[-1] = new_point
                extra_data.index = extra_data.index + 1
            continue
        extra_data.loc[-1] = new_point
        extra_data.index = extra_data.index + 1
        if duplicate:
            extra_data.loc[-1] = new_point
            extra_data.index = extra_data.index + 1
            p = np.random.rand()
            if p < 0.5:
                extra_data.loc[-1] = new_point
                extra_data.index = extra_data.index + 1
    updated = pd.concat((data, extra_data)).reset_index()
    updated['date'] = pd.to_datetime(updated[['year', 'month', 'day']])
    updated=updated.drop(['year', 'month', 'day'], axis=1)
    return updated