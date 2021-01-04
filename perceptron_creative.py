from pystruct.models import chain_crf
from pystruct.learners import StructuredPerceptron
import pandas as pd
import numpy as np
from time import time
from syntethic_data_MUNGE import advanced_knn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, f1_score
from math import ceil




columns_reduced_next = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence', 'artist_name', 'date','year','day','month', 'song_popularity']

columns_chosen = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence', 'song_popularity']

train_cols = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence']

columns = columns_chosen

def make_data():
    data=pd.read_csv("song_data.csv")
    info=pd.read_csv("song_info.csv")
    info = info.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    input_file=pd.merge(data,info)
    input_file.drop_duplicates(subset=["song_name"],inplace=True)
    input_file["day"].fillna(1, inplace=True)
    input_file=input_file.dropna()
    input_file['count']=input_file.groupby("artist_name")["artist_name"].transform('count')
    print("overall data:",len(input_file))
    input_file=input_file[input_file['count'] >=6]
    print("after num songs data:",len(input_file))
    #input_file["song_popularity"]=np.where(input_file["song_popularity"]>=70,1,0)#change trashold
    #print("1:",len(input_file[input_file["song_popularity"]==1]))
    #print("0:",len(input_file[input_file["song_popularity"]==0]))
    input_file = input_file.drop(['count'], axis=1)
    input_file['date'] = pd.to_datetime(input_file[['year', 'month', 'day']])
    input_file = input_file[columns_reduced_next]  # columns_reduced_next
    input_file = input_file.sort_values(by='date', ascending=True)
    scaler = MinMaxScaler()
    input_file[train_cols] = scaler.fit_transform(input_file[train_cols])
    return input_file


def split_train_test(df_artist):
    train_idx = int(np.floor(0.8 * len(df_artist)))
    df_train = df_artist[:train_idx]
    df_test=df_artist[train_idx:]
    df_test=df_test.drop(['year', 'month', 'day'], axis=1)
    df_train = advanced_knn(df_train)
    df_train=df_train.drop(['index'], axis=1) #check
    df_train["song_popularity"] = np.where(df_train["song_popularity"] >= 70, 1, 0)
    df_test["song_popularity"] = np.where(df_test["song_popularity"] >= 70, 1, 0)
    return df_train, df_test

def split_to_chains(df_artist, seq_length,is_train=False):
    X_data,Y_data = [], []
    cnt_0, cnt_1 = 0, 0
    artists_dates = df_artist.sort_values(by='date', ascending=True)
    artists_dates = artists_dates.reset_index(drop=True)
    num_records = len(df_artist)
    for i in range(0, num_records, seq_length):
        if i + seq_length > len(artists_dates):
            break
        df_seq = artists_dates[i:i + seq_length]
        df_seq = df_seq[columns_chosen]
        # TODO we need do deside featrs for X in the iloc
        X, Y = df_seq.iloc[:, :-1], df_seq.iloc[:, -1]
        X_chain = X.to_numpy()
        Y_chain = Y.to_numpy()
        X_data.append(X_chain)
        Y_data.append(Y_chain)
        cnt_1 += len(Y_chain[Y_chain == 1])
        cnt_0 += len(Y_chain[Y_chain == 0])
    if is_train:
        print("Number of training sample chains: " + str(len(X_data)))
        print("Number of training label chains: " + str(len(Y_data)))
        print("0 in train:", cnt_0)
        print("1 in train:", cnt_1)
    else:
        print("Number of test sample chains: " + str(len(X_data)))
        print("Number of test label chains: " + str(len(Y_data)))

    return X_data,Y_data



def createModel(data, labels, num_classes=2):
    weight_class=np.ones(2)
    model = chain_crf.ChainCRF(directed=True)
    clf = StructuredPerceptron(model=model,max_iter=200,verbose=False,batch=False,average=True)
    print("Structured Perceptron + Chain CRF")
    train_start = time()
    clf.fit(X=data, Y=labels)
    train_end = time()
    print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
    return clf


def get_seq_len(test_size):
    if test_size < 3:
        seq_length = test_size
    else:
        seq_length = 3
    return seq_length
def tmp(y_train):
    cnt_0=0
    num_classes_list = list()
    for chain in y_train:
        for score in chain:
            num_classes_list.append(score)
            if score==0:
                cnt_0+=1

    num_classes = len(np.unique(num_classes_list))
    if num_classes == 1 and not cnt_0:
        y_train[0][0] = 0
        y_train[0][1] = 1
    return y_train

def main():
    df=make_data()
    start = time()
    df.to_csv('advanced.csv', index = False)
    y_true=[]
    y_pred=[]
    artists_names = df['artist_name'].unique()
    for artist in artists_names:
        df_artist = df[df['artist_name']==artist]
        #num_classes = df_artist['song_popularity'].nunique()
        #artist = str(artist) + " -> " + str(df_artist['Team'].unique()[0])
        print("\n"+artist+"\n")
        test_size=ceil(0.2*len(df_artist)) #the num is for now,maybe change
        train,test = split_train_test(df_artist)
        if artist =="Kanye West":
            print("hi")
        x_train, y_train = split_to_chains(train, seq_length=3, is_train=True)
        y_train=tmp(y_train)
        seq_length=get_seq_len(test_size)
        x_test, y_test = split_to_chains(test, seq_length=seq_length)
        model = createModel(x_train, y_train, 2)
        predictions = model.predict(x_test)
        for true, pred in zip(y_test, predictions):
            y_true += list(true)
            y_pred += list(pred)


    F1 = classification_report(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print("confusion_matrix:", conf_matrix)
    score = accuracy_score(y_true, y_pred)
    print("Accuracy: ", score)
    print('F1:', F1)


if __name__ == '__main__':
    main()
