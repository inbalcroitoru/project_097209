from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, f1_score
from pystruct.models import chain_crf
from pystruct.learners import StructuredPerceptron
from pystruct.plot_learning import plot_learning as plt
import pandas as pd
import numpy as np
from time import time
from math import ceil
from sklearn.preprocessing import MinMaxScaler


columns_reduced_next = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence', 'artist_name','date','playlist', 'song_popularity']
columns_chosen = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence', 'song_popularity']
train_cols = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence']


def split_train_test(df, seq_length, test_size):
    test_size = test_size * (-1)
    artists = df['artist_name'].unique()
    test_artists=artists[test_size:]
    X_train,Y_train,X_test,Y_test = [],[],[],[]
    cnt_0,cnt_1=0,0
    for artist in artists:
        df_artist = df[df['artist_name'] == artist]
        artists_dates = df_artist.sort_values(by= 'date', ascending=True)
        artists_dates=artists_dates.reset_index(drop=True)
        num_records=len(df_artist)
        for i in range(0,num_records,seq_length):
            if i+seq_length > len(artists_dates):
                break
            df_seq=artists_dates[i:i+seq_length]
            df_seq=df_seq[columns_chosen]
            #TODO we need do deside featrs for X in the iloc
            X, Y = df_seq.iloc[:,:-1], df_seq.iloc[:,-1]
            X_chain = X.to_numpy()
            Y_chain = Y.to_numpy()
            if artist not in test_artists:
                X_train.append(X_chain)
                Y_train.append(Y_chain)
                cnt_1+=len(Y_chain[Y_chain==1])
                cnt_0+=len(Y_chain[Y_chain==0])
            else:
                X_test.append(X_chain)
                Y_test.append(Y_chain)
    # X_test = np.array(X_train[test_size:])
    # Y_test = np.array(Y_train[test_size:])
    # X_train = np.array(X_train[:test_size])
    # Y_train = np.array(Y_train[:test_size])

    print("Number of training sample chains: " + str(len(X_train)))
    print("Number of training label chains: " + str(len(Y_train)))
    print("Number of test sample chains: " + str(len(X_test)))
    print("Number of test label chains: " + str(len(Y_test)))
    print("0 in train:",cnt_0)
    print("1 in train:",cnt_1)

    return X_train, Y_train, X_test, Y_test

def evaluateModel(clf, data, labels, test_flag=False):
    prediction_start = time()
    F1 = None
    if test_flag:
        predictions = clf.predict(data)
        y_true, y_pred = [], []
        for true, pred in zip(labels, predictions):
            y_true += list(true)
            y_pred += list(pred)
        F1 = classification_report(y_true, y_pred)
        conf_matrix=confusion_matrix(y_true, y_pred)
        print("confusion_matrix:", conf_matrix)
    score = clf.score(data,labels)
    print("Accuracy: ", score)
    print('F1:', F1)
    prediction_end = time()
    print("Evaluation took " + str((prediction_end - prediction_start) / 60) + " minutes to complete\n")

def createModel(data, labels, num_classes=2):
    weight_class=np.ones(2)
    model = chain_crf.ChainCRF(directed=True)
    clf = StructuredPerceptron(model=model,max_iter=200,batch=False,average=True)
    print("Structured Perceptron + Chain CRF")
    train_start = time()
    clf.fit(X=data, Y=labels)
    train_end = time()
    print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
    return clf

def make_data():
    data=pd.read_csv("song_data.csv")
    info=pd.read_csv("song_info.csv")
    input_file=pd.merge(data,info)
    input_file.drop_duplicates(subset=["song_name"],inplace=True)
    input_file=input_file.dropna()
    input_file['count']=input_file.groupby("artist_name")["artist_name"].transform('count')
    print("overall data:",len(input_file))
    input_file=input_file[input_file['count'] >=3]
    print("after num songs data:",len(input_file))
    input_file["song_popularity"]=np.where(input_file["song_popularity"]>=70,1,0)#change trashold
    print("1:",len(input_file[input_file["song_popularity"]==1]))
    print("0:",len(input_file[input_file["song_popularity"]==0]))
    input_file['date'] = pd.to_datetime(input_file[['year', 'month','day']])
    input_file=input_file[columns_reduced_next]
    scaler = MinMaxScaler()
    input_file[train_cols] = scaler.fit_transform(input_file[train_cols])
    input_file["artist_name2"]= input_file["artist_name"].copy()
    input_file["artist_name2"]= input_file["artist_name2"].astype("category")
    input_file = pd.get_dummies(input_file, columns=["artist_name2"])
    input_file["playlist"] = input_file["playlist"].astype("category")
    input_file = pd.get_dummies(input_file, columns=["playlist"])
    return input_file


def main():
    df=make_data()
    start = time()
    df.to_csv('reduced.csv', index = False)
    artists = df['artist_name'].unique()
    test_size=181
    X_train, Y_train, X_test, Y_test = split_train_test(df, seq_length=3, test_size=test_size)
    num_classes_list = list()
    for chain in Y_train:
        for score in chain:
            num_classes_list.append(score)
    num_classes = len(np.unique(num_classes_list))
    model= createModel(X_train, Y_train, num_classes)
    print("train")
    evaluateModel(model, X_train, Y_train)
    print("test")
    evaluateModel(model, X_test, Y_test, test_flag=True)
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")
if __name__ == '__main__':
    main()
