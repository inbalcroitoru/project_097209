from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, f1_score
from pystruct.models import chain_crf
from pystruct.learners import StructuredPerceptron
import pandas as pd
import numpy as np
from time import time
from sklearn.preprocessing import MinMaxScaler
from resample_data import over_sampling,under_sampling

columns_reduced_next = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence','date', 'artist_name','song_popularity']#'playlist', 'artist_name'

columns_chosen = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence', 'song_popularity']



train_cols2 = ['song_duration_ms', 'acousticness',
       'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
       'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
       'audio_valence']


def read_data():
    data=pd.read_csv("song_data.csv")
    info=pd.read_csv("song_info.csv")
    info = info.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    input_file=pd.merge(data,info)
    input_file.drop_duplicates(subset=["song_name"],inplace=True)
    input_file["day"].fillna(1, inplace=True)
    input_file=input_file.dropna()
    input_file['count']=input_file.groupby("artist_name")["artist_name"].transform('count')
    print("overall data:",len(input_file))
    input_file=input_file[input_file['count'] >=3]
    print("after num songs data:",len(input_file))
    input_file["song_popularity"]=np.where(input_file["song_popularity"]>=70,1,0)#change trashold
    print("1:",len(input_file[input_file["song_popularity"]==1]))
    print("0:",len(input_file[input_file["song_popularity"]==0]))
    input_file = input_file.drop(['count'], axis=1)
    return input_file

def split_train_test(df, test_size):
    test_size = test_size * (-1)
    artists = df['artist_name'].unique()
    test_artists = artists[test_size:]
    test = df[df.artist_name.isin(test_artists)]
    train = df[~df.artist_name.isin(test_artists)]
    return train, test

def make_data(input_file):
    input_file['date'] = pd.to_datetime(input_file[['year', 'month', 'day']])
    input_file = input_file[columns_reduced_next]
    scaler = MinMaxScaler()
    input_file[train_cols2] = scaler.fit_transform(input_file[train_cols2]) #train_cols
    return input_file

def split_to_chains(df, seq_length,is_train=False):
    X_data,Y_data = [], []
    cnt_0, cnt_1 = 0, 0
    artists = df['artist_name'].unique()
    for artist in artists:
        df_artist = df[df['artist_name'] == artist]
        artists_dates = df_artist.sort_values(by='date', ascending=True)
        artists_dates = artists_dates.reset_index(drop=True)
        num_records = len(df_artist)
        for i in range(0, num_records, seq_length):
            if i + seq_length > len(artists_dates):
                break
            df_seq = artists_dates[i:i + seq_length]
            df_seq = df_seq[columns_chosen]
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
    clf = StructuredPerceptron(model=model,max_iter=1000,batch=False,average=True)
    print("Structured Perceptron + Chain CRF")
    train_start = time()
    clf.fit(X=data, Y=labels)
    train_end = time()
    print("Training took " + str((train_end - train_start) / 60) + " minutes to complete\n")
    return clf

def evaluateModel(clf, data, labels, test_flag=False):
    prediction_start = time()
    F1 = None
    if test_flag:
        predictions = clf.predict(data)
        # print("Predictions:")
        # print(predictions)

        y_true, y_pred = [], []
        for true, pred in zip(labels, predictions):
            y_true += list(true)
            y_pred += list(pred)
            #print("true:",true,"pred:",pred)
        F1 = classification_report(y_true, y_pred)
        conf_matrix=confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:", conf_matrix)
    score = clf.score(data,labels)
    print("Accuracy: ", score)
    print('Classification Report::', F1)
    prediction_end = time()
    print("Evaluation took " + str((prediction_end - prediction_start) / 60) + " minutes to complete\n")
    return score, F1

def sampling(Train,Test,is_over=False,is_under=False,is_knn=False):
    start = time()
    if is_under:
        print("\nunder_sampling \n")
        #new_train=pd.read_csv("under_sampling.csv")
        new_train = under_sampling(Train)

        print("num_0",len(new_train[new_train['song_popularity']==0]))
        print("num_1",len(new_train[new_train['song_popularity']==1]))
        # new_train = under_sampling(Train)
    elif is_over:
        print("\nover_sampling \n")
        new_train=pd.read_csv("over_sampling.csv")
        #new_train = over_sampling(Train)
        print("num_0",len(new_train[new_train['song_popularity']==0]))
        print("num_1",len(new_train[new_train['song_popularity']==1]))
    new_train = make_data(new_train)
    new_train.reset_index(drop=True,inplace=True)
    Test = make_data(Test)
    Test.reset_index(drop=True,inplace=True)


    X_train, Y_train = split_to_chains(new_train, seq_length=3, is_train=True)
    X_test, Y_test = split_to_chains(Test, seq_length=3)

    model = createModel(X_train, Y_train, 2)
    accuracy_train = evaluateModel(model, X_train, Y_train)
    print("train:", accuracy_train)
    accuracy_test = evaluateModel(model, X_test, Y_test, test_flag=True)
    print("test:", accuracy_test)
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")

def main():
    print("\nreg \n")
    df=read_data()
    start = time()
    df.to_csv('reduced.csv', index = False)
    artists = df['artist_name'].unique()
    #test_size=ceil(len(artists)/9)
    test_size=181
    Train,Test=split_train_test(df, test_size=test_size)
    train=make_data(Train.copy())
    test = make_data(Test.copy())
    x_train,y_train=split_to_chains(train,seq_length=3,is_train=True)
    x_test,y_test=split_to_chains(test,seq_length=3)
    num_classes_list = list()
    for chain in y_train:
        for score in chain:
             num_classes_list.append(score)
    num_classes = len(np.unique(num_classes_list))
    if num_classes<2:
         breakpoint()
    model= createModel(x_train, y_train, num_classes)
    accuracy_train = evaluateModel(model, x_train, y_train)
    print("train:",accuracy_train)
    accuracy_test= evaluateModel(model, x_test, y_test, test_flag=True)
    print("test:",accuracy_test)
    end = time()
    print("\nProcess took " + str((end - start) / 60) + " minutes to complete")
    sampling(Train,Test,is_over=True)
    #sampling(Train,Test,is_under=True)
    # sampling(Train,Test,is_knn=True)




if __name__ == '__main__':
    main()

