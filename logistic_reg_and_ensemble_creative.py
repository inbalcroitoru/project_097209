import pandas as pd
from textblob import TextBlob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,recall_score,precision_score,confusion_matrix,f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


columns_reduced_next = ['song_duration_ms', 'acousticness',
                        'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                        'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
                        'audio_valence', 'date', 'artist_name', 'playlist', 'song_popularity']

columns_no_date = ['song_duration_ms', 'acousticness',
                   'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                   'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
                   'audio_valence', 'year', 'song_popularity']

columns_chosen = ['song_duration_ms', 'acousticness',
                  'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                  'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
                  'audio_valence', 'playlist', 'song_popularity']

train_cols = ['song_duration_ms', 'acousticness',
              'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
              'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
              'audio_valence']

def make_data():
    data = pd.read_csv("song_data.csv")
    info = pd.read_csv("song_info.csv")
    input_file = pd.merge(data, info)
    input_file.drop_duplicates(subset=["song_name"], inplace=True)
    input_file["day"].fillna(1, inplace=True)
    input_file = input_file.dropna()
    input_file['count'] = input_file.groupby("artist_name")["artist_name"].transform('count')
    input_file = input_file[input_file['count'] >= 3]
    # input_file["song_popularity"] = np.where(input_file["song_popularity"] >= 70, 1, 0)  # change trashold
    input_file['date'] = pd.to_datetime(input_file[['year', 'month', 'day']])
    input_file = input_file[columns_reduced_next]
    input_file = input_file.sort_values(by='date', ascending=True)
    scaler = MinMaxScaler()
    input_file[train_cols] = scaler.fit_transform(input_file[train_cols])
    return input_file


def split_train_test(df):  # test_size
    x_train_list, y_train_list = pd.DataFrame(), pd.DataFrame()
    x_test_list, y_test_list = pd.DataFrame(), pd.DataFrame()
    for artist in df['artist_name'].unique():
        df_artist = df[df['artist_name'] == artist]
        df_artist = df_artist.sort_values(by='date', ascending=True).reset_index()
        df_artist.drop(['artist_name','playlist'], axis=1, inplace=True)
        df_artist.drop('date', axis=1, inplace=True)
        train_idx = int(np.floor(0.8 * len(df_artist)))
        df_train = df_artist.head(train_idx)
        df_test = df_artist.tail(len(df_artist) - train_idx)
        df_train = knn(df_train,df_train)
        df_train.drop(['index'], axis=1, inplace=True)
        df_train_new = df_train.copy()
        df_test_new = df_test.copy()
        df_train["song_popularity"] = np.where(df_train_new["song_popularity"] >= 70, 1, 0)
        df_test["song_popularity"] = np.where(df_test_new["song_popularity"] >= 70, 1, 0)
        x_train = df_train.drop('song_popularity', axis=1)
        y_train = df_train['song_popularity']
        x_test = df_test.drop('song_popularity', axis=1)
        y_test = df_test['song_popularity']
        x_test.drop('index', axis=1, inplace=True)
        x_train_list = pd.concat((x_train_list,x_train))
        y_train_list = pd.concat((y_train_list,y_train))
        x_test_list = pd.concat((x_test_list,x_test))
        y_test_list = pd.concat((y_test_list,y_test))
    return x_train_list,x_test_list, y_train_list, y_test_list


def knn(data, song_data):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
    extra_data = pd.DataFrame(columns=song_data.columns)

    for idx,point in data.iterrows():
        neigh = nbrs.kneighbors([point], return_distance=False)
        neigh_idx = neigh[0][1]
        original_point = song_data.loc[idx]
        original_neigh = song_data.loc[neigh_idx]
        new_point = original_neigh.copy()
        duplicate = False
        not_popular = False
        for name,value in original_neigh.iteritems():
            p = np.random.rand()
            if type(value) == np.float64 or type(value) == np.float:
                norm = new_point[name]
                if p < 0.5:
                    mean = new_point[name]
                    std = abs(original_neigh[name]-original_point[name])/len(data)
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
            if p < 0.05:
                extra_data.loc[-1] = new_point
                extra_data.index = extra_data.index + 1
            continue
        extra_data.loc[-1] = new_point
        extra_data.index = extra_data.index + 1
        if duplicate:
            extra_data.loc[-1] = new_point
            extra_data.index = extra_data.index + 1
    return pd.concat((data, extra_data))


def main():
    test_size = 181
    input_file = make_data()
    x_train, x_test, y_train, y_test = split_train_test(input_file)

    y_train.rename(columns={0: 'song_popularity'}, inplace=True)
    y_test.rename(columns={0: 'song_popularity'}, inplace=True)


    print('len train: ', len(y_train))
    print('len test: ', len(y_test))
    print('the % of test is ', len(y_test) / (len(y_train) + len(y_test)))
    print('number of unpopular in train: ', len(y_train[y_train['song_popularity'] == 0]))
    print('number of unpopular in test: ', len(y_test[y_test['song_popularity'] == 0]))

    # x_train.drop(['playlist'], axis=1, inplace=True)
    # x_test.drop(['playlist'], axis=1, inplace=True)

    # x_train["playlist"] = x_train["playlist"].astype("category")
    # x_train_ = pd.get_dummies(x_train, columns=["playlist"])
    #
    # x_test["playlist"] = x_test["playlist"].astype("category")
    # x_test_ = pd.get_dummies(x_test, columns=["playlist"])

    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    #
    # x_train = pd.DataFrame(x_train).reset_index()
    # x_test = pd.DataFrame(x_test).reset_index()


    #logistic regression
    clf = LogisticRegression(random_state=0, max_iter=400, solver='lbfgs').fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print("LogisticRegression ", acc)

    cm = confusion_matrix(y_test,y_pred)
    print('Classification report: \n',classification_report(y_test, y_pred))

    sns.heatmap(cm, annot=True, fmt="d").set(xlabel='Predicted value', ylabel='Ground truth')
    plt.show()

    """random forest"""

    from sklearn.ensemble import RandomForestClassifier
    rf=RandomForestClassifier(n_estimators=150,random_state = 3)
    rf.fit(x_train,y_train)
    print("Train accuracy of random forest",rf.score(x_train,y_train))
    print("Test accuracy of random forest",rf.score(x_test,y_test))
    RandomForestClassifier_score=rf.score(x_test,y_test)
    y_pred=rf.predict(x_test)
    t_true=y_test

    from sklearn.model_selection import cross_val_score
    k = 10
    cv_result = cross_val_score(rf,x_train,y_train,cv=k) # uses R^2 as score
    cv_result_randomforest=np.sum(cv_result)/k
    print('Cross_val Scores: ',cv_result)
    print('Cross_val scores average: ',np.sum(cv_result)/k)


    rf = RandomForestClassifier(random_state = 4)
    rf.fit(x_train,y_train)
    y_pred = rf.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    print('Confusion matrix: \n',cm)
    print('Classification report: \n',classification_report(y_test,y_pred))

    sns.heatmap(cm,annot=True,fmt="d").set(xlabel='Predicted value', ylabel='Ground truth')
    plt.show()

    """Voting Classifier"""
    from sklearn.ensemble import VotingClassifier
    ensemble=VotingClassifier(estimators=[('Random Forest', rf), ('Logistic Regression', clf)],
                           voting='soft', weights=[2,1]).fit(x_train,y_train)
    print('The train accuracy for Random Forest and Logistic Regression is:',ensemble.score(x_train,y_train))
    print('The test accuracy for Random Forest and Logistic Regression is:',ensemble.score(x_test,y_test))

    from sklearn.model_selection import cross_val_score
    k = 5
    cv_result = cross_val_score(ensemble,x_train,y_train,cv=k) # uses R^2 as score
    print('Cross_val Scores: ',cv_result)
    print('Cross_val scores average: ',np.sum(cv_result)/k)
    y_pred = ensemble.predict(x_test)
    print('Classification report: \n',classification_report(y_test,y_pred))


if __name__ == '__main__':
    main()
