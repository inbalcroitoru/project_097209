import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score,confusion_matrix



columns_reduced_next = ['song_duration_ms', 'acousticness',
                        'danceability', 'energy', 'instrumentalness', 'key', 'liveness',
                        'loudness', 'audio_mode', 'speechiness', 'tempo', 'time_signature',
                        'audio_valence', 'day','month','year', 'date','artist_name', 'playlist', 'song_popularity']


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
    med = np.median(input_file["song_popularity"])
    input_file['date'] = pd.to_datetime(input_file[['year', 'month', 'day']])
    input_file = input_file[columns_reduced_next]
    input_file = input_file.sort_values(by='date', ascending=True)
    scaler = MinMaxScaler()
    input_file[train_cols] = scaler.fit_transform(input_file[train_cols])

    input_file["playlist"] = input_file["playlist"].astype("category")
    input_file = pd.get_dummies(input_file, columns=["playlist"])
    return input_file




def split_train_test(df, test_size):
    test_idx = 831-test_size
    artists = df['artist_name'].unique()
    x_train = pd.DataFrame()
    y_train = pd.DataFrame()
    x_test = pd.DataFrame()
    y_test = pd.DataFrame()
    ratio = len(df[df['song_popularity'] == 1]) / (
                len(df[df['song_popularity'] == 1]) + len(df[df['song_popularity'] == 0]))
    for artist_idx, artist in enumerate(artists):
        df_artist = df[df['artist_name'] == artist]
        df_artist = df_artist.drop('artist_name', axis=1)
        if artist_idx < test_idx:
            df_artist = over_sampling(df_artist,ratio)
        df_artist['date'] = pd.to_datetime(df_artist[['year', 'month', 'day']])
        df_artist = df_artist.sort_values(by='date', ascending=True)
        df_artist.drop('date', axis=1, inplace=True)
        df_artist.drop(['year', 'month', 'day'],  axis=1, inplace=True)
        df_artist["song_popularity"] = np.where(df_artist["song_popularity"] >= 70, 1, 0)  # change trashold
        x = df_artist.copy()
        x.drop('song_popularity', axis=1, inplace=True)
        y = df_artist['song_popularity']
        if artist_idx < test_idx:
            x_train = pd.concat((x_train,x))
            y_train = pd.concat((y_train,y))
        else:
            x_test = pd.concat((x_test, x))
            y_test = pd.concat((y_test, y))
    return x_train, x_test, y_train, y_test

def over_sampling(data, ratio):
    new_data = pd.DataFrame(columns=data.columns)
    for idx, point in data.iterrows():
        # reshaped_point = point.reshape(1, -1)
        if point['song_popularity'] < 70:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            continue
        p = np.random.rand()
        if p > ratio:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
        else:
            new_data.loc[-1] = point
            new_data.index = new_data.index + 1
    return new_data

def under_sampling(data, ratio):
    new_data = pd.DataFrame(columns=data.columns)
    for idx, point in data.iterrows():
        if point['song_popularity'] < 70:
            p = np.random.rand()
            if p > ratio:
                new_data.loc[-1] = point
                new_data.index = new_data.index + 1
            continue
        new_data.loc[-1] = point
        new_data.index = new_data.index + 1
    return new_data


def main():
    test_size = 181
    input_file = make_data()
    x_train, x_test, y_train, y_test = split_train_test(input_file,test_size)

    print('len train: ', len(y_train))
    print('len test: ', len(y_test))
    print('the % of test is ', len(y_test) / (len(y_train) + len(y_test)))
    print('number of unpopular in train: ', len(y_train[y_train[0] == 0]))
    print('number of unpopular in test: ', len(y_test[y_test[0] == 0]))

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

    """Voting Classifier"""
    from sklearn.ensemble import VotingClassifier
    rf = RandomForestClassifier(random_state = 4)
    rf.fit(x_train,y_train)
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
