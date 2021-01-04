import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


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

columns = columns_chosen

class dataset(Dataset):
    def __init__(self, df, test_size):
        super().__init__()
        self.dataset = self.split_train_test(df,test_size)

    @staticmethod
    def split_train_test(df_artist): #test_size
        # test_idx = 831 - test_size
        train = list()
        test = list()
        # if artist_idx < test_idx:
        #     df_artist = knn(df_artist)
        #     df_artist.drop(['index', 'level_0'], axis=1, inplace=True)
        # df_artist['date'] = pd.to_datetime(df_artist[['year', 'month', 'day']])
        df_artist = df_artist.sort_values(by='date', ascending=True).reset_index()
        df_artist.drop('date', axis=1, inplace=True)
        df_artist.drop(['year', 'month', 'day'], axis=1, inplace=True)
        train_idx = int(np.floor(0.8*len(df_artist)))
        df_train = df_artist.head(train_idx)
        df_test = df_artist.tail(len(df_artist)-train_idx)
        df_train = knn(df_train)
        df_train.drop(['index', 'level_0'], axis=1, inplace=True)
        df_train_new = df_train.copy()
        df_test_new = df_test.copy()
        df_train["song_popularity"] = np.where(df_train_new["song_popularity"] >= 70, 1, 0)
        df_test["song_popularity"] = np.where(df_test_new["song_popularity"] >= 70, 1, 0)
        x_train = df_train.drop('song_popularity', axis=1)
        y_train = df_train['song_popularity']
        x_test = df_test.drop('song_popularity', axis=1)
        y_test = df_test['song_popularity']
        x_test.drop('index', axis=1 , inplace= True)
        for i in range(0, len(x_train), 3):
            if i >= len(x_train):
                break
            df_seq = x_train[i:i+3]
            df_seq_y = y_train[i:i+3]
            x_tr = torch.tensor(df_seq.values, dtype=torch.float, requires_grad=True)
            y_tr = torch.tensor(df_seq_y.values, dtype=torch.float, requires_grad=True)
            train.append((x_tr, y_tr))

        for i in range(0, len(x_test), 3):
            if i >= len(x_test):
                break
            df_seq = x_test[i:i + 3]
            df_seq_y = y_test[i:i + 3]
            x_te = torch.tensor(df_seq.values, dtype=torch.float, requires_grad=True)
            y_te = torch.tensor(df_seq_y.values, dtype=torch.float, requires_grad=True)
            test.append((x_te, y_te))
        return train, test

def knn(data):
    data = data.reset_index()
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
    data.drop('level_0',axis=1,inplace=True)
    extra_data.drop('level_0',axis=1,inplace=True)
    updated = pd.concat((data, extra_data)).reset_index()
    return updated


class LSTM(nn.Module):

    def __init__(self, num_features, hidden_size, linear_out_dim, num_layers=1):
        """
        :param num_features: Corresponds to the number of features in the input.
        :param hidden_size: Specifies the number of hidden layers along with the number of neurons in each layer.
        :param linear_out_dim:
        :param num_layers:
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers= 1 ,batch_first=True)
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=linear_out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=linear_out_dim, out_features=linear_out_dim)
        self.out_linear = nn.Linear(in_features=linear_out_dim, out_features=1)
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(linear_out_dim)
        self.batchnorm2 = nn.BatchNorm1d(linear_out_dim)

    def forward(self, year_data):
        x,_ = self.lstm(year_data)
        # x = self.relu(self.linear1(year_data.view(-1, year_data.shape[2])))
        x = self.relu(self.linear1(x.view(-1, x.shape[2])))
        if x.shape[0] != 1:
            x = self.batchnorm1(x)
        x = self.relu(self.linear2(x))
        if x.shape[0] != 1:
            x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.out_linear(x)
        return x.view(x.shape[1], -1)

def acc_evaluate1(data_loader, model, criterion):
    pred_list = list()
    real_list = list()
    with torch.no_grad():
        tmp_loss = 0
        tmp_acc = 0
        f1_0 = 0
        f1_1 = 0
        cnt_0 =0
        cnt_1 =0
        for input in data_loader:
            X, Y = input
            pred = model(X)
            accuracy, loss = binary_acc(pred, Y, criterion)  # Y.unsqueeze(1)
            report = classification_report(Y.tolist()[0], pred.tolist()[0], output_dict = True)
            if '0.0' in report:
                f1_0 += report['0.0']['f1-score']
                cnt_0 +=1
            if '1.0' in report:
                f1_1 += report['1.0']['f1-score']
                cnt_1 +=1
            tmp_loss += loss
            tmp_acc += accuracy
            y_pred_tag = torch.round(torch.sigmoid(pred))
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            pred_list.extend(pred.tolist()[0])
            real_list.extend(Y.tolist()[0])
            # print('Y REAL:', Y)
            # print('Y pred:', y_pred_tag)
            # print('acc:', accuracy)
        tmp_acc = tmp_acc / len(data_loader)
        tmp_loss = tmp_loss / len(data_loader)

    return tmp_acc, tmp_loss, [int(i) for i in pred_list], [int(i) for i in real_list]


def binary_acc(y_pred, y_test, criterion):
    y_pred_tag = y_pred
    # y_pred_tag[y_pred_tag <= 0.5] = 0
    # y_pred_tag[y_pred_tag > 0.5] = 1
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    # y_test = y_test.type(torch.float)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[1]
    acc = torch.round(acc * 100)
    loss = criterion(y_pred_tag, y_test)
    return acc, loss

def make_data():
    data = pd.read_csv("song_data.csv")
    info = pd.read_csv("song_info.csv")
    input_file = pd.merge(data, info)
    input_file.drop_duplicates(subset=["song_name"], inplace=True)
    input_file["day"].fillna(1, inplace=True)
    input_file = input_file.dropna()
    input_file['count'] = input_file.groupby("artist_name")["artist_name"].transform('count')
    input_file = input_file[input_file['count'] >= 6]
    med = np.median(input_file["song_popularity"])
    # input_file["song_popularity"] = np.where(input_file["song_popularity"] >= 70, 1, 0)  # change trashold
    input_file['date'] = pd.to_datetime(input_file[['year', 'month', 'day']])
    input_file = input_file[columns_reduced_next]
    input_file = input_file.sort_values(by='date', ascending=True)
    scaler = MinMaxScaler()
    input_file[train_cols] = scaler.fit_transform(input_file[train_cols])
    # input_file["artist_name"] = input_file["artist_name"].astype("category")
    # input_file = pd.get_dummies(input_file, columns=["artist_name"])
    # input_file = input_file.reset_index()
    # input_file = pd.get_dummies(input_file.reset_index(), columns=['level_0'])

    # input_file["playlist"] = input_file["playlist"].astype("category")
    # input_file = pd.get_dummies(input_file, columns=["playlist"])
    input_file.drop('playlist', axis=1, inplace=True)
    return input_file

def plot_graphs(train_accuracy, train_loss, test_accuracy, test_loss, model_str):
    x = list(range(1, len(train_accuracy) + 1))
    plt.plot(x, train_accuracy, c="royalblue", label="Train Accuracy", marker='o')
    plt.plot(x, test_accuracy, c="firebrick", label="Test Accuracy", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.xticks(x, rotation=45)
    plt.grid()
    plt.legend()
    plt.title(f'Train and test accuracy along epochs {model_str}')
    plt.show()

    plt.plot(x, train_loss, c="royalblue", label="Train Loss", marker='o')
    plt.plot(x, test_loss, c="firebrick", label="Test Loss", marker='o')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(x, rotation=45)
    plt.grid()
    plt.legend()
    plt.title(f'Train and test loss along epochs {model_str}')
    plt.show()

def main():
    df = make_data()
    hidden_size = 100
    linear_out_dim = 100
    epochs = 20
    acumulate_grad_steps = 1
    # test_size = 181
    artists = df['artist_name'].unique()
    epoch_dict_test_acc = defaultdict(float)
    epoch_dict_train_acc = defaultdict(float)
    epoch_dict_test_loss = defaultdict(float)
    epoch_dict_train_loss = defaultdict(float)
    real_epoch_dict = defaultdict(list)
    pred_epoch_dict = defaultdict(list)
    f1_0_list = list()
    f1_1_list = list()
    for art in artists:
        if art == 'Kanye West':
            print('1')
        df_artist = df[df['artist_name']== art]
        df_artist.drop('artist_name', axis=1, inplace=True)
        train, test = dataset.split_train_test(df_artist)
        train_dataloader = DataLoader(train, shuffle=True)
        test_dataloader = DataLoader(test, shuffle=False)
        num_features = train[0][0].shape[1]  # num of feature we use
        model = LSTM(num_features, hidden_size, linear_out_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(epochs):
            train_loss, acc_train = 0, 0
            i = 0
            for batch_idx, input_data in enumerate(train_dataloader):
                X, Y = input_data
                i += 1
                pred = model(X)
                loss = criterion(pred, Y)
                loss = loss / acumulate_grad_steps
                loss.backward()
                if i % acumulate_grad_steps == 0:
                    optimizer.step()
                    # for p in model.parameters():
                    #     print(p.grad)
                    model.zero_grad()
                train_loss += loss.item()

            # TEST
            acc_test, loss_test, pred_list, real_list = acc_evaluate1(test_dataloader, model, criterion)
            acc_train_new, loss_train_new,_,_ = acc_evaluate1(train_dataloader, model, criterion)
            epoch_dict_test_acc[epoch] += acc_test/len(artists)
            epoch_dict_train_acc[epoch] += acc_train_new/len(artists)
            epoch_dict_test_loss[epoch] += loss_test/len(artists)
            epoch_dict_train_loss[epoch] += loss_train_new/len(artists)
            real_epoch_dict[epoch] +=  real_list
            pred_epoch_dict[epoch] += pred_list

    for e in range(epochs):
        pred = pred_epoch_dict[e]
        real = real_epoch_dict[e]
        report = classification_report(real, pred, output_dict=True)
        f1_0_list.append(report['0']['f1-score'])
        f1_1_list.append(report['1']['f1-score'])

    print('train acc:', epoch_dict_train_acc)
    print('test acc:', epoch_dict_test_acc)
    print('train loss:', epoch_dict_train_loss)
    print('test loos:', epoch_dict_test_loss)
    print('f1_0:', f1_0_list)
    print('f1_1:', f1_1_list)

    # return epoch_dict_train_acc, epoch_dict_test_acc, f1_0_list , f1_1_list


if __name__ == '__main__':
    main()

