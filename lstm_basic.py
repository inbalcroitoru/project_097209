import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import time
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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
    def split_train_test(df, test_size):
        artists = df['artist_name'].unique()
        data = list()
        for artist in artists:
            df_artist = df[df['artist_name'] == artist]
            df_artist = df_artist.sort_values(by='date', ascending=True)
            df_artist.drop('date', axis=1, inplace=True)
            df_artist.drop('artist_name', axis=1, inplace=True)
            X = df_artist.copy()
            X.drop('song_popularity', axis=1, inplace=True)
            Y = df_artist['song_popularity']
            X_chain = torch.tensor(X.values, dtype=torch.float, requires_grad=True)
            Y_chain = torch.tensor(Y.values, dtype=torch.float, requires_grad=True)
            data.append((X_chain, Y_chain))

        train = data[:831-test_size]
        test = data[831-test_size:]

        return train, test


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
        x = self.relu(self.linear1(x.view(-1, x.shape[2])))

        x = self.batchnorm1(x)
        x = self.relu(self.linear2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.out_linear(x)
        return x.view(x.shape[1], -1)


def acc_evaluate(X_test, Y_test, model, criterion):
    tmp_loss = 0
    tmp_acc = 0
    for X, Y in zip(X_test, Y_test):
        pred = model(X)
        accuracy, loss = binary_acc(pred, Y.unsqueeze(1), criterion)
        tmp_loss += loss
        tmp_acc += accuracy
        y_pred_tag = torch.round(torch.sigmoid(pred))
        print('Y REAL:', Y)
        print('Y pred:', y_pred_tag)
        print('acc:', accuracy)
    tmp_acc = tmp_acc / len(X_test)
    tmp_loss = tmp_loss / len(X_test)
    return tmp_acc, tmp_loss


def acc_evaluate1(data_loader, model, criterion):
    pred_list = list()
    real_list = list()
    with torch.no_grad():
        tmp_loss = 0
        tmp_acc = 0
        for input in data_loader:
            X, Y = input
            pred = model(X)
            accuracy, loss = binary_acc(pred, Y, criterion)  # Y.unsqueeze(1)
            tmp_loss += loss
            tmp_acc += accuracy
            y_pred_tag = torch.round(torch.sigmoid(pred))
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            pred_list.extend(pred.tolist()[0])
            real_list.extend(Y.tolist()[0])
        tmp_acc = tmp_acc / len(data_loader)
        tmp_loss = tmp_loss / len(data_loader)
    return tmp_acc, tmp_loss, [int(i) for i in pred_list], [int(i) for i in real_list]


def binary_acc(y_pred, y_test, criterion):
    y_pred_tag = y_pred
    # y_pred_tag[y_pred_tag <= 0.5] = 0
    # y_pred_tag[y_pred_tag > 0.5] = 1
    y_pred_tag = torch.round(torch.sigmoid(y_pred_tag))
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
    input_file = input_file[input_file['count'] >= 3]
    med = np.median(input_file["song_popularity"])
    input_file["song_popularity"] = np.where(input_file["song_popularity"] >= 70, 1, 0)  # change trashold
    input_file['date'] = pd.to_datetime(input_file[['year', 'month', 'day']])
    input_file = input_file[columns_reduced_next]
    input_file = input_file.sort_values(by='date', ascending=True)
    scaler = MinMaxScaler()
    input_file[train_cols] = scaler.fit_transform(input_file[train_cols])

    input_file["playlist"] = input_file["playlist"].astype("category")
    input_file = pd.get_dummies(input_file, columns=["playlist"])
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
    acumulate_grad_steps = 30
    train_loss_list = list()
    train_accuracy_list = list()
    test_accuracy_list = list()
    test_loss_list = list()
    start_year = time.time()
    test_size = 181

    train, test = dataset.split_train_test(df, test_size)
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
            # acc_train, loss = binary_acc(pred, Y, criterion)  # Y.unsqueeze(1)
            loss = criterion(pred, Y)
            loss = loss / acumulate_grad_steps
            loss.backward()
            # acc_train += acc_train.item()
            if i % acumulate_grad_steps == 0:
                optimizer.step()
                model.zero_grad()
            train_loss += loss.item()
        train_loss = train_loss * acumulate_grad_steps / len(train)
        acc_train = acc_train / len(train)

        # TEST
        acc_test, loss_test,pred_list, real_list = acc_evaluate1(test_dataloader, model, criterion)
        acc_train_new, loss_train_new,_, _ = acc_evaluate1(train_dataloader, model, criterion)

        train_loss_list.append(loss_train_new)
        train_accuracy_list.append(acc_train_new)
        test_accuracy_list.append(acc_test)
        test_loss_list.append(loss_test)
        print('epoch:', epoch)
        print('epoch time:', (time.time() - start_year))
        print('acc train new:', float(acc_train_new))
        print('acc test:', float(acc_test))
        print('loss train new:', float(loss_train_new))
        print('loss test:', float(loss_test))
        print('Classification report: \n', classification_report(pred_list, real_list))
        print('\n')
    plot_graphs(train_accuracy_list, train_loss_list, test_accuracy_list, test_loss_list, '')


if __name__ == '__main__':
    main()
