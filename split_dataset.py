import csv
from sklearn.model_selection import train_test_split


def split_data(file_path: str, out_train: str, out_test: str):
    sentences = []
    labels = []
    with open(file_path, encoding="utf-8", mode='r') as in_file:
        for line in in_file:
            vals = line.strip().split(',')
            sentences.append([vals[0], vals[1], vals[2], vals[4], vals[5]])
            labels.append(vals[3])
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.1, random_state=42)
        with open(out_train, encoding="utf-8", mode='w', newline='') as out_file:
            csv_writer = csv.writer(out_file, delimiter=',')
            for i in range(len(y_train)):
                csv_writer.writerow([X_train[i][0], X_train[i][1], X_train[i][2], X_train[i][3], X_train[i][4], y_train[i]])
        with open(out_test, encoding="utf-8", mode='w', newline='') as out_file2:
            csv_writer = csv.writer(out_file2, delimiter=',')
            for i in range(len(y_test)):
                csv_writer.writerow([X_test[i][0], X_test[i][1], X_test[i][2], X_test[i][3], X_test[i][4], y_test[i]])


def split_data_valid(file_path: str, out_train: str, out_test: str):
    sentences = []
    labels = []
    with open(file_path, encoding="utf-8", mode='r') as in_file:
        for line in in_file:
            vals = line.strip().split(',')
            sentences.append([vals[0], vals[1], vals[2], vals[3], vals[4]])
            labels.append(vals[5])
        X_train, X_valid, y_train, y_valid = train_test_split(sentences, labels, test_size=0.1, random_state=42)
        with open(out_train, encoding="utf-8", mode='w', newline='') as out_file:
            csv_writer = csv.writer(out_file, delimiter=',')
            for i in range(len(y_train)):
                csv_writer.writerow(
                    [X_train[i][0], X_train[i][1], X_train[i][2], X_train[i][3], X_train[i][4], y_train[i]])
        with open(out_test, encoding="utf-8", mode='w', newline='') as out_file2:
            csv_writer = csv.writer(out_file2, delimiter=',')
            for i in range(len(y_valid)):
                csv_writer.writerow([X_valid[i][0], X_valid[i][1], X_valid[i][2], X_valid[i][3], X_valid[i][4], y_valid[i]])


if __name__ == "__main__":
    split_data("processed_data.csv", 'current_train.csv', 'test.csv')
    split_data_valid("current_train.csv", 'train.csv', 'valid.csv')

