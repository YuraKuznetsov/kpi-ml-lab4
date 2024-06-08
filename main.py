import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import *

from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier


def metrics(model, x, y):
    predictions = model.predict(x)
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted')
    recall = recall_score(y, predictions, average='weighted')
    f_score = f1_score(y, predictions, average='weighted')
    mcc = matthews_corrcoef(y, predictions)
    balanced_acc = balanced_accuracy_score(y, predictions)

    return {
      'Accuracy': accuracy,
      'Precision': precision,
      'Recall': recall,
      'F1-Score': f_score,
      'MCC': mcc,
      'Balanced Accuracy': balanced_acc,
    }


def main():
    # 1
    df = pd.read_csv('WQ-R.csv', sep=';')

    # 2
    print("Rows:", df.shape[0])
    print("Cols:", df.shape[1])

    # 3
    for i, c in enumerate(df.columns):
        print(f'{i + 1}) {c}')

    # 4
    spliter = ShuffleSplit(n_splits=10, train_size=0.8, random_state=1)
    train_indexes, test_indexes = list(spliter.split(df))[7]

    df_train, df_test = df.loc[train_indexes], df.loc[test_indexes]
    df_train['quality'].value_counts()
    df_test['quality'].value_counts()

    # 5
    x_train, y_train = df_train.iloc[:, :-1], df_train.iloc[:, -1]
    x_test, y_test = df_test.iloc[:, :-1], df_test.iloc[:, -1]

    model_k = KNeighborsClassifier()
    model_k.fit(x_train, y_train)

    # 6
    test_k = metrics(model_k, x_test, y_test)
    train_k = metrics(model_k, x_train, y_train)

    plt.bar(test_k.keys(), test_k.values())
    plt.title('Test knn metrics values')
    plt.show()

    # 7
    xs, ys_test, ys_train = [], [], []
    for i in range(1, 21):
        model = KNeighborsClassifier(i)
        model.fit(x_train, y_train)
        xs.append(i)
        ys_train.append(balanced_accuracy_score(y_train, model.predict(x_train)))
        ys_test.append(balanced_accuracy_score(y_test, model.predict(x_test)))

    plt.plot(xs, ys_test, label='test')
    plt.plot(xs, ys_train, label='train')
    plt.legend()
    plt.title('Bплив кількості сусідів на результати класифікації.')
    plt.xticks(range(1, 21))
    plt.show()


if __name__ == '__main__':
    main()

