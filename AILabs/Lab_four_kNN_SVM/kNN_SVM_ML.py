import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection, linear_model
import sklearn.datasets
import colorama
from colorama import Fore, Back, Style

# Выполнил: Филоненко Никита УВП-311

data = pd.read_csv("../../DataSet/wines.csv", delimiter=",", index_col=0, parse_dates=True)

def infAboutWines():
    print(Style.BRIGHT + Fore.GREEN + "\n\tInfo for dataset:" + Style.RESET_ALL)
    print(f"{data.info}")
    print(Style.BRIGHT + Fore.BLUE + f"\n\tNumber of zero lines:" + Style.RESET_ALL)
    print(f"{data.isnull().sum()}")

def decisionML():
    choice = input("\nMethod:\n1) Logistic Regression\n2) kNN\n3) SVM\n\n")

    match(choice):
        case "1":
            logisticRegression()
        case "2":
            method_kNN()
        case "3":
            method_SVM()
        case _:
            print("Try again.")


def logisticRegression():
    data["ntype"] = data['type'].apply(lambda x: 0 if x == "White" else 1)

    # Let's split the data into a sample for training and testing
    df_train, df_test = sk.model_selection.train_test_split(data, train_size=0.2)

    Y = df_train["type"]
    Y_t = df_test["type"]

    numeric = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide','total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    X = df_train[numeric]

    X_t = df_test[numeric]

    model = sk.linear_model.LogisticRegression().fit(X, Y)
    Y_pred = model.predict(df_test)

    print(f"Accuracy={sk.metrics.accuracy_score(Y_t, Y_pred)}")

    plt.scatter(data[:, 0], data[:, 1], c=['r' if x else 'b' for x in data])
    xs = data[:, 0].min(), data[:, 0].max()
    b = model.intercept_
    W = model.coef_[0]
    plt.plot(xs, [-b / W[1]-x * W[0] / W[1] for x in xs])
    plt.ylim([-2, 6])
    plt.show()

def method_kNN():
    print()

def method_SVM():
    print()



while True:
    a = input("\nWhat do you want to choose?\n1) Task: Machine learning (Logistic regression, kNN, SVM).\n2) Information about of dataset." +
              "\n\n\tEnter 'close', to exit.\n")

    match a:
        case "1":
            decisionML()
        case "2":
            infAboutWines()
        case "close":
            break
        case _:
            print("Try again.")