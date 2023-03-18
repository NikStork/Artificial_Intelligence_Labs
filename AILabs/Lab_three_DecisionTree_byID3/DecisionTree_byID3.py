import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import tree
import graphviz
import colorama
from colorama import Fore, Back, Style

# Выполнил: Филоненко Никита УВП-311

data = pd.read_csv("../../DataSet/wines.csv", delimiter=",", index_col=0, parse_dates=True)

graphviz.backend.dot_command.DOT_BINARY = 'C:\\Program Files\\Graphviz\\bin\\dot.exe'

def infAboutWines():
    print(Style.BRIGHT + Fore.GREEN + "\n\tInfo for dataset:" + Style.RESET_ALL)
    print(f"{data.info}")
    print(Style.BRIGHT + Fore.BLUE + f"\n\tNumber of zero lines:" + Style.RESET_ALL)
    print(f"{data.isnull().sum()}")

def decisionTreeML():
    data["ntype"] = data['type'].apply(lambda x: 0 if x == "White" else 1)

    for s in data['total sulfur dioxide'].unique():
        print(f"Probability for total sulfur dioxide = {s} is {data[data['total sulfur dioxide'] == s]['ntype'].mean()}")

    # Let's split the data into a sample for training and testing
    df_train, df_test = sk.model_selection.train_test_split(data, train_size=0.2)

    Y = df_train["type"]
    Y_t = df_test["type"]

    numeric = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']


    X = df_train[numeric]
    model = sk.tree.DecisionTreeClassifier().fit(X, Y)

    X_t = df_test[numeric]
    Y_t_pred = model.predict(X_t)

    accuracy = sk.metrics.accuracy_score(Y_t, Y_t_pred)
    print(f"\nTotal accuracy: {accuracy}")

    graph = graphviz.Source(
        sk.tree.export_graphviz(model, feature_names=numeric, class_names=['White', 'Red'], filled=True, rounded=True, special_characters=True, impurity=False)
    )

    graph.render("iris")
    graph.view()


while True:
    a = input("\nWhat do you want to choose?\n1) Task: Decision Tree by algorithm ID3.\n2) Information about of dataset." +
              "\n\n\tEnter 'close', to exit.\n")

    match a:
        case "1":
            decisionTreeML()
        case "2":
            infAboutWines()
        case "close":
            break
        case _:
            print("Try again.")