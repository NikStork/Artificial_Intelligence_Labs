import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import model_selection, linear_model, svm
from sklearn.decomposition import PCA
import sklearn.datasets
import colorama
from colorama import Fore, Back, Style
from sklearn.neighbors import KNeighborsClassifier

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

    Y = df_train["ntype"]
    Y_t = df_test["ntype"]

    numeric = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    X = df_train[numeric]

    X_t = df_test[numeric]

    # Apply PCA to reduce the dimensionality of the data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    X_t_pca = pca.transform(X_t)

    model = sk.linear_model.LogisticRegression().fit(X_pca, Y)
    Y_pred = model.predict(X_t_pca)

    print(f"\nAccuracy={sk.metrics.accuracy_score(Y_t, Y_pred)}")

    # Define the range of x values
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5

    # Calculate corresponding y values for the decision boundary
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the training data
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap=plt.cm.coolwarm)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.show()

def method_kNN():
    data["ntype"] = data['type'].apply(lambda x: 0 if x == "White" else 1)

    # Let's split the data into a sample for training and testing
    df_train, df_test = sk.model_selection.train_test_split(data, train_size=0.2)

    numeric = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    # Обучающая выборка
    X_train = df_train[numeric]
    Y_train = df_train["ntype"]

    # Тестовая выборка
    X_test = df_test[numeric]
    Y_test = df_test["ntype"]

    model = KNeighborsClassifier(n_neighbors=10).fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(f"\nAccuracy={sk.metrics.accuracy_score(Y_test, Y_pred)}")


def method_SVM():
    data["ntype"] = data['type'].apply(lambda x: 0 if x == "White" else 1)

    # Let's split the data into a sample for training and testing
    df_train, df_test = sk.model_selection.train_test_split(data, train_size=0.2)

    numeric = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
               'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    # Обучающая выборка
    X_train = df_train[numeric]
    Y_train = df_train["ntype"]

    # Тестовая выборка
    X_test = df_test[numeric]
    Y_test = df_test["ntype"]

    model = sk.svm.SVC(kernel="linear", C=1000).fit(X_train, Y_train)
    Y_pred = model.predict(X_train)

    print(f"Linear accuracy = {sk.metrics.accuracy_score(Y_test, model.predict(X_test))}")
    print(f"Non-linSVM accuracy = {sk.metrics.accuracy_score(Y_test, model.predict(X_test))}")

    h = 0.02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict output for all points in meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision function and data points
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Function')
    plt.show()


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