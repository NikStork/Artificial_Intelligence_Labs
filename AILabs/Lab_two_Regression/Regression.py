import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.compose
import sklearn.linear_model
import matplotlib.pyplot as plt
import seaborn as sns
import colorama
from colorama import Fore, Back, Style

# Выполнил: Филоненко Никита УВП-311

data = pd.read_csv("../../DataSet/wines.csv", delimiter=",", index_col=0, parse_dates=True)

def infAboutWines():
    print(Style.BRIGHT + Fore.GREEN + "\n\tInfo for dataset:" + Style.RESET_ALL)
    print(f"{data.info}")
    print(Style.BRIGHT + Fore.BLUE + f"\n\tNumber of zero lines:" + Style.RESET_ALL)
    print(f"{data.isnull().sum()}")

def linearRegressionML():
    # Visualize the signs on the histograms
    data.hist(bins=50, figsize=(20, 15))
    plt.show()

    # Visualize the signs on the distribution
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (22, 12)})
    sns.pairplot(data=data, kind='scatter', diag_kind='kde')
    plt.show()

    # Let's split the data into a sample for training and testing
    df_train, df_test = sk.model_selection.train_test_split(data, train_size=0.2)

    Y = df_train["quality"]
    Y_t = df_test["quality"]

    numeric = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    categorical = ['type']

    X = df_train[numeric]
    model = sk.linear_model.LinearRegression()
    model = model.fit(X, Y)

    X_t = df_test[numeric]
    Y_t_pred = model.predict(X_t)

    # The output of the trained model is based only on numerical features
    print(f"MSE = {sk.metrics.mean_squared_error(Y_t, Y_t_pred)}, "
          f"MAE = {sk.metrics.mean_absolute_error(Y_t, Y_t_pred)}, "
          f"MAE(%) = {sk.metrics.mean_absolute_percentage_error(Y_t, Y_t_pred)}")

    transformer = sk.compose.make_column_transformer(
        (sk.preprocessing.OneHotEncoder(), ['type']),
        remainder='passthrough'
    )

    X = transformer.fit_transform(
        df_train[numeric + categorical]
    )

    model = sk.linear_model.LinearRegression()
    model = model.fit(X, Y)
    X_t = transformer.transform(df_test[numeric + categorical])
    Y_t_pred = model.predict(X_t)

    # Output of the trained model based on numerical and categorical features
    print(f"\nMSE = {sk.metrics.mean_squared_error(Y_t, Y_t_pred)}, "
          f"MAE = {sk.metrics.mean_absolute_error(Y_t, Y_t_pred)}, "
          f"MAE(%) = {sk.metrics.mean_absolute_percentage_error(Y_t, Y_t_pred)}")


while True:
    a = input("\nWhat do you want to choose?\n1) Task: Regression.\n2) Information about of dataset." +
              "\n\n\tEnter 'close', to exit.\n")

    match a:
        case "1":
            linearRegressionML()
        case "2":
            infAboutWines()
        case "close":
            break
        case _:
            print("Try again.")