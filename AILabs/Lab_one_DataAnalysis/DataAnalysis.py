import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random as r
import colorama
from colorama import Fore, Back, Style

# Выполнил: Филоненко Никита УВП-311

data = pd.read_csv("../../DataSet/real_estate.csv", delimiter=",", index_col=0, parse_dates=True)

# Information about the dataset
def readDataset():
    print(Style.BRIGHT + Fore.GREEN + "\n\tInfo for dataset:" + Style.RESET_ALL)
    print(f"{data.info}")
    print(Style.BRIGHT + Fore.BLUE + f"\n\tNumber of zero lines:" + Style.RESET_ALL)
    print(f"{data.isnull().sum()}")

# Creating a graph using Matplotlib
def buildingGraphsMatplotlib_OneType():
    plt.figure(figsize=(22, 12), dpi=120)
    plt.plot(data.index, data["Price"], color="mediumslateblue")

    font = {'family': 'calibri',
            'color': 'orange',
            'weight': 'bold',
            'size': 39,
            }
    plt.ylabel("Price (thousands)", fontdict=font)
    plt.ylim(0, 135)
    plt.xlabel("Indexes", fontdict=font)
    plt.xlim(0, 416)
    plt.title("Line chart (Age of the house - The corresponding price)\nDataSet: real_estate.csv",
              fontdict={'family': 'calibri',
                        'color': 'g',
                        'weight': 'bold',
                        'size': 39,
                        })
    plt.show()

def buildingGraphsMatplotlib_TwoType():
    data_test = data.head(5)

    data_test.plot("Age", ["Price", "Distance To Transport", "Shops", "Longitude"], figsize=(22,10))

    font = {'family': 'calibri',
            'color': 'orange',
            'weight': 'bold',
            'size': 39,
            }
    plt.ylabel("Info for real estate", fontdict=font)
    plt.ylim(0, 1111)
    plt.xlabel("Age", fontdict=font)
    plt.title("Line chart (Age of the house - Info for real estate)\nDataSet: real_estate.csv",
              fontdict={'family': 'calibri',
                        'color': 'g',
                        'weight': 'bold',
                        'size': 39,
                        })
    plt.show()

# Creating a graph using Seaborn
def buildingGraphsSeaborn_OneType():
    sns.set_style("whitegrid")
    sns.set(rc={'figure.figsize': (22, 12)})
    sns.pairplot(data, hue="Shops")

    plt.show()

def buildingGraphsSeaborn_TwoType():
    data_test = data.sample(4)

    sns.set_style("whitegrid")
    temp = sns.FacetGrid(data=data_test, col='Shops', hue='Price', col_wrap=2)
    temp.map(sns.kdeplot, data=data_test, x='Age')
    temp.add_legend()

    plt.show()

# Creating a Bar charts using Matplotlib
def buildingBarChartsMatplotlib_OneType():
    df = data.groupby('Price').size().reset_index(name='Age')

    all_colors = list(plt.cm.colors.cnames.keys())
    n = df['Price'].unique().__len__() + 1
    r.seed(106)
    c = r.choices(all_colors, k=n)

    plt.figure(figsize=(22, 12), dpi=120)
    plt.bar(df['Age'], df['Price'], color=c, width=0.5)

    for i, val in enumerate(df['Price'].values):
        plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight': 500, 'size': 29})

    font = {'family': 'calibri',
            'color': 'orange',
            'weight': 'bold',
            'size': 39,
            }
    plt.ylabel("Price", fontdict=font)
    plt.xlabel("Age", fontdict=font)
    plt.gca().set_xticklabels(df['Age'], rotation=60, horizontalalignment='right')
    plt.title("Bar (Age of the house - The corresponding price)\nDataSet: real_estate.csv", fontdict={'family': 'calibri',
            'color': 'g',
            'weight': 'bold',
            'size': 39,
            })
    plt.show()

def buildingBarChartsMatplotlib_TwoType():
    data_test = data.head(5)

    all_colors = list(plt.cm.colors.cnames.keys())
    n = data_test['Price'].unique().__len__() + 1
    r.seed(106)
    color = r.choices(all_colors, k=n)

    data_test.plot(x="Age", y=["Price", "Distance To Transport", "Shops", "Latitude"], kind="bar", color=color, figsize=(22,10))

    font = {'family': 'calibri',
            'color': 'orange',
            'weight': 'bold',
            'size': 39,
            }
    plt.ylabel("Info for real estate", fontdict=font)
    plt.xlabel("Age", fontdict=font)

    plt.title("Bar chart (Age of the house - Info for real estate)\nDataSet: real_estate.csv", fontdict={'family': 'calibri',
            'color': 'g',
            'weight': 'bold',
            'size': 39,
            })
    plt.show()

# Creating a Bar charts using Seaborn
def buildingBarChartsSeaborn_OneType():
    data_test = data.sample(22)

    sns.set(rc={'figure.figsize': (22, 12)})
    sns.barplot(x="Age", y="Price", data=data_test, hue="Shops")

    sns.set_style("whitegrid")
    font = {'family': 'calibri',
            'color': 'orange',
            'weight': 'bold',
            'size': 39,
            }
    plt.ylabel("Price", fontdict=font)
    plt.xlabel("Age", fontdict=font)

    plt.title("Bar chart (Age of the house - The corresponding price)\nDataSet: real_estate.csv",
              fontdict={'family': 'calibri',
                        'color': 'g',
                        'weight': 'bold',
                        'size': 39,
                        })
    plt.show()

def buildingBarChartsSeaborn_TwoType():
    data_test = data.head(20)

    sns.set(rc={'figure.figsize': (22, 12)})
    sns.barplot(x="Age", y="Price", data=data_test)

    sns.set_style("darkgrid")
    font = {'family': 'calibri',
            'color': 'orange',
            'weight': 'bold',
            'size': 39,
            }
    plt.ylabel("Price", fontdict=font)
    plt.xlabel("Age", fontdict=font)

    plt.title("Bar chart (Age of the house - The corresponding price)\nDataSet: real_estate.csv",
              fontdict={'family': 'calibri',
                        'color': 'g',
                        'weight': 'bold',
                        'size': 39,
                        })
    plt.show()

while True:
    a = input("\nWhat do you want to choose?\n1) Task one (Information about the dataset)\n2) Task two (Linear hexagons)" +
              "\n3) Task three (Bar charts)\n\n\tEnter 'close', to exit.\n")

    match a:
        case "1":
            readDataset()
        case "2":
            choice = input("\nWhat is the way to build a graph?\n1) Matplotlib.pyplot\n2) Seaborn\n")

            if choice.__eq__("1"):
                double_choice = input("\n1) First type\n2) Second type\nEnter, please:\n")
                if double_choice.__eq__("1"):
                    buildingGraphsMatplotlib_OneType()
                elif double_choice.__eq__("2"):
                    buildingGraphsMatplotlib_TwoType()
                else:
                    print("Try again")
            elif choice.__eq__("2"):
                double_choice = input("\n1) First type\n2) Second type\nEnter, please:\n")
                if double_choice.__eq__("1"):
                    buildingGraphsSeaborn_OneType()
                elif double_choice.__eq__("2"):
                    buildingGraphsSeaborn_TwoType()
                else:
                    print("Try again")
            else:
                print("Try again.")
        case "3":
            choice = input("\nWhat is the way to build a graph?\n1) Matplotlib.pyplot\n2) Seaborn\n")

            if choice.__eq__("1"):
                double_choice = input("\n1) First type\n2) Second type\nEnter, please:\n")
                if double_choice.__eq__("1"):
                    buildingBarChartsMatplotlib_OneType()
                elif double_choice.__eq__("2"):
                    buildingBarChartsMatplotlib_TwoType()
                else:
                    print("Try again")
            elif choice.__eq__("2"):
                double_choice = input("\n1) First type\n2) Second type\nEnter, please:\n")
                if (double_choice.__eq__("1")):
                    buildingBarChartsSeaborn_OneType()
                elif (double_choice.__eq__("2")):
                    buildingBarChartsSeaborn_TwoType()
                else:
                    print("Try again.")
            else:
                print("Try again.")
        case "close":
            break
        case _:
            print("Try again.")