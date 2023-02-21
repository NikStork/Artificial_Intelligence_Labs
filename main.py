import numpy as np
import pandas as pd

# Выполнил: Филоненко Никита УВП-311

# Практическая работа №1. Информация о датасете


def doSomething(do):
    if do[2].__eq__("*"):
        square = int(do[0]) * int(do[4])
        print(f"Total multiplication: {square}")
    elif do[2].__eq__("+"):
        square = int(do) * int(do)
        print(f"Total summary: {square}")
    elif do[2].__eq__("-"):
        square = int(do) * int(do)
        print(f"Total subtraction: {square}")
    elif do[2].__eq__("/"):
        square = int(do) * int(do)
        print(f"Total division: {square}")
    else:
        print("Try again :0")

while True:
    do = input("Введите ваше действие: ").split()

    if do[2].__eq__("*"):
        doSomething(do)
    elif do[2].__eq__("/"):
        doSomething(do)
    elif do[2].__eq__("-"):
        doSomething(do)
    elif do[2].__eq__("+"):
        doSomething(do)
    elif do[2].__eq__("exit"):
        break
    else:
        print(do)