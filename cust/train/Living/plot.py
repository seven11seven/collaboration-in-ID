"""
Plot training process
"""
from matplotlib import pyplot as plt
import os


def load_log(file_name):
    """ file content to list """
    with open(file_name, "r") as rf:
        lines = rf.readlines()
        lines = [line.strip().split(",") for line in lines]
    lines = lines[1:]
    numbers = []
    for line in lines:
        numbers.append([float(i) for i in line])
    return numbers


def plot_fd(numbers):
    #### dataset id
    x_0, y_0 = [], []
    x_1, y_1 = [], []
    x_2, y_2 = [], []
    for number in numbers:
        if number[1] == 0:
            x_0.append(number[0])
            y_0.append(number[-1])
        elif number[1] == 1:
            x_1.append(number[0])
            y_1.append(number[-1])
        elif number[1] == 3:
            x_2.append(number[0])
            y_2.append(number[-1])

    #### 
    plt.title("federated learning for client 3")

    plt.plot(x_0, y_0, label="loss on D_1")
    plt.plot(x_1, y_1, label="loss on D_2")
    plt.plot(x_2, y_2, label="loss on D_3")
    plt.legend()
    plt.show()


def plot_fd_eval(numbers):
    x_0, y_0 = [], []
    x_1, y_1 = [], []
    x_2, y_2 = [], []
    for number in numbers:
        if number[0]==0:
            x_0 = [1, 2]
            y_0 = [number[1], number[2]]
        elif number[0]==1:
            x_1 = [1, 2]
            y_1 = [number[1], number[2]]
        elif number[0]==2:
            x_2 = [1, 2]
            y_2 = [number[1], number[2]]
    #### 
    plt.title("pre- and post- error of clients' models")
    plt.plot(y_0, label="client 1")
    plt.plot(y_1, label="client 2")
    plt.plot(y_2, label="client 3")
    plt.legend()
    plt.show()


def plot_suc_fd():
    numbers_1 = load_log(file_name="./log/fd_2_0.txt")
    numbers_2 = load_log(file_name="./log/fd_2_1.txt")
    numbers_3 = load_log(file_name="./log/fd_2_3.txt")
    numbers_4 = load_log(file_name="./log/fd_2_4.txt")
    ####
    x_0, y_0 = [], []
    x_1, y_1 = [], []
    x_2, y_2 = [], []
    x_3, y_3 = [], []
    x_4, y_4 = [], []
    ####
    epoch = 0
    for number in numbers_1:
        if number[0] >149:
            break
        if number[1] == 0:
            epoch += 1
            x_0.append(epoch)
            y_0.append(number[-1])
        elif number[1] == 1:
            x_1.append(epoch)
            y_1.append(number[-1])

    for number in numbers_2:
        if number[0] >149:
            break
        if number[1] == 0:
            epoch += 1
            x_0.append(epoch)
            y_0.append(number[-1])
        elif number[1] == 1:
            x_2.append(epoch)
            y_2.append(number[-1])
    
    for number in numbers_3:
        if number[0] >149:
            break
        if number[1] == 0:
            epoch += 1
            x_0.append(epoch)
            y_0.append(number[-1])
        elif number[1] == 1:
            x_3.append(epoch)
            y_3.append(number[-1])
    
    for number in numbers_3:
        if number[0] >149:
            break
        if number[1] == 0:
            epoch += 1
            x_0.append(epoch)
            y_0.append(number[-1])
        elif number[1] == 1:
            x_4.append(epoch)
            y_4.append(number[-1])

    plt.title("loss in cascade training of client 3")
    plt.plot(x_0, y_0, label="package 1")
    plt.plot(x_1, y_1, label="package 2")
    plt.plot(x_2, y_2, label="package 3")
    plt.plot(x_3, y_3, label="package 4")
    plt.plot(x_4, y_4, label="package 5")
    plt.legend()
    plt.show()


def plot_self_train():
    with open("./log/on_self_loss.txt", "r") as rf:
        lines = rf.readlines()
        lines = [line.strip() for line in lines]
    for idx, line in enumerate(lines):
        if idx == 1:
            line = line.split(",")
            y_1 = [float(num) for num in line]
        if idx == 3:
            line = line.split(",")
            y_2 = [float(num) for num in line]
        if idx == 5:
            line = line.split(",")
            y_3 = [float(num) for num in line]
        if idx == 7:
            line = line.split(",")
            y_4 = [float(num) for num in line]

    plt.title("loss on self owned data")
    plt.plot(y_1[:500], label="client 1")
    plt.plot(y_2[:500], label="client 2")
    plt.plot(y_3[:500], label="client 3")
    plt.plot(y_4[:500], label="client 4")
    plt.legend()
    plt.show()


def plot_suc_fd_eval():

    y_1 = [1.63233, 1.65664, 1.65203, 1.70656, 1.68317]
    y_2 = [1.7049,  1.73335, 1.72879, 1.78018, 1.75539]
    y_3 = [1.70502, 1.74301, 1.70192, 1.72193, 1.66715]
    plt.title("eval error by cascade training")
    plt.plot(y_1, label="client 1")
    plt.plot(y_2, label="client 2")
    plt.plot(y_3, label="client 3")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    # numbers = load_log(file_name="./log/on_fd_dps.txt")
    # plot_fd_eval(numbers)
    plot_suc_fd_eval()
