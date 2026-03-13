import neural_network as nnw
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def load_training_example(type = 'train'):

    return(pd.read_csv(f'test_library/MNIST_CSV/mnist_{type}.csv').to_numpy())



def display_test(np_array, i):
    ex_entry = np_array[i][0]
    ex = np.reshape(np_array[i][1:], (28,28))
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(ex, origin='upper')   
    fig.suptitle(ex_entry)
    plt.show()

def number_display(X, Y='NaN'):
    plt.style.use('_mpl-gallery-nogrid')
    fig, ax = plt.subplots()
    ax.imshow(X, origin='upper')   
    fig.suptitle(Y)
    plt.show()


# display_test(load_training_example(), 6)

def feed_NN_data(np_array):
    zeros = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    return np_array[:, 1:]/255, np.array([zeros[i] for i in np_array[:, 0]])

def training_basic(np_array):
    NN = nnw.NeuralNetwork((784,100,50,10), [nnw.ReLU, nnw.ReLU, nnw.sigmoid], 'test_library/MNIST_CSV')
    NN.theta_generate()
    NN.theta_recover()
    X, Y = feed_NN_data(np_array)
    for i in range(100):
        print(i)
        NN.train(X[i*100:(i+1)*100], Y[i*100:(i+1)*100], 1000, [0.01, 0.001][i >= 50])
    NN.theta_save()

training_basic(load_training_example())

# NN = nnw.NeuralNetwork((784,100,50,10), [nnw.ReLU, nnw.sigmoid, nnw.sigmoid, nnw.sigmoid], 'test_library/MNIST_CSV')
# NN.theta_generate()
# NN.theta_recover()
# X, Y = feed_NN_data(load_training_example())
# number_display(np.reshape(X[2], (28,28)), Y[2])
# NN.train(np.array([X[2]]), np.array([Y[2]]), 8000, 0.1)
# print(np.argmax(NN.feedforward(X[2])[-1]))
# print(NN.feedforward(X[2])[-1])

NN = nnw.NeuralNetwork((784,100,50,10), [nnw.ReLU, nnw.ReLU, nnw.sigmoid], 'test_library/MNIST_CSV')
NN.theta_recover()
X, Y = feed_NN_data(load_training_example(type = 'test'))
Y_predict_mx = NN.feedforward(X)[-1]
Y_predict = np.argmax(Y_predict_mx, axis = 1)
correction_rate = sum([Y[i][Y_predict[i]] for i in range(len(Y_predict))])/len(Y)
print('correction rate = ', correction_rate)

for i in range(len(Y)):
    Y_predict_mx = NN.feedforward(X[i])[-1]
    Y_predict = np.argmax(Y_predict_mx)
    number_display(np.reshape(X[i], (28,28)), Y_predict)



