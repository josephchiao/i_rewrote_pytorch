import numpy as np
import os
import shutil
import theta_init as theta_init

class NeuralNetwork:
    def __init__(self, dim = None, norm_fcn = None):
        
        self.dim = dim
        if dim is None:
            print('failed')
            quit()        
        self.leng = len(dim)

        if norm_fcn is None:
            norm_fcn = [sigmoid] * (self.leng - 1)
        self.norm_fcn = norm_fcn
        

    def theta_generate(self, n):

        """For initializing training set"""

        folder = 'matrix_library'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        for dataset in range(n):
            theta_init.create_file(self.dim,
                                   file_name = f"/Users/joseph_chiao/Desktop/Advance Research/Machine Learning/i_rewrote_pytorch/matrix_library/nn_theta_set_{dataset}.npz", ## Fix formating
                                   init_type = "logistic")

    def theta_recover(self, location):

        data = np.load(location, allow_pickle=True)
        self.theta = [data[f'arr_{i}'] for i in range(self.leng - 1)]
        self.b = [data[f'arr_{i}'] for i in range(self.leng - 1, self.leng * 2 - 2)]

    def theta_single_use(self):
        
        self.theta = [theta_init.logistic_theta_init(0, 1, (self.dim[i], self.dim[i+1]))  for i in range(len(self.dim) - 1)]  
        self.b = [np.zeros((1,self.dim[i+1])) for i in range(len(self.dim) - 1)]

    def feedforward(self, X):
        '''Vaugely tested, probably good. Spits back out all the layers'''
        layers = []
        layers.append(X)
        for i in range(self.leng-1):
            X_hid = np.dot(layers[-1], self.theta[i]) + self.b[i]
            layers.append(self.norm_fcn[i](X_hid))

        return layers

    def backward(self, X, y, learning_rate):

        layers = self.feedforward(X)
        delta = [(y - layers[-1]) * self.norm_fcn[-1](layers[-1], type = 'Derivative')]
        for i in range(2, self.leng):
            delta.append(np.dot(delta[-1], self.theta[-i+1].T) * self.norm_fcn[-i](layers[-i], type = 'Derivative'))
        
        for i in range(1, self.leng):
            self.theta[i-1] += np.dot(layers[i-1].T, delta[-i+3]) * learning_rate
            self.b[-i] += np.sum(delta[i-1], axis=0, keepdims=True) * learning_rate

        return layers

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            layers = self.backward(X, y, learning_rate) 
            if epoch % 4000 == 0:
                loss = np.mean(np.square(y - layers[-1]))
                print(f'Epoch {epoch}, Loss:{loss}')


  
def sigmoid(x, type = 'Normal'):
    if type == 'Derivative':
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def ReLU(x, type = 'Normal'):
    if type == 'Derivative':
        return 1 * (x > 0)
    return x * (x > 0)


NN = NeuralNetwork((2,5,5,4), [ReLU, sigmoid, sigmoid, sigmoid])
NN.theta_recover('/Users/joseph_chiao/Desktop/Advance Research/Machine Learning/i_rewrote_pytorch/matrix_library/nn_theta_set_0.npz')
print(NN.feedforward([np.array([[0,1], [1,0]])])[-1])
NN.train(np.array([[0,1], [1,0]]), np.array([[0,0.25,0.75,1], [0,0,1,1]]), 16001, 2)
print(NN.feedforward([np.array([[0,1], [1,0]])])[-1])


# # Basic test:
# NN = NeuralNetwork((2,2,2), [ReLU, ReLU])
# NN.theta = [np.array([[1,1], [1,1]]), np.array([[1,1], [1,1]])]
# NN.b = [np.array([[0,0]]), np.array([[0,0]])]
# print(NN.feedforward([np.array([1.5, 0.5])]))