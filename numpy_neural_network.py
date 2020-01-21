# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 09:08:00 2019

@author: Xzw

E-mail: diligencexu@gmail.com
"""
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_info(train_info, hidden_unit, learning_rate):
    plt.plot(list(range(1,
                        len(train_info['loss']) + 1)),
             train_info['loss'],
             label='loss',
             linewidth=1,
             color='r')
    plt.plot(list(range(1,
                        len(train_info['weight_loss']) + 1)),
             train_info['weight_loss'],
             label='weight_loss',
             linewidth=3,
             color='b')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Change Gragh')
    plt.legend()
    plt.grid()
    plt.savefig('./h{}--lr{} loss.png'.format(hidden_unit, learning_rate))
    plt.show()

    plt.plot(list(range(1,
                        len(train_info['acc']) + 1)),
             train_info['acc'],
             label='acc',
             linewidth=1.5,
             color='g')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy Change Gragh')
    plt.legend()
    plt.grid()
    plt.savefig('./h{}-lr{} acc.png'.format(hidden_unit, learning_rate))
    plt.show()


class tanh(object):
    def __init__(self, ):
        self.data = None
        self.forward = None

    def calc(self, data):
        self.data = data
        self.forward = np.zeros_like(self.data, dtype=np.float64)
        self.forward = np.tanh(self.data)
        return self.forward

    def diff(self, data):
        self.data = data
        self.forward = np.zeros_like(self.data, dtype=np.float64)
        return 1 - np.tanh(self.data)**2


class softmax(object):
    def __init__(self, ):
        self.data = None
        self.forward = None

    def calc(self, data):
        self.data = data
        self.forward = np.zeros_like(self.data, dtype=np.float64)
        for k in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.forward[k][j][0] = np.exp(self.data[k][j][0]) / np.sum(
                    np.exp(self.data[k]))
        return self.forward

    def diff(self, data):
        self.data = data
        #print(data)
        self.forward = np.zeros_like(self.data, dtype=np.float64)
        for k in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                self.forward[k][j][0] = np.exp(self.data[k][j][0]) / np.sum(
                    np.exp(self.data[k]))
        return self.forward * (1 - self.forward)


class sigmoid(object):
    def __init__(self):
        self.data = None
        self.forward = None

    def calc(self, data):
        self.data = data
        self.forward = .5 * (1 + np.tanh(.5 * self.data))
        return self.forward

    def diff(self, data):
        self.data = data
        return .5 * (1 +
                     np.tanh(.5 * self.data)) * (1 - .5 *
                                                 (1 + np.tanh(.5 * self.data)))


class relu(object):
    def __init__(self):
        self.data = None
        self.forward = None

    def calc(self, data):
        self.data = data
        self.forward = self.data.clip(min=0)
        return self.forward

    def diff(self, data):
        self.data = data
        return self.data >= 0


class node_weight(object):
    def __init__(self, input_dim, output_dim, activate_func, batch_size,
                 learning_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activate_func = activate_func
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weights = 0.1 * np.random.randn(self.input_dim + 1,
                                             self.output_dim)
        self.input = None
        self.output = None
        self.loss = np.array([0.] * self.output_dim * self.batch_size).reshape(
            self.batch_size, self.output_dim, -1)
        self.delta = np.zeros((self.input_dim + 1, self.output_dim),
                              dtype=np.float64)

    def forward(self, input):
        self.input = input
        bias = np.array(self.batch_size * [1])
        self.input = np.hstack(
            [bias.reshape(self.batch_size, 1, -1), self.input])
        #print(self.weights.T.shape)
        #print(self.input.transpose(1, 2, 0).reshape([self.batch_size, -1]).shape)
        self.output = np.matmul(
            self.weights.T,
            self.input.transpose(1, 2, 0).reshape([-1, self.batch_size]))
        self.output = self.output[:, np.newaxis, :].transpose(2, 0, 1)
        self.output = self.activate_func.calc(self.output)
        return self.output

    def get_delta(self, ):
        diff = self.activate_func.diff(self.output)
        for j in range(self.output_dim):
            for i in range(self.input_dim + 1):
                for k in range(self.batch_size):
                    self.delta[i][j] += diff[k][j][0] * self.loss[k][j][
                        0] * self.input[k][i][0]
        self.loss = np.array([0.] * self.output_dim * self.batch_size).reshape(
            self.batch_size, self.output_dim, -1)

    def get_loss(self, back_layer):
        for k in range(self.batch_size):
            for i in range(self.output_dim):
                for j in range(back_layer.output_dim):
                    self.loss[k][i][0] += back_layer.weights[i][
                        j] * back_layer.loss[k][j][0]

    def backward(self):
        self.weights += self.delta * self.learning_rate
        self.delta = np.zeros([self.input_dim + 1, self.output_dim],
                              dtype=np.float64)


class nn(object):
    def __init__(self,
                 shape=[3, 16, 3],
                 activate_funcs=[sigmoid(), tanh()],
                 batch_size=4,
                 learning_rate=1e-2):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.net = []
        self.info = {'loss': [], 'weight_loss': [], 'acc': []}
        for i in range(1, len(shape)):
            self.net.append(
                node_weight(shape[i - 1], shape[i], activate_funcs[i - 1],
                            self.batch_size, self.learning_rate))

    def fit(self, x):
        for i in range(len(self.net)):
            x = self.net[i].forward(x)
        return x

    def eval(self, x, y):
        tmp_x = np.copy(x)
        for net in self.net:
            bias = np.array(len(tmp_x) * [1])
            length = len(tmp_x)
            tmp_x = np.hstack([bias.reshape(len(x), 1, -1), tmp_x])
            #print(self.weights.T.shape)
            #print(self.input.transpose(1, 2, 0).reshape([self.batch_size, -1]).shape)
            tmp_x = np.matmul(net.weights.T,
                              tmp_x.transpose(1, 2, 0).reshape([-1, length]))
            tmp_x = tmp_x[:, np.newaxis, :].transpose(2, 0, 1)
            tmp_x = net.activate_func.calc(tmp_x)
        count = 0
        for i in range(len(tmp_x)):
            if np.argmax(tmp_x[i]) == np.argmax(y[i]):
                count += 1
        return count / len(x)

    def backward(self, x, y):
        z = self.fit(x)
        loss = (y - z)
        self.net[-1].loss = loss
        for i in reversed(range(0, len(self.net) - 1)):
            self.net[i].get_loss(self.net[i + 1])
        for i in reversed(range(0, len(self.net))):
            self.net[i].get_delta()
            self.net[i].backward()
        return np.mean(0.5 * loss * loss) * self.net[-1].output_dim

    def train(self, x, y, max_epoch, print_step):
        weight_loss = None
        for i in range(max_epoch):
            batch_index = random.sample(range(len(x)), self.batch_size)
            batch_x = x[batch_index]
            batch_y = y[batch_index]
            loss = self.backward(batch_x, batch_y)
            if not weight_loss:
                weight_loss = loss
            else:
                weight_loss = (1 - (50 / max_epoch)) * weight_loss + (
                    50 / max_epoch) * loss
            acc = self.eval(x, y)
            if (i + 1) % print_step == 0 or i == 0:
                print('epoch:{} loss:{:.4f} weight_loss:{:.4f} acc:{:.2f}'.
                      format(i + 1, loss, weight_loss, acc))
            self.info['loss'].append(loss)
            self.info['acc'].append(acc)
            self.info['weight_loss'].append(weight_loss)


if __name__ == '__main__':
    label_1 = np.array(
        [[1.58, 2.32, -5.8], [0.67, 1.58, -4.78], [1.04, 1.01, -3.63],
         [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
         [1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [0.45, 1.33, -4.38],
         [-0.76, 0.84, -1.96]],
        dtype=np.float64)
    label_1 = label_1[:, :, np.newaxis]

    label_2 = np.array(
        [[0.21, 0.03, -2.21], [0.37, 0.28, -1.8], [0.18, 1.22, 0.16],
         [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
         [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [0.44, 1.31, -0.14],
         [0.46, 1.49, 0.68]],
        dtype=np.float64)
    label_2 = label_2[:, :, np.newaxis]

    label_3 = np.array(
        [[-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [1.55, 0.99, 2.69],
         [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
         [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [0.25, 0.68, -0.99],
         [0.66, -0.45, 0.08]],
        dtype=np.float64)
    label_3 = label_3[:, :, np.newaxis]

    x_data = np.vstack([label_1, label_2, label_3])
    y_data = np.vstack([
        np.array([[[1], [0], [0]]] * 10),
        np.array([[[0], [1], [0]]] * 10),
        np.array([[[0], [0], [1]]] * 10)
    ])

    HIDDEN_UNIT = 128
    BATCH_SIZE = 30
    LEARNING_RATE = 0.01
    model = nn(shape=[3, HIDDEN_UNIT, 3],
               activate_funcs=[relu(), sigmoid()],
               batch_size=BATCH_SIZE,
               learning_rate=LEARNING_RATE)
    model.train(x_data, y_data, 5000, 1)
    train_info = model.info
    plot_info(train_info, HIDDEN_UNIT, LEARNING_RATE)