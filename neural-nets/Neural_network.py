from matplotlib import pyplot as plt
import numpy as np
import logging
import sys
import os

from keras.datasets import mnist
from keras.utils import np_utils
from sklearn import metrics
import tensorflow as tf


class NeuralNetwork:
    def __init__(self, LEARNING_RATE=0.1, NUMBER_OF_ITERATIONS=35, NUMBER_OF_TRAINING_UNITS = 1000, NUMBER_OF_TEST_UNITS = 100, HIDDEN_LAYER_NEURONS = 50):
        self.LEARNING_RATE = LEARNING_RATE
        self.NUMBER_OF_ITERATIONS = NUMBER_OF_ITERATIONS
        self.NUMBER_OF_TRAINING_UNITS = NUMBER_OF_TRAINING_UNITS
        self.NUMBER_OF_TEST_UNITS = NUMBER_OF_TEST_UNITS
        self.HIDDEN_LAYER_NEURONS = HIDDEN_LAYER_NEURONS

    def init_weight_vector(input_count, neuron_count):
        '''Initialises a vector of given dimensions with random weights'''
        weight_vector = []
        for _ in range(input_count):
            random_weights = []
            for _ in range(neuron_count):
                random_weights.append(np.random.standard_normal())
            weight_vector.append(random_weights)
        return np.array(weight_vector)

    def activation_function(input_vector):
        '''Returns a sigmoid / logistic function value in range of 0 to 1 based on given input vector'''
        sigmoid = 1 / (1 + np.exp(-input_vector))
        return sigmoid

    def calculate_mse(observed_values, predicted_values):
        '''Returns a Mean Square Error value based on given actual and predicted values'''
        mse = np.square(predicted_values - observed_values)
        mse = np.sum(mse) / len(predicted_values)
        return mse

    def forward_propagation(_input, weight_vector1, weight_vector2):
        '''
            Wrapper for forward propagation mechanism in neural networks
            It propagates through a hidden layer and output layer (marked respectively as layer 2 and layer 3)
            It takes a vector of input values and uses it to calculate hidden layer output
            It takes a vector of hidden layer output values and uses it to calculate final layer output
            Calculations are based on given two weight vectors (each one for one layer)
            Returns a final output value from the network as vector

            '''
        layer2_output = activation_function(np.dot(_input, weight_vector1))
        layer3_output = activation_function(np.dot(layer2_output, weight_vector2))
        return layer3_output

    def backward_propagation(_input, correct_output, weight_vector1, weight_vector2, alpha):
            '''
            Wrapper for backward propagation mechanism in neural networks
            It propagates through a hidden layer and output layer (just like in forward propagation mechanism)
            It uses the layer outputs, weight vectors and known answer to calculate errors in each layer
            The function then calculates gradient and uses it for vector adjustment
            The adjustment rate can be controlled by modifying alpha parameter
            It outputs the modified weight vectors

            '''
        layer2_output = activation_function(np.dot(_input, weight_vector1))
        layer3_output = activation_function(np.dot(layer2_output, weight_vector2))
        layer3_error = layer3_output - correct_output
        layer2_error = np.multiply(np.transpose(np.dot(weight_vector2, np.transpose(layer3_error))),
                                    np.multiply(layer2_output, 1 - np.array(layer2_output)))
        gradient_vector1 = np.dot(np.transpose(_input), layer2_error)
        gradient_vector2 = np.dot(np.transpose(layer2_output), layer3_error)
        weight_vector1 -= alpha * gradient_vector1
        weight_vector2 -= alpha * gradient_vector2
        return weight_vector1, weight_vector2

    def plot_graph(x_values, y_values, title, filename):
        '''Helper function that saves the plot to a file with a given filename and parameters'''
        plt.plot(x_values, y_values)
        plt.title(title)
        plt.savefig(f'plots/{filename}')

    def train_network(inputs, actual, weight_vector1, weight_vector2):
            '''
            The training function for this neural network
            It provides information about the progress via collecting MSE for each iteration
            It iterates a given number of iterations and uses forward propagation and backward propagation
            to calculate the MSE and update network weights
            The information is printed to a file; additionally the MSE graph is plotted
            It takes training set inputs, actual answers and weight vectors
            It doesn't return anything

            '''
        mse = []
        logging.info('Training started')
        for iteration in range(NUMBER_OF_ITERATIONS):
            total_mse = []
            for position in range(len(inputs)):
                output = forward_propagation(inputs[position], weight_vector1, weight_vector2)
                total_mse.append(calculate_mse(actual[position], output))
                weight_vector1, weight_vector2 = backward_propagation(inputs[position], actual[position],
                                                                        weight_vector1, weight_vector2, LEARNING_RATE)
            mse.append(np.sum(total_mse) / len(inputs))
            logging.info(f'Iteration: {iteration}; \tMSE: {np.sum(total_mse) / len(inputs):.5f}')
        logging.info(f'Final MSE: {np.sum(total_mse) / len(inputs):.5f}')
        logging.info('Done training, plotting MSE graph')
        plot_graph(range(1, NUMBER_OF_ITERATIONS + 1), mse, 'Mean Square Error Graph', 'mse')
        logging.info('MSE graph plotted')

    def make_prediction(case, weight_vector1, weight_vector2):
        '''
        The prediction function for this neural network
        It predicts the answer based on given input and two weight vectors
        It looks for the maximum value in predicted answer vector to return the solution
        It returns the solution position (which in this case it is also the human-readable answer)
        and the solution in categorical format

        '''
        prediction = forward_propagation(case, weight_vector1, weight_vector2)
        maximum = prediction[0][0]
        candidate_pos = 0
        for position in range(len(prediction[0])):
            if prediction[0][position] > maximum:
                maximum = prediction[0][position]
                candidate_pos = position
        categorical_candidate = np_utils.to_categorical(candidate_pos, num_classes=10)
        return candidate_pos, categorical_candidate

    # confusion matrix, accuracy, f1-score, precision, recall
    def test_network_performance(x_test, y_test, weight_vector1, weight_vector2):
        '''
        This is a wrapper function to evaluate the trained neural network performance
        It converts the predicted set and known answer set to their respective position values
        in categorical format (which are also human-readable answers) for comparison purposes
        It takes test set as it's input and two weight vector values for conducting forward propagation
        It collects predicted values, which are results of forward propagation
        It compares them to known values and plots Confusion Matrix
        It writes to the log file 4 different scores: accuracy, precision, recall and F1
        It doesn't return anything

        '''
        actual = []
        predictions = []
        logging.info('Testing neural network performance')
        for position in range(len(y_test)):
            actual.append(y_test[position].tolist().index(1.0))
        for position in range(len(x_test)):
            prediction, _ = make_prediction(x_test[position], weight_vector1, weight_vector2)
            predictions.append(prediction)
        logging.info('Made all required predictions')
        logging.info('Plotting Confusion Matrix')
        confusion_matrix = metrics.confusion_matrix(actual, predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        cm_display.plot()
        plt.title('Confusion Matrix')
        plt.savefig('plots/confusion_matrix.png')
        logging.info('Confusion Matrix plotted\n')
        logging.info('Calculating final scores')
        accuracy_score = metrics.accuracy_score(actual, predictions)
        logging.info(f'Accuracy score: {accuracy_score:.5f}')
        precision_score = metrics.precision_score(actual, predictions, average='weighted')
        logging.info(f'Precision score: {precision_score:.5f}')
        recall_score = metrics.recall_score(actual, predictions, average='weighted')
        logging.info(f'Recall score: {recall_score:.5f}')
        f1_score = metrics.f1_score(actual, predictions, average='weighted')
        logging.info(f'F1 score: {f1_score:.5f}')
        logging.info('Done testing the neural network')

    def make_neural_network():
        logging.basicConfig(filename='nn.log',
                            filemode='w',
                            format='%(asctime)s - %(message)s',
                            level=logging.INFO)

        if not os.path.exists("plots"):
            os.makedirs("plots")
        logging.info('Program start')
        logging.info('3-Layer Neural Network Implementation\n')
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
        x_train = x_train.astype('float32')
        x_train /= 255
        y_train = np_utils.to_categorical(y_train)

        x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
        x_test = x_test.astype('float32')
        x_test /= 255
        y_test = np_utils.to_categorical(y_test)

        weight_vector1 = self.init_weight_vector(28 * 28, self.HIDDEN_LAYER_NEURONS)
        weight_vector2 = init_weight_vector(self.HIDDEN_LAYER_NEURONS, 10)

        train_network(x_train[0:self.NUMBER_OF_TRAINING_UNITS], y_train[0:self.NUMBER_OF_TRAINING_UNITS], weight_vector1,
                          weight_vector2)
        test_network_performance(x_test[0:self.NUMBER_OF_TEST_UNITS], y_test[0:self.NUMBER_OF_TEST_UNITS], weight_vector1,
                                     weight_vector2)

        logging.info('Program end')

neural_net = NeuralNetwork()
neural_net.make_neural_network()


