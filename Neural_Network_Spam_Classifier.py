import numpy as np
import time
import matplotlib.pyplot as plt


##########################################################################
######################### SpamClassifier Class ###########################
##########################################################################

class SpamClassifier:

    ######## Constructor
    def __init__(self, input_size=None, output_size=None):

        # Initializing weights an bias (Random Method)

        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    ######## prediction
    # data_inputs_row are all features in a row
    def prediction(self, data_inputs_row, training=False, weights=None, bias=None, A_func=None):

        if training is False:
            self.weigths = weights
            self.bias = bias

            # print(self.weights.shape, data_inputs_row.shape, self.bias.shape)
        # print(data_inputs_row.shape, self.weights.shape, self.bias.shape)
        Z = np.dot(self.weights, data_inputs_row) + self.bias
        # print(f"Z: {Z.shape}")

        # print(A_func)

        if A_func == 0:

            Act_func = self._sigmoid(Z)
            prediction = Act_func
            # print(f"prediction : {prediction.shape}")

        elif A_func == 1:

            Act_func = self.relu(Z)
            prediction = Act_func
            # print(f"prediction : {prediction.shape}")

        return Z, prediction

    ######## sigmoid (Activasion Function)
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    ######## _sigmoid_deriv (Derivative of Activasion Function)
    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_deriv(self, x):
        return np.array(x > 0, dtype=np.float32)

    ######## _update_parameters
    def _update_parameters(self, derror_dbias, derror_dweights, learning_rate):
        self.bias = self.bias - (learning_rate * derror_dbias)
        # print(self.weights.shape, derror_dweights.shape)
        self.weights = self.weights - (learning_rate * derror_dweights)

##########################################################################
############################ Classify Class ##############################
##########################################################################

class Classify(SpamClassifier):

    ######## Constructor
    def __init__(self, object_layers, weight_bias_pre_values):

        self.object_layers = object_layers
        self.weight_bias_pre_values = weight_bias_pre_values

    ######## predict
    def predict(self, data_inputs_row):

        prediction_matrix = np.array([])
        z_matrix = np.array([])
        real_prediction_matrix = np.array([])

        for row in range(len(data_inputs_row)):

            # print(data_inputs_row[row].shape)
            data_inputs = data_inputs_row[row][:, np.newaxis]
            # print(data_inputs.shape)

            output = [(None, data_inputs)]  # Saving (Sum (Z), Activation Function

            for l, layer in enumerate(self.object_layers):
                # print(l)
                if l == (len(self.object_layers) - 1):
                    A_Func = 0

                else:
                    A_Func = 1
                # print(A_Func)
                Z, prediction = layer.prediction(output[-1][1], False, self.weight_bias_pre_values[l][0],
                                                 self.weight_bias_pre_values[l][1], A_Func)

                # print(prediction)
                output.append((Z, prediction))

            activ = output[-1][1]
            z_sum = output[-1][0]

            real_prediction_matrix = np.append(real_prediction_matrix, activ)
            z_matrix = np.append(z_matrix, z_sum)

            if activ >= 0.5:
                activ = 1
            else:
                activ = 0

            prediction_matrix = np.append(prediction_matrix, activ)
        #print(prediction_matrix)
        #print(z_matrix)
        #print(real_prediction_matrix)
        return prediction_matrix

##########################################################################
############################## Functions #################################
##########################################################################


######## create_NN
# Creating the Neural Network layers
def create_NN(Neural_Network_topology):
    NN_vector = []

    for l, layer in enumerate(Neural_Network_topology[:-1]):
        NN_vector.append(SpamClassifier(Neural_Network_topology[l], Neural_Network_topology[l + 1]))

    return NN_vector


######## train
# Training the Neural network
def train(Neural_Network_layers, data_inputs, data_outputs, _lambda, learning_rate):
    fordward_output = [(None, data_inputs)]  # Saving (Sum (Z), Activation Function

    # Forward
    for l, layer in enumerate(Neural_Network_layers):
        # print(f"l: {l}")
        if l == (len(Neural_Network_layers) - 1):  # Last Layer
            A_func = 0

        else:
            A_func = 1

        Z, prediction = layer.prediction(fordward_output[-1][1], True, A_func=A_func)
        fordward_output.append((Z, prediction))

    m = data_outputs.shape[1]

    for l in reversed(range(0, len(Neural_Network_layers))):

        if l == (len(Neural_Network_layers) - 1):  # Last Layer

            # print(fordward_output[l+1][1].shape, data_outputs.shape)
            dZ = (fordward_output[l + 1][1] - data_outputs)
            # print(f"dZ.shape: {fordward_output[l+1][1].shape}")

        else:  # Other Layers

            # print(Neural_Network_layers[l+1].weights.T.shape, dZ.shape)
            dA = np.dot(Neural_Network_layers[l + 1].weights.T, dZ)
            # print(dA.shape)
            dZ = np.multiply(dA, np.int64(fordward_output[l + 1][1] > 0))
            # print(dZ.shape)

        # print(dZ.shape, fordward_output[l][1].T.shape, Neural_Network_layers[l].weights.shape)
        dW = (1 / m) * np.dot(dZ, fordward_output[l][1].T) + ((_lambda / m) * Neural_Network_layers[l].weights)
        # print(dW.shape)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        # print(db.shape)

        Neural_Network_layers[l]._update_parameters(db, dW, learning_rate)

        np.savetxt(f'Weight_Values_{l + 1}.csv', Neural_Network_layers[l].weights, delimiter=",")
        np.savetxt(f'Bias_Values_{l + 1}.csv', Neural_Network_layers[l].bias, delimiter=",")

    return fordward_output[-1][1]


##########################################################################
################################ Main ####################################
##########################################################################


if __name__ == '__main__':

    # np.random.seed(1)

    # Organizing Data
    training_spam = np.loadtxt(open("data/training_spam.csv"), delimiter=",")

    data_outputs = training_spam[:, 0]  # 1 = spam, 0 = ham
    data_outputs = data_outputs[:, np.newaxis].T
    # print(data_outputs.shape)
    # data_outputs = data_outputs[:, np.newaxis] # Converting from (1000,) to (1000, 1)
    data_inputs = training_spam[:, 1:].T  # All 54 features values over 1000 samples for training mode
    n_features, n_samples = data_inputs.shape  # Number of samples (1000), Number of features (54)

    # Parameters and Hyperparameters to modify
    # Training_Mode is true, a new ANN will be created, if FALSE the prediction mode (predict method) is ON
    TRAINING_MODE = False
    Neural_Network_topology = [n_features, 8, 2, 1]  # Layers and neurons per layer (Lenght -1 = # of layers, elements are # of Neurons)
    Neural_Network_Layers = create_NN(Neural_Network_topology)  # Creating the ANN
    iterations = 200000
    _lambda = 0.01
    learning_rate = 0.001

    # Training
    if TRAINING_MODE:

        total_time = 0
        start_time = time.process_time()
        cost = []  # List to save values of cost function

        for i in range(iterations):

            predictions = train(Neural_Network_Layers, data_inputs, data_outputs, _lambda, learning_rate)

            # Adding weight matrix to be implemened in L2 regularitazion
            weight_data_1 = np.loadtxt(open("Weight_Values_1.csv"), delimiter=",")
            weight_data_2 = np.loadtxt(open("Weight_Values_2.csv"), delimiter=",")
            weight_data_3 = np.loadtxt(open("Weight_Values_3.csv"), delimiter=",")

            if i % 1 == 0:
                # Cost Function
                cost_1 = - (1 / n_samples) * np.sum(
                    data_outputs * np.log(predictions) + (1 - data_outputs) * (np.log(1 - predictions)))
                L2_regularization_cost = (_lambda / (2 * n_samples)) * (
                            np.sum(np.square(weight_data_1)) + np.sum(np.square(weight_data_2))
                            + np.sum(np.square(weight_data_3)))
                total_cost = cost_1 + L2_regularization_cost
                cost.append(total_cost)

                PLOTTING = True  # True to plot

        if PLOTTING:
            # Plotting Cost Function
            plt.plot(cost)
            plt.xlabel("Iterations")
            plt.ylabel("Error for all training instances")
            plt.title('Minimization Curve of Cost Function')
            plt.savefig(f"Minimization Curve of Cost Function lambda={_lambda} & lr={learning_rate}.png")

        total_time += time.process_time() - start_time
        print(f"\t\tTime for {iterations} iterations: {total_time:.4} seconds")

    # Prediction Mode
    else:

        # Loading the CSV Files based on last training mode
        weight_data_1 = np.loadtxt(open("Weight_Values_1.csv"), delimiter=",")
        weight_data_2 = np.loadtxt(open("Weight_Values_2.csv"), delimiter=",")
        weight_data_3 = np.loadtxt(open("Weight_Values_3.csv"), delimiter=",")
        bias_data_1 = np.loadtxt(open("bias_Values_1.csv"), delimiter=",")
        bias_data_2 = np.loadtxt(open("bias_Values_2.csv"), delimiter=",")
        bias_data_3 = np.loadtxt(open("bias_Values_3.csv"), delimiter=",")

        bias_data_1 = bias_data_1[:, np.newaxis]
        bias_data_2 = bias_data_2[:, np.newaxis]
        weight_data_2 = weight_data_2[:, np.newaxis]

        weight_bias_pre_values = np.array([[weight_data_1, bias_data_1],
                                           [weight_data_2, bias_data_2],
                                           [weight_data_3, bias_data_3]], dtype=object)

        classifier = Classify(Neural_Network_Layers, weight_bias_pre_values)
        # predictions = classifier.predict(test_data)
