import numpy as np
import os

class Layer:
    def __init__(self,input_size,output_size,activation_func="sigmoid"):
        self.activation_func = activation_func
        self.weights = np.round(np.random.randn(output_size,input_size))
        self.biases = np.zeros((output_size,1))
        self.weighted_inputs = np.zeros((output_size,1))
        self.activations = np.zeros((output_size,1))
        
        self.gradientW = np.zeros((output_size,input_size))
        self.gradientB = np.zeros((output_size,1))

class Activation:
    def calculate(self,func,x):
        if func == "relu":
            return self.relu(x)
        elif func == "sigmoid":
            return self.sigmoid(x)
        elif func == "softmax":
            return self.softmax(x)

    def relu(self,x):
        return np.maximum(0,x)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def softmax(self,x):
        exp_values = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_values / np.sum(exp_values, axis=0, keepdims=True)

    def derivative(self,func,x):
        if func == "relu":
            return np.where(x > 0, 1, 0)
        elif func == "sigmoid":
            return self.sigmoid(x)*(1-self.sigmoid(x))
        
class Network:
    def __init__(self,layers_sizes) -> None:
        self.layers = [Layer(layers_sizes[i], layers_sizes[i+1]) for i in range(len(layers_sizes)-1)]
        self.layers[-1].activation_func = "softmax"
        self.actv = Activation()
        
    def feed_forward(self,a):
        for layer in self.layers:
            layer.weighted_inputs = np.dot(layer.weights,a) + layer.biases
            a = self.actv.calculate(layer.activation_func,layer.weighted_inputs)
            layer.activations = a
        return a

    def mse_loss(self, actual, predicted):
        squared_error = (predicted - actual) ** 2
        sum_squared_error = np.sum(squared_error, axis=0)
        mean_squared_error = np.mean(sum_squared_error)
        
        return mean_squared_error
    
    def cross_entropy_loss(self, actual, predicted):
        epsilon = 1e-12  # To prevent log(0)
        predicted = np.clip(predicted, epsilon, 1. - epsilon)
        
        return np.sum(-np.sum(actual * np.log(predicted),axis=0)) / actual.shape[1]

    def mse_gradient(self,actual,predicted):
        return 2*(predicted - actual) / len(actual[1])

    def backpropagate(self, actual, a,lr):
        predicted = self.feed_forward(a)

        delta = predicted - actual  # gradient for softmax + cross-entropy

        for i, layer in reversed(list(enumerate(self.layers))):

            if i > 0:
                prev_activations = self.layers[i-1].activations
            else:
                prev_activations = a  # Input activations

            # Only apply the activation function's derivative for hidden layers, not the output
            if i < len(self.layers) - 1:
                delta = delta * self.actv.derivative(layer.activation_func, layer.weighted_inputs)

            layer.gradientW = np.dot(delta, prev_activations.T)
            layer.gradientB = np.sum(delta, axis=1, keepdims=True)

            # Update weights and biases
            layer.weights -= lr * layer.gradientW
            layer.biases -= lr * layer.gradientB

            delta = np.dot(layer.weights.T, delta)

    def train(self,X,Y,learnRate=0.001,epochs=1000,batch_size=100,show_training_progress=False,show_training_progress_rate=100):
        for epoch in range(epochs):
            shuffled_X,shuffled_Y = self.shuffle_data(X,Y)
            for i in range(0,X.shape[0],batch_size):
                X_batch = shuffled_X[:,i:i+batch_size]
                Y_batch = shuffled_Y[:,i:i+batch_size]

                self.backpropagate(Y_batch, X_batch,learnRate)

            if show_training_progress:
                if epoch % show_training_progress_rate == 0:
                    loss = self.cross_entropy_loss(Y, self.feed_forward(X))
                    print(f"Epoch {epoch}, Loss: {loss}")

    def save_model(self, file_prefix='model'):
        os.mkdir("model")
        os.chdir("model")
        for i, layer in enumerate(self.layers):
            np.save(f'{file_prefix}_weights_{i}.npy', layer.weights)
            np.save(f'{file_prefix}_biases_{i}.npy', layer.biases)
        os.chdir("..")
    
    def load_model(self, file_prefix='model'):
        for i, layer in enumerate(self.layers):
            layer.weights = np.load(f'{file_prefix}_weights_{i}.npy')
            layer.biases = np.load(f'{file_prefix}_biases_{i}.npy')

    def shuffle_data(self, X, Y):
        # X: features (input data)
        # Y: labels (output data)
        
        # Generate a permutation of indices
        permutation = np.random.permutation(X.shape[1])  # Assuming data is in column-major form (features, examples)
        
        # Shuffle both X and Y using the permutation
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        return X_shuffled, Y_shuffled
