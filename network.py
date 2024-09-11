import numpy as np

class Layer:
    def __init__(self,input_size,output_size,activation_func="relu"):
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

    def derivative(self,func,x,y=None):
        if func == "relu":
            return np.where(x > 0, 1, 0)
        elif func == "sigmoid":
            return self.sigmoid(x)*(1-self.sigmoid(x))
        elif func == "softmax":
            return x-y
        
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

    def loss(self, actual, predicted):
        squared_error = (predicted - actual) ** 2
        sum_squared_error = np.sum(squared_error, axis=0)
        mean_squared_error = np.mean(sum_squared_error)
        
        return mean_squared_error

    def mse_gradient(self,actual,predicted,n):
        return 2*(predicted - actual) / n

    def backpropagate(self, actual, a, n):
        lr = 0.01
        epochs = 3000
    
        for e in range(epochs):
            predicted = self.feed_forward(a)
            delta = self.mse_gradient(actual, predicted, n)
            
            for i, layer in reversed(list(enumerate(self.layers))):

                if i > 0:
                    prev_activations = self.layers[i-1].activations
                else:
                    prev_activations = a  # Input activations

                delta = delta * self.actv.derivative(layer.activation_func,layer.activations,actual)

                layer.gradientW = np.dot(delta,prev_activations.T)
                layer.gradientB = np.sum(delta,axis=1,keepdims=True)


                layer.weights -= lr * layer.gradientW
                layer.biases -= lr * layer.gradientB
                
                delta = np.dot(layer.weights.T,delta)
            
            loss = self.loss(actual, predicted)
            if e % 1000 == 0:
                print(f"epoch {e} | loss = {loss}")   
     
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