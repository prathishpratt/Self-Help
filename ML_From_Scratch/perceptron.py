## This might help: https://www.youtube.com/watch?v=aOEoxyA4uXU
### But I coded it myself by first writing down the formulas and then coding it


import numpy as np

def activation_func (x):
        return 0 if x<0 else 1


class Perceptron():
    
    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
        self.weights = None
        self.bias = None
        
    
    def fit(self, x, y):
        self.weights = np.zeros(len(x[0]))
        self.bias = 0            #np.random.randn()
        
        for i in range(0,self.n_iter):
            
            for j in range(0,len(x)):
                y_hat = activation_func( np.dot(x[j] , self.weights ) + self.bias )
                
                if y_hat != y[j]:
                    delta = self.learning_rate * (y[j] - y_hat) * x[j]
                    self.weights = self.weights + delta
                    
                    delta_bias = self.learning_rate * (y[j] - y_hat)
                    self.bias = self.bias + delta_bias
                    
        
    def predict(self, x):
        return activation_func( np.dot(x , self.weights ) + self.bias)
    


if __name__ == "__main__":
    
    
    #This from ChatGPT to evaluate 
    
    # Example Data (AND operation)
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    Y = np.array([0, 0, 0, 1])  # Expected output for AND

    # Create the Perceptron object
    model = Perceptron(learning_rate=0.1, n_iter=10)

    # Train the model
    model.fit(X, Y)

    # Test the model
    predictions = [model.predict(i) for i in X]
    print("Predictions:", predictions)
    print("Actual:", Y.tolist())
                    
                    
