# I just saw the formulas here: https://sanjayasubedi.com.np/deeplearning/stochastic-gradient-descent-from-scratch/
# And wrote my own functions by first writing in paper


import numpy as np
import matplotlib.pyplot as plt


class gradient:
    
    def __init__(self, l_rate = 0.01, n_iter = 1000):
        self.l_rate = l_rate
        self.n_iter = n_iter
        
        
    def calculate(self, weights, bias, x):
        y_pred = np.dot(weights, x) + bias
        return y_pred
    
    def descent(self, weights, bias, x, y_true, y_pred):
        '''
        Basically torch.backward() does this only.
        It calculates the gradients
        '''
        
        weights_diff = -2 * (y_true - y_pred) * x * self.l_rate
        bias_diff = -2 * (y_true - y_pred) * self.l_rate
        
        return weights - weights_diff, bias - bias_diff
        
        
    
    def fit(self, x, y):
        
        #get the number of features so we can generate weight matrix
        weight_shape = len(x[0])
        
        #generate a matrix of this shape
        weights = np.random.rand(weight_shape)
        bias = np.random.rand()
        
        for _ in range(self.n_iter):
            
            for i in range(len(x)):
                y_pred = self.calculate(weights, bias, x[i])
                
                weights, bias = self.descent(weights, bias, x[i], y[i], y_pred)
                
        return weights, bias
                
                
    def predict(self, x, weights, bias):
        return self.calculate( weights, bias, x)
    
    

if __name__ == "__main__":
    
    #From GPT

    # Create a larger dataset for testing
    # y = 3x + 2 (expected linear relationship)
    np.random.seed(0)  # For reproducibility
    x_train = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
    y_train = 3 * x_train.flatten() + 2 + np.random.randn(100)  # y = 3x + 2 with some noise

    # Instantiate the Gradient class
    model = gradient(l_rate=0.001, n_iter=2000)

    # Train the model
    weights, bias = model.fit(x_train, y_train)

    # Make predictions using the trained modely_pred = model.predict(x_train, weights, bias) 
    y_pred = [model.predict(x, weights, bias) for x in x_train]

    # Output the learned weights and bias
    print(f"Weights: {weights}")
    print(f"Bias: {bias}")

    # Compare the predictions to the actual values
    print(f"Predictions (first 10): {y_pred[:10]}")
    print(f"Actual (first 10): {y_train[:10]}")

    # Plot the results to visually confirm
    plt.scatter(x_train, y_train, color='blue', label='Actual data', alpha=0.5)
    plt.plot(x_train, y_pred, color='red', label='Fitted line')
    plt.title('Gradient Descent Linear Regression (Larger Dataset)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
            