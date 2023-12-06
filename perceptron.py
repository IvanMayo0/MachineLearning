import random
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_size):
        # Initialize weights with random values between -1 and 1
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)
        self.learning_rate = 0.1

    def activation(self, x):
        # Define a simple step function as the activation function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Compute the weighted sum of inputs
        weighted_sum = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.activation(weighted_sum)

    def train(self, inputs, target):
        # Compute the prediction
        prediction = self.predict(inputs)

        # Update the weights and bias
        error = target - prediction
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        self.bias += self.learning_rate * error




        ########################333

if __name__ == '__main__':
    # Create a Perceptron with 2 input features
    perceptron = Perceptron(2)

    # Generate random training data for the first disorganized dataset
    training_data_1 = []
    for _ in range(100):
        inputs = [random.uniform(0, 1), random.uniform(0, 1)]
        target = 1 if inputs[0] + inputs[1] > 1 else 0
        training_data_1.append((inputs, target))

    # Generate organized training data for the second dataset
    training_data_2 = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1)
    ]

    # Lists to store the points for plotting
    x0_1 = []
    y0_1 = []
    x1_1 = []
    y1_1 = []
    
    x0_2 = []
    y0_2 = []
    x1_2 = []
    y1_2 = []

    # Train the Perceptron for 100 epochs on the first dataset
    for epoch in range(100):
        for inputs, target in training_data_1:
            perceptron.train(inputs, target)

    # Collect points for the first dataset
    for inputs, target in training_data_1:
        if target == 0:
            x0_1.append(inputs[0])
            y0_1.append(inputs[1])
        else:
            x1_1.append(inputs[0])
            y1_1.append(inputs[1])

    # Train the Perceptron for 100 epochs on the second dataset
    for epoch in range(100):
        for inputs, target in training_data_2:
            perceptron.train(inputs, target)

    # Collect points for the second dataset
    for inputs, target in training_data_2:
        if target == 0:
            x0_2.append(inputs[0])
            y0_2.append(inputs[1])
        else:
            x1_2.append(inputs[0])
            y1_2.append(inputs[1])

    # Plot the results for both datasets
    plt.figure(figsize=(12, 5))
    
    # Plot for the first disorganized dataset
    plt.subplot(1, 2, 1)
    plt.scatter(x0_1, y0_1, color='red', label='Class 0')
    plt.scatter(x1_1, y1_1, color='blue', label='Class 1')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend()
    plt.title('Perceptron Classification (Disorganized Dataset)')
    
    # Plot for the second organized dataset
    plt.subplot(1, 2, 2)
    plt.scatter(x0_2, y0_2, color='red', label='Class 0')
    plt.scatter(x1_2, y1_2, color='blue', label='Class 1')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.legend()
    plt.title('Perceptron Classification (Organized Dataset)')
    
    plt.tight_layout()
    plt.show()