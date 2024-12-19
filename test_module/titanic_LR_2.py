import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

#%%
# Function to perform stochastic gradient descent
def stochastic_gd(x, y, l_rate, iterations):
    start_time = time.time()  # Start time of the function
    iteration_times = []  # To store time taken for each iteration

    # Reshape and normalize x and y
    x = np.array(x).reshape(-1, 1)
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    y = np.array(y).reshape(-1, 1)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x = scaler_x.fit_transform(x)
    y = scaler_y.fit_transform(y)

    W = np.zeros((x.shape[1], 1))  # Initialize weights
    residuals = []  # To store residuals

    for i in range(iterations):
        iter_start_time = time.time()  # Start time of the iteration

        # Randomly select one data point
        idx = np.random.randint(0, x.shape[0])
        x_i = x[idx, :].reshape(1, -1)
        y_i = y[idx, :].reshape(1, -1)

        # Prediction and gradient calculation based on one data point
        y_pred = np.dot(x_i, W)
        dW = np.dot(x_i.T, (y_pred - y_i))

        # Update weights
        W -= l_rate * dW

        # Calculate and store residual
        y_pred_all = np.dot(x, W)
        residual = np.sum((y_pred_all - y) ** 2)
        residuals.append(residual)

        # Store iteration time
        iteration_time = time.time() - iter_start_time
        iteration_times.append(iteration_time)

    total_time = time.time() - start_time  # Total time of the function
    average_time_per_iteration = total_time / iterations  # Average time per iteration

    # Plotting
    plt.plot(range(iterations), residuals, label='Residuals')
    plt.xlabel('Iteration')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Iterations')
    plt.legend()
    plt.figtext(0.7, 0.2, f'Average Time per Iteration: {average_time_per_iteration:.8f} seconds', ha="left")
    plt.figtext(0.7, 0.005, f'Total Time: {total_time:.8f} seconds', ha="left")
    plt.show()

    return W

# Function to perform batch gradient descent
def batch_gd(x, y, l_rate, iterations):
    start_time = time.time()  # Record the start time of the function
    iteration_times = []  # Initialize a list to store iteration times

    x = np.array(x).reshape(-1, 1)  # Reshape x into a column vector
    ones = np.ones(shape=(x.shape[0], 1))
    x = np.concatenate((ones, x), axis=1)
    y = np.array(y).reshape(-1, 1)

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    y = scaler.fit_transform(y)

    W = np.zeros(shape=(x.shape[1], 1))
    residuals = []  # Initialize an empty list for residuals

    for i in range(iterations):
        iter_start_time = time.time()  # Start time of the iteration

        y_pred = x @ W
        residual = np.sum((y_pred - y) ** 2)  # Calculate the sum of squares of the residuals
        residuals.append(residual)  # Append the residual to the list

        dW = np.dot(x.T, y_pred - y)
        W = W - l_rate * dW

        iteration_time = time.time() - iter_start_time  # Time taken for the iteration
        iteration_times.append(iteration_time)  # Append iteration time to the list

    total_time = time.time() - start_time  # Total time of the function
    average_time_per_iteration = total_time / iterations  # Average time per iteration

    # Plotting
    plt.plot(range(iterations), residuals, label='Residuals')
    plt.xlabel('Iteration')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Iterations')
    plt.legend()
    plt.figtext(0.7, 0.2, f'Average Time per Iteration: {average_time_per_iteration:.8f} seconds', ha="left")
    plt.figtext(0.7, 0.005, f'Total Time: {total_time:.8f} seconds', ha="left")
    plt.show()

#%%
# Load and prepare data
df = pd.read_csv("../data/raw/train.csv")
df.info(verbose=True)

# Perform gradient descent
batch_gd(df['LotArea'], df['SalePrice'], 0.01, 1000)
stochastic_gd(df['LotArea'], df['SalePrice'], 0.01, 100000)
# %%
