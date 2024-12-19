# Necessary imports
import numpy as np
import matplotlib.pyplot as plt


'''This script perform a locally weighted regression on the dataset Area vs Price
Para no ser buen mecanismo para unos datos con outliers
'''
#%%
# function to perform locally weighted linear regression
def local_weighted_regression(x0, X, Y, tau):
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]

    # fit model: normal equations with kernel
    xw = X.T * weights_calculate(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    # "@" is used to
    # predict value
    return x0 @ theta

# function to perform weight calculation
def weights_calculate(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau **2)))

# plot locally weighted regression for different bandwidth values
def plot_lwr(tau):
    # prediction
    domain = np.linspace(0, 1, num=300)
    prediction = [local_weighted_regression(x0, X, Y, tau) for x0 in domain]

    plt.figure()
    plt.title('lot area vs price')
    plt.scatter(X, Y, alpha=.3)
    plt.xlabel('Lot Area')
    plt.ylabel('price')
    plt.plot(domain, prediction, linewidth=2, color='red')
    plt.legend()
    plt.show()

def prepare_dataset(df):
    # Extracting features and target variables
    X = np.array(df['LotArea']).reshape(-1, 1)
    Y = np.array(df['SalePrice']).reshape(-1, 1)

    # Initializing scalers
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Scaling features and target
    X_scaled = scaler_x.fit_transform(X)
    Y_scaled = scaler_y.fit_transform(Y)

    return X_scaled, Y_scaled, scaler_x, scaler_y

    return None
#%%
# Load and prepare data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd  # Missing import added
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("../data/raw/train.csv")
df.info(verbose=True)

# generate dataset

X, Y, scaler_x, scaler_y = prepare_dataset(df)

# show the plots for different values of Tau
for i in range(0,10):
    plot_lwr(10**(-i))
# %%
