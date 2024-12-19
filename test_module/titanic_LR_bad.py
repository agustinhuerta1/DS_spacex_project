import numpy as np
import matplotlib.pyplot as plt

#%%

import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
def remove_outliers(df, column_name, num_std_dev=0.01):
    """
    Removes outliers from a dataframe based on the number of standard deviations from the mean.
    
    Parameters:
    - df: DataFrame from which to remove outliers
    - column_name: Name of the column to check for outliers
    - num_std_dev: Number of standard deviations for defining an outlier
    
    Returns:
    - DataFrame with outliers removed
    """
    mean_value = df[column_name].mean()
    std_dev = df[column_name].std()
    # Calculate the cutoff values
    lower_limit = mean_value - (num_std_dev * std_dev)
    upper_limit = mean_value + (num_std_dev * std_dev)
    # Filter out the outliers
    filtered_df = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]
    return filtered_df

def plot_lotarea_saleprice(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['LotArea'], df['SalePrice'], alpha=0.5)
    plt.title('Lot Area vs Sale Price')
    plt.xlabel('Lot Area')
    plt.ylabel('Sale Price')
    plt.show()

from sklearn.linear_model import LinearRegression

def perform_linear_regression(df):
    # Reshape data for sklearn
    X = df['LotArea'].values.reshape(-1, 1)  # Features
    y = df['SalePrice'].values  # Target variable

    # Create linear regression object
    reg = linear_model.Ridge(alpha=10000)

    # Train the model using the training sets
    reg.fit(X, y)

    # The coefficients
    print('Coefficients: \n', reg.coef_)

    # Plot the original data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['LotArea'], df['SalePrice'], alpha=0.5)
    plt.title('Lot Area vs Sale Price')
    plt.xlabel('Lot Area')
    plt.ylabel('Sale Price')

    # Predict y values for the regression line
    y_pred = reg.predict(X)

    # Plot the regression line
    plt.plot(df['LotArea'], y_pred, color='red', linewidth=2)

    plt.show()

def perform_linear_regression_with_outlier_removal(df):
    # Remove outliers based on 'LotArea' and 'SalePrice'
    df_filtered = remove_outliers(df, 'LotArea')
    df_filtered = remove_outliers(df_filtered, 'SalePrice')
    
    # Reshape data for sklearn
    X = df_filtered['LotArea'].values.reshape(-1, 1)  # Features
    y = df_filtered['SalePrice'].values  # Target variable

    # Create linear regression object
    reg = LinearRegression()

    # Train the model using the training sets
    reg.fit(X, y)

    # The coefficients
    print('Coefficients: \n', reg.coef_)

    # Plot the original data
    plt.figure(figsize=(10, 6))
    plt.scatter(df_filtered['LotArea'], df_filtered['SalePrice'], alpha=0.5)
    plt.title('Lot Area vs Sale Price (Outliers Removed)')
    plt.xlabel('Lot Area')
    plt.ylabel('Sale Price')

    # Predict y values for the regression line
    y_pred = reg.predict(X)

    # Plot the regression line
    plt.plot(df_filtered['LotArea'], y_pred, color='red', linewidth=2)

    plt.show()
#%%
import pandas as pd
df = pd.read_csv("../data/raw/train.csv")

# Display DataFrame information
#%%
df.info(verbose=True)
# %%
plot_lotarea_saleprice(df)
# %%
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
perform_linear_regression(df)

# %%
perform_linear_regression_with_outlier_removal(df)
# %%
