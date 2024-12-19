import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
# from tensorflow import tf
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split


fuel = pd.read_csv("C:/Users/Admin/Desktop/RESEARCH TRANSLATION/fuel.csv")

X = fuel.copy()
# Remove target
y = X.pop('FE')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=object)),
)

X = preprocessor.fit_transform(X)
y = np.log(y) # log transform target instead of standardizing

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))

# Uncomment to see original data
# print(fuel.head())
# Uncomment to see processed features
# processed_features = pd.DataFrame(X[:10,:]).head()
# print(processed_features)

# define the layers to use
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])


# 1) Add Loss and Optimizer
model.compile(
    optimizer="adam",
    loss="mae",
)

# 2) Train Model
history = model.fit(
    X, y,
    batch_size=128,
    epochs=200
)


# Plot training and validation loss
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();
# plt.show()


# 3) Evaluate Training
