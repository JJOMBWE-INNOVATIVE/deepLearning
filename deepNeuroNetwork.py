import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import tf
import matplotlib.pyplot as plt

concrete = pd.read_csv("C:/Users/Admin/Desktop/RESEARCH TRANSLATION/concrete.csv")
print(concrete.head())

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units= 512, activation='relu', input_shape=[8]),
    layers.Dense(units= 512, activation='relu'),
    layers.Dense(units= 512, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])


# ### another way to write activation layers
# model = keras.Sequential([
# the hidden ReLU layers
#     layers.Dense(units= 512, input_shape=[8]),
#     layers.Activation('relu'),
#     layers.Dense(512),
#     layers.Activation('relu'),
#     layers.Dense(512),
#     layers.Activation('relu'),
 # the linear output layer 
#     layers.Dense(1),
# ])

# YOUR CODE HERE: Change 'relu' to 'elu', 'selu', 'swish'... or something else
activation_layer = layers.Activation('swish')

x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()