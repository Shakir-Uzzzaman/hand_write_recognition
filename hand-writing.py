import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

i=100

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

x_train =x_train /255
x_test=x_test/255

x_train_flatten=x_train.reshape(len(x_train), 28*28)
x_test_flatten=x_test.reshape(len(x_test), 28*28)
#n=x_train_flatten.shape
#print(n)

model=keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
    
    ])

model.compile(
    optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']
    
    )

model.fit(x_train_flatten,y_train,epochs=7)
model.fit(x_test_flatten,y_test)

plt.matshow(x_test[i])

y_predict=model.predict(x_test_flatten)
print(y_predict[i])
n=np.argmax(y_predict[i])
print(n)

