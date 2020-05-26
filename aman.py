#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from keras.preprocessing import image


# In[2]:


(train_x, train_y) , (test_x, test_y) = mnist.load_data()
train_x = train_x.reshape(60000,784)
test_x = test_x.reshape(10000,784)
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)


# In[ ]:


accuracy = 0
counter =1
learning_rate =0.1
epoch= 10


# In[ ]:



while accuracy < .90 :
    model = Sequential()
    for i in range(counter) :
            model.add(Dense(units=128,activation="relu",input_shape=(784,)))
    
    
    
    counter = counter +1
    print("counter is ",counter)
    
    model.add(Dense(units=10,activation="softmax"))      
    learning_rate = learning_rate/10
    print("learning_rate is", learning_rate)
    model.compile(optimizer=SGD(learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(train_x,train_y,batch_size=32,epochs=epoch,verbose=1)
    model.summary()
    Accuracy = model.evaluate(x=test_x,y=test_y,batch_size=32)
    print("Accuracy: ",Accuracy[1])
    accuracy = Accuracy[1]
    print(accuracy)s