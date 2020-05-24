#!/usr/bin/env python3
# coding: utf-8



from keras.layers import Convolution2D





from keras.layers import MaxPooling2D


	


from keras.layers import Flatten


# In[4]:


from keras.layers import Dense


# In[5]:


from keras.models import Sequential


# In[6]:


model = Sequential()


# In[7]:


model.add(Convolution2D(filters=32,
                        kernel_size=(3,3),
                        activation='relu',
                    input_shape=(64,64,3)
                       ))


# In[8]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[9]:


model.add(Flatten())


# In[10]:


model.add(Dense(units=128,activation='relu'))


# In[11]:


model.add(Dense(units=1,activation='sigmoid'))


# In[12]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[13]:


from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set   = train_datagen.flow_from_directory(
        '/training_set/',
        target_size=(64, 64 ),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        '/test_set/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
model.fit(
        training_set,
        steps_per_epoch=8000,
        epochs=2,
        validation_data=test_set,
        validation_steps=800)





