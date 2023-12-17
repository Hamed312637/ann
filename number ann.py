#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
from tensorflow import keras
from keras.layers import Input,Dense,SimpleRNN
from tensorflow.keras.optimizers import Adam, SGD
from keras.models import Model
import pandas as pd
import numpy as np


# In[7]:


path = "E:\\mnst\\mnist_train.csv"
path2 = "E:\\mnst\\mnist_test.csv"
train = pd.read_csv(path)
test = pd.read_csv(path2)
y_train=train.iloc[:,0]
x_train=train.iloc[:,1:]
y_test=test.iloc[:,0]
x_test=test.iloc[:,1:]
x_test=x_test.to_numpy()
x_train=x_train.to_numpy()
y_test=y_test.to_numpy()
y_train=y_train.to_numpy()
print(x_train.shape)
print(x_test.shape)

x_train,x_test=x_train/255.0,x_test/255.0


# In[8]:


model=tf.keras.models.Sequential([
    
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(200,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[9]:


r=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)


# In[10]:


import matplotlib.pyplot as plt
plt.plot(r.history['loss'],label='loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()


# In[11]:


plt.plot(r.history['accuracy'],label='acc')
plt.plot(r.history['val_accuracy'],label='val_acc')
plt.legend()


# In[12]:


print(model.evaluate(x_test,y_test))


# In[13]:


from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
p_test=model.predict(x_test).argmax(axis=1)


# In[14]:


x_test=np.reshape(x_test,(10000,28,28))
correctly_classified=np.where(p_test == y_test)[0]
j=np.random.choice(correctly_classified)
plt.imshow(x_test[j],cmap = 'gray')
plt.title("true label:%s predicted %s" % (y_test[j],p_test[j]));


# In[15]:


misclassified_idx=np.where(p_test != y_test)[0]
i=np.random.choice(misclassified_idx)
plt.imshow(x_test[i],cmap='gray')
plt.title("true label: %s predicted %s" % (y_test[i],p_test[i]));


# In[ ]:




