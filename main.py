import matplotlib
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy
from model import *




mnist=input_data.read_data_sets('MNIST_data/',one_hot=False)

X_train = np.vstack([img.reshape(-1,) for img in mnist.train.images])
#X_train=mnist.train.images
y_train = mnist.train.labels

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
#X_test=mnist.test.images
y_test = mnist.test.labels

X_train=np.reshape(X_train,(-1,1,28,28))
X_test=np.reshape(X_test,(-1,1,28,28))

num_train= X_train.shape[0]
num_val=1000

mask=list(range(num_train-num_val,num_train))
X_val=X_train[mask]
y_val=y_train[mask]

mask=list(range(num_train-num_val))
X_train=X_train[mask]
y_train=y_train[mask]

mean_image=np.mean(X_train,axis=0)
X_train -= mean_image
X_test -= mean_image
X_val -= mean_image

print(X_train.shape)


model=CNNModel(weight_scale=5e-1,learning_rate=3e-3)

channel_num=X_train.shape[1]
print('channelnum: ',channel_num)

X_tmp=X_train[0:5]
print(X_tmp.shape)
out=model.conv_layer(X_tmp,(7,32,1,3),(2,2))
out=model.fc_layer(out,1024)
out=model.fc_layer(out,10)


model.train(X_train,y_train,X_val,y_val,128,10,'sgd')


