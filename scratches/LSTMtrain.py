@@ -0,0 +1,71 @@
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import pandas as pd


tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
inputsize=28
hiddensize=128
rnnstep=28
outputsize=10
weights={'hidden': tf.Variable(tf.random_normal([inputsize,hiddensize])),
         'out': tf.Variable(tf.random_normal([hiddensize,outputsize]))}
bias={'hidden': tf.Variable(tf.random_normal([hiddensize])),
         'out': tf.Variable(tf.random_normal([outputsize]))}

'''a=np.random.uniform(2,3,size=[2,3,2])
print("The original tensor:{}".format(a))
b=np.transpose(a,[1,0,2])
print("The transposed tensor:{}".format(b))
c=np.reshape(b,[-1,2])
print("The new shaped tensor:{}".format(c))
d=np.split(c,3,0)
print("The new splited tensor:{}".format(d))
print(d[0])
e=np.random.uniform(2,3,size=[3,5])
f=np.ones(size=[3,5])
print(e*f)'''
batch_size=128
def _Rnn(_X, _W, _b, nsteps, _name ):
    _X=tf.reshape(_X, [-1, inputsize])
    _H=tf.matmul(_X, _W['hidden']) + _b['hidden']
    _Hsplit=tf.reshape(_H,[-1, nsteps, hiddensize])
    #_Hsplit=tf.split(_H, nsteps, 0)

    cell=tf.nn.rnn_cell.BasicLSTMCell(hiddensize, forget_bias=1.0)
    #init_state=cell.zero_state(batch_size,dtype=tf.float32)
    LSTM_O, LSTM_S=tf.nn.dynamic_rnn(cell, _Hsplit, dtype=tf.float32, time_major=False) #initial_state=init_state)
    print(LSTM_O.shape)
    LSTM_O=tf.transpose(LSTM_O,[1,0,2])
    _out=tf.matmul(LSTM_O[-1], _W['out']) + _b['out']
    return{'x': _X, 'h': _H, 'split': _Hsplit, 'lstm_0': LSTM_O, 'lstm_s':LSTM_S, 'out': _out}

learning_rate=0.001
x =tf.placeholder("float", [None, rnnstep,inputsize])
y =tf.placeholder("float", [None, outputsize])
myrnn =_Rnn(x,weights, bias, rnnstep, 'basic' )
pred =myrnn['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optm=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
accr=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(y,1)),tf.float32))
init=tf.global_variables_initializer()
print("RNN has been established")
sess=tf.Session()
sess.run(init)
for epoch in range (5):
    avg_cost=0
    total_batch=1000
    for i in range(total_batch):
        bach_xs, bach_ys = mnist.train.next_batch(batch_size)
        bach_xs=bach_xs.reshape(-1, rnnstep, inputsize)
        sess.run(optm, feed_dict={x: bach_xs, y: bach_ys})
testings= mnist.test.images.reshape(-1, rnnstep, inputsize)
testlabels=mnist.test.labels
test_acc=sess.run(accr, feed_dict={x:testings, y:testlabels})
print("Test accuracy: %.3f" %(test_acc))


