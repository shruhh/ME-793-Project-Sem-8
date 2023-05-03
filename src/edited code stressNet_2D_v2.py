from tensorflow.compat import v1 as tf
from tensorflow import keras
import os
import numpy as np
import random
import time
from tf_slim import conv2d_transpose

tf.compat.v1.disable_eager_execution()

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

num_epochs = 1
batch_size =1
decay_steps = 500
decay_rate = 0.98
starter_learning_rate = 1e-4
train_ratio = 0.99
reg_constant = 0.01
drop_out_rate = 0.5
# load the data
train_path = './slice_sample'

data = []
# add all data to list "data"
for f in os.listdir(train_path):
	item = np.load(train_path + '/' + f,'r', True)
	data.append(item)

data = np.array(data) # shape of (2, 32, 32)
data = np.moveaxis(data, 1, -1) # move the axis to the last, then the shape is (32, 32, 2)
indx = int(0.8*len(data))
train = data[:indx]
val = data[indx:]

#normalisation
dat_mean = np.mean(train[:,:,:,0] ,axis=0, keepdims=True)
dat_std = np.std(train[:,:,:,0], axis=0, keepdims=True)
X_train = ((train[:,:,:,0]-dat_mean)/dat_std).reshape(-1,32,32,1)

y_train = train[:,:,:,1].reshape(-1,32,32,1)*10e2
X_val = ((val[:,:,:,0]-dat_mean)/dat_std).reshape(-1,32,32,1)

y_val = val[:,:,:,1].reshape(-1,32,32,1)*10e2


def reset_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

def weight_variable(shape):
	initial = tf.random.truncated_normal(shape, stddev=0.1)

	return(tf.Variable(initial))

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return(tf.Variable(initial))

def conv2d(x, W, s=[1,1,1,1], padding='SAME'):
	if (padding.upper() == 'VALID'):
		return (tf.nn.conv2d(x,filters=W,strides=s,padding='VALID'))
	# SAME
	return (tf.nn.conv2d(x,filters=W,strides=s,padding='SAME'))

def seblock(x, in_cn):
	print(x.shape)
	squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)

	with tf.compat.v1.variable_scope('sq'):
		w = weight_variable([in_cn, in_cn//16])
		b = bias_variable([in_cn//16])
		h = tf.matmul(squeeze, w) + b
		excitation = tf.nn.relu(h)

	with tf.compat.v1.variable_scope('ex'):
		w = weight_variable([in_cn//16, in_cn])  # None1*128
		b = bias_variable([in_cn])
		h = tf.matmul(excitation, w) + b
		excitation = tf.nn.sigmoid(h)  # None*128
		excitation = tf.reshape(excitation, [-1, 1, 1, in_cn])
	return x * excitation

def residual_block(x, cn, scope_name, training):
	with tf.compat.v1.variable_scope(scope_name):
		shortcut = x
		w1 = weight_variable([3, 3, cn, cn])
		b1 = bias_variable([cn])
		x1 = tf.compat.v1.layers.batch_normalization(tf.nn.relu(conv2d(x, w1) + b1), training = training)
		w2 = weight_variable([3,3, cn, cn])
		b2 = bias_variable([cn])
		x2 = tf.compat.v1.layers.batch_normalization(conv2d(x1, w2) + b2, training = training)
		x3 = seblock(x2, cn)

	return x3 + shortcut

reset_graph() # this is very important for batchNorm to work

X = tf.compat.v1.placeholder(tf.float32, shape = [None, 32, 32, 1], name = 'geometry')
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 32, 32, 1], name = 'stress')
training = tf.compat.v1.placeholder_with_default(False, shape=(), name='training')


C_1 = tf.compat.v1.layers.conv2d(inputs = X, kernel_size = 3, filters = 32, padding = 'same',activation = tf.nn.relu)
C_1 = tf.nn.dropout(C_1, rate=1 - (drop_out_rate))
C_1 = tf.compat.v1.layers.max_pooling2d(inputs = C_1, pool_size = 2, strides = 2, padding = 'same') # shape 16


C_2 = tf.compat.v1.layers.conv2d(inputs = C_1, kernel_size = 3, filters = 64, padding = 'same', activation = tf.nn.relu)
C_2 = tf.nn.dropout(C_2, rate=1 - (drop_out_rate))
C_2 = tf.compat.v1.layers.max_pooling2d(inputs = C_2, pool_size = 2, strides = 2, padding = 'same') # shape 8

x4 = residual_block(C_2, 64, 'res1', training = training)
x5 = residual_block(x4, 64, 'res2',training = training)
x6 = residual_block(x5, 64, 'res3',training = training)
x7 = residual_block(x6, 64, 'res4',training = training)
x8 = residual_block(x7, 64, 'res5',training = training)

x9 = conv2d_transpose(x8, 64, kernel_size = 3, stride = 2, padding = 'SAME')
x9 = tf.nn.dropout(x9, rate=1 - (drop_out_rate))
x9 = tf.nn.relu(x9)
x9 = tf.compat.v1.layers.batch_normalization(x9, training = training) # shape 16

x10 = conv2d_transpose(x9,32, kernel_size = 3, stride= 2, padding='SAME')
x10 = tf.nn.dropout(x10, rate=1 - (drop_out_rate))
x10 = tf.nn.relu(x10)
x10 = tf.compat.v1.layers.batch_normalization(x10, training = training) # shape 32
output = tf.compat.v1.layers.conv2d(x10, filters = 1, kernel_size = 3, padding = 'SAME') # shape 32 32 1


global_step = tf.Variable(0, trainable=False)
add_global = global_step.assign_add(1)
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate,
                                               global_step=global_step,
											   decay_steps=decay_steps,decay_rate=decay_rate)


with tf.compat.v1.variable_scope('loss'):
	reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
	loss = tf.compat.v1.losses.mean_squared_error(y, output) + reg_constant*sum(reg_losses)
	train_step = tf.compat.v1.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
	SS_tol = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
	SS_res = tf.reduce_sum(tf.square(y-output))
	R_squared = 1. - SS_res/SS_tol


extra_graphkeys_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

#set GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
saver = tf.compat.v1.train.Saver()
start_time = time.localtime()
print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))

epoch_data = []

with tf.compat.v1.Session(config=config) as sess:
	sess.run(tf.compat.v1.global_variables_initializer())
	num= int(len(X_train)/batch_size)
	best_validation = 0
	for epoch in range(num_epochs):
		train_loss = 0
		train_mae = 0
		for itr in range(num):
			X_batch = X_train[itr*batch_size:(itr+1)*batch_size]
			y_batch = y_train[itr*batch_size:(itr+1)*batch_size]
			_train_step, _loss, _r2 , _update= sess.run([train_step, loss, R_squared, extra_graphkeys_update_ops], feed_dict = {training: True, X: X_batch, y: y_batch})
			train_loss += _loss
			train_mae += _r2
		val_loss, val_r2 = sess.run([loss, R_squared], feed_dict = {X: X_val, y: y_val})

		if val_r2 > best_validation:
			saved_model = saver.save(sess, './model_do_all_rate0.5')
			best_validation = val_r2
		epoch_data.append([val_loss, val_r2])
		print('Epoch: {} Training Loss: {:.4f} Trianing R2: {:.4f} Val Loss: {:.4f} Val R2: {:.4f} Best So Far {:.4f}'.format(epoch+1,train_loss/num, _r2, val_loss, val_r2, best_validation))




