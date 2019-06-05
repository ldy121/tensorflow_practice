import tensorflow as tf
import numpy as np
import input_data

conv_size = 3;
pooling_size = 2;
channel_size = 1;
learning_rate = 0.01;
epochs = 100;

def create_layer(prev_layer, input_channel, output_channel) :
	batch_size = 1;
	weight = tf.Variable(tf.random_normal([conv_size, conv_size, input_channel, output_channel]), name = 'conv_weight');
	conv = tf.nn.conv2d(prev_layer, filter = weight, strides = [1, 1, 1, 1], padding = 'SAME');
	relu = tf.nn.relu(conv);
	layer = tf.nn.max_pool(relu, [batch_size, pooling_size, pooling_size, channel_size],
			strides = [batch_size, pooling_size, pooling_size, channel_size], padding = 'SAME');

	return layer;

def fully_connected_layer(prev_layer, input_size, output_size) :
	weight = tf.Variable(tf.random_normal([input_size, output_size]), name = 'fully_connected_layer_weight');
	bias = tf.Variable(tf.random_normal([output_size]), name = 'fully_connected_layer_bias');
	layer = tf.matmul(prev_layer, weight) + bias;
	return layer;

def create_logits(input_layer, input_channel, output_channel) :
	conv1_filter = 64;
	conv2_filter = 32;

	conv1 = create_layer(input_layer, input_channel, conv1_filter);
	conv2 = create_layer(conv1, conv1_filter, conv2_filter);

	fully_connected_layer_input_size = tf.cast(conv2.shape[1] * conv2.shape[2] * conv2.shape[3], tf.int32);

	fully_connected_layer_input = tf.reshape(conv2, [-1, fully_connected_layer_input_size]);

	return fully_connected_layer(fully_connected_layer_input, fully_connected_layer_input_size, output_channel);

if __name__ == '__main__' :
	batch_size = 100;
	input_size = 784;
	output_size = 10;

	x = tf.placeholder(dtype = tf.float32, shape = [None, input_size]);
	y = tf.placeholder(dtype = tf.float32, shape = [None, output_size]);

	input_layer = tf.reshape(x, [-1, 28, 28, 1]);

	logits = create_logits(input_layer, 1, output_size);

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y));
	train_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost);

	prediction_accurate = tf.equal(tf.argmax(tf.nn.softmax(logits), 1), tf.argmax(y, 1));
	test_op = tf.reduce_mean(tf.cast(prediction_accurate, tf.float32));

	mnist = input_data.read_data_sets('/mnt/mnist', one_hot = True);
	total_batch = int(mnist.train.num_examples / batch_size);
	print (total_batch);
	with tf.Session() as sess :
		sess.run(tf.initialize_all_variables());
		for i in range(epochs) :
			for j in range(total_batch) :
				train_x, train_y = mnist.train.next_batch(batch_size);
				_, ret = sess.run([train_op, cost], feed_dict = {x : train_x, y : train_y});

			accuracy = sess.run(test_op, feed_dict = {x : mnist.test.images, y : mnist.test.labels});
			print (accuracy);
