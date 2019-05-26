import tensorflow as tf
import input_data
import numpy as np

def get_logist(x) :
	w = tf.Variable(tf.zeros([784, 10]));
	b = tf.Variable(tf.zeros([10]));
	y = tf.nn.softmax(tf.matmul(x,w) + b);
	return y;

def get_loss(logist, y) :
	cross_entory = -tf.reduce_sum(y * tf.log(logist));
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entory);
	return train_step;

if __name__ == '__main__' :
	x = tf.placeholder(name = 'x_data', shape = [None, 784], dtype = tf.float32);
	y = tf.placeholder(name = 'y_data', shape = [None, 10], dtype = tf.float32);
	logists = get_logist(x);
	train_fp = get_loss(logists, y);
	mnist = input_data.read_data_sets('/mnt/mnist', one_hot = True);

	with tf.Session() as sess :
		sess.run(tf.initialize_all_variables());
		for i in range(1000) :
			batch_xs, batch_ys = mnist.train.next_batch(100);
			sess.run(train_fp, feed_dict = {x : batch_xs, y : batch_ys});
			random_number = int(np.random.rand(1) * 100);
			result = (sess.run(logists, feed_dict = {x : [batch_xs[random_number]]}));

			estimated_result = sess.run(tf.argmax(result,1))[0];
			valid_result = np.argmax(batch_ys[random_number]);
			if (estimated_result != valid_result) :
				print ("Wrong estimatation : %d / valid value : %d" % (estimated_result, valid_result));
	
