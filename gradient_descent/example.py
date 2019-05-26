import tensorflow as tf
import numpy as np

def logistic(x_data) :
	b = tf.Variable(tf.zeros([1]));
	w = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0));
	y = tf.matmul(w, x_data) + b;

	return y, w, b;

def train(logistic, y_data) :
	loss = tf.reduce_mean(tf.square(logistic - y_data));
	optimizer = tf.train.GradientDescentOptimizer(0.5);
	return optimizer.minimize(loss);


if __name__ == '__main__' :
	x_data = np.float32(np.random.rand(2, 100));
	y_data = np.dot([0.100, 0.200], x_data) + 0.300;
	logistic_fn, W, b = logistic(x_data);
	train_fn = train(logistic_fn, y_data);

	with tf.Session() as sess :
		sess.run(tf.initialize_all_variables());
		for step in xrange(0, 201) :
			sess.run(train_fn);
			if (step % 20) == 0 :
				print (step, sess.run(W), sess.run(b));
