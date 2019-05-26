import tensorflow as tf
import input_data
import numpy as np

learning_rate = 0.001;
epoch_size = 15;
batch_size = 100;

def fully_connected(input, input_size, output_size, name) :
	w = tf.Variable(tf.random_normal([input_size, output_size]), name = name + '_weight');
	b = tf.Variable(tf.random_normal([output_size]), name = name + '_bias');
	y = tf.add(tf.matmul(input,w), b);
	return y;

def get_logits(x) :
	input_layer_width = 784;
	hidden_layer_width = 256;
	output_layer_width = 10;
	layer1 = fully_connected(x, input_layer_width, hidden_layer_width, name = 'input_layer');
	layer2 = fully_connected(layer1, hidden_layer_width, hidden_layer_width, name = 'hidden_layer1');
	layer3 = fully_connected(layer2, hidden_layer_width, output_layer_width, name = 'hidden_layer2');
	return fully_connected(layer3, output_layer_width, output_layer_width, name = 'last_layer');

def get_train_op(logits, y) :
	loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y));
	optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate);
	train_op = optimizer.minimize(loss_op);
	return train_op, loss_op;

def evaluate(logits, x_data, y) :
	x = tf.placeholder(name = 'x_data', shape = [None, 784], dtype = tf.float32);
	result = (sess.run(logits, feed_dict = {x : x_data}));
	estimated_result = sess.run(tf.argmax(result,1))[0];
	valid_result = np.argmax(y);
	if (estimated_result != valid_result) :
		print ("Wrong estimatation : %d / valid value : %d" % (estimated_result, valid_result));

if __name__ == '__main__' :
	x = tf.placeholder(name = 'x_data', shape = [None, 784], dtype = tf.float32);
	y = tf.placeholder(name = 'y_data', shape = [None, 10], dtype = tf.float32);
	logits = get_logits(x);
	train_fp, loss_fp = get_train_op(logits, y);
	mnist = input_data.read_data_sets('/mnt/mnist', one_hot = True);

	with tf.Session() as sess :
		sess.run(tf.initialize_all_variables());
		for i in range(epoch_size) :
			avg_cost = 0.;
			total_batch = int(mnist.train.num_examples / batch_size);
			for j in range(total_batch) :
				batch_xs, batch_ys = mnist.train.next_batch(batch_size);
				_, c = sess.run([train_fp, loss_fp], feed_dict = {x : batch_xs, y : batch_ys});
				avg_cost += c / total_batch;
			print ("Epoch : %04d" % (i + 1), "cost = {:.9f}".format(avg_cost));

		pred = tf.nn.softmax(logits);
		correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y, 1));
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'));
		print(accuracy.eval({x : mnist.test.images, y : mnist.test.labels}));
