import input_data
import tensorflow as tf

learning_rate = 0.001;
training_epochs = 15;
batch_size = 100;
display_step = 1;
hidden_layer1 = 256;
hidden_layer2 = 256;
input_size = 784;
class_size = 10;

weights = {
	'h1' : tf.Variable(tf.random_normal([input_size, hidden_layer1])),
	'h2' : tf.Variable(tf.random_normal([hidden_layer1, hidden_layer2])),
	'out' : tf.Variable(tf.random_normal([hidden_layer2, class_size]))
};

biases = {
	'b1' : tf.Variable(tf.random_normal([hidden_layer1])),
	'b2' : tf.Variable(tf.random_normal([hidden_layer2])),
	'out' : tf.Variable(tf.random_normal([class_size]))
};

def mlp(x) :
	layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1']);
	layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2']);
	out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out']);
	return out_layer;


x = tf.placeholder(dtype = tf.float32, shape = [None, input_size]);
y = tf.placeholder(dtype = tf.float32, shape = [None, class_size]);

logits = mlp(x);

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y));
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate);
train_op = optimizer.minimize(loss_op);

mnist = input_data.read_data_sets('/mnt/mnist', one_hot = True);

with tf.Session() as sess :
	sess.run(tf.global_variables_initializer());

	for epoch in range(training_epochs) :
		avg_cost = 0.;
		total_batch = int(mnist.train.num_examples / batch_size);
		for i in range(total_batch) :
			batch_x, batch_y = mnist.train.next_batch(batch_size);
			_, c = sess.run([train_op, loss_op], feed_dict={x:batch_x, y:batch_y});
			avg_cost += c / total_batch;
			
		if (epoch % display_step) == 0 :
			print ("Epoch : %04d" % (epoch + 1), "cost = {:.9f}".format(avg_cost));
	print('Optimization Finished!');

	pred = tf.nn.softmax(logits);
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1));
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32));
	print ('Accuracy : ', accuracy.eval({x:mnist.test.images, y : mnist.test.labels}));
