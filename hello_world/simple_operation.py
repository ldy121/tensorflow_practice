import tensorflow as tf

if __name__ == '__main__' :
	sess = tf.Session();
	print (sess.run(tf.constant(1) + tf.constant(1)));
