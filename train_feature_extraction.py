import pickle, time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle

nb_classes = 43

# TODO: Load traffic signs data.
with open('./train.p', 'rb') as f:
	data = pickle.load(f)

# TODO: Split data into training and validation sets.

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.2, random_state=0)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, nb_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 64
EPOCHS = 20
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))
    return total_loss/ num_examples, total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	for i in range(EPOCHS):
		X_train ,y_train = shuffle(X_train, y_train)
		t0 = time.time()
		for offset in range(0, X_train.shape[0], BATCH_SIZE):
			end = offset + BATCH_SIZE
			sess.run(training_operation, feed_dict={x: X_train[offset:end], y: y_train[offset:end]})

		val_loss, val_acc = evaluate(X_val, y_val)
		print("Epoch %d" % i+1)
		print("Time: %.3f seconds" % (time.time() - t0))
		print("Validation loss: %.3f" % val_loss)
		print("Validation accuracy: %.3f" % val_acc)
		print()
