# python 2.7.10

import tensorflow as tf
import numpy as np
from PIL import Image

def unpicle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

''' 4.1. 
    shows a random image from cifar10 to make sure we're reading the data correctly '''

def show_pic(rgbArr):
    img = np.array(rgbArr).reshape(3, 32, 32).transpose(1, 2, 0)
    img = Image.fromarray(img, 'RGB')
    img.show()

# dict_1 = unpicle("cifar-10-batches-py/data_batch_1")
# show_pic(dict_1['data'][0])


''' 4.2.
    training cifar10 with two layer CNN '''

def conv_layer(input, channel_in, channel_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, channel_in, channel_out], mean=0, dtype=tf.float32, stddev=0.2), name="W")
        b = tf.Variable(tf.constant(0.1, tf.float32, [channel_out]), name="b")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='SAME') + b
        act = tf.nn.relu(conv)
        tf.summary.histogram("W", w)
        tf.summary.histogram("b", b)
        scaled = (w - tf.reduce_min(w)) / (tf.reduce_max(w) - tf.reduce_min(w))
        transposed = tf.transpose(scaled, [3, 0, 1, 2])
        if w.shape[2] == 3:
            tf.summary.image('first_layer_filters', transposed, 64)
        else:
            for i in range(2):
                transposed2 = tf.transpose(transposed[i], [2, 0, 1])
                # for j in range(64):
                im = tf.reshape(transposed2, [-1, 5, 5, 1])
                tf.summary.image('second_layer_filters', im, 64)
        return act

def fc_layer(input, channel_in, channel_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([channel_in, channel_out], mean=0, dtype=tf.float32, stddev=0.2), name="W")
        b = tf.Variable(tf.constant(0.1, tf.float32, [channel_out]), name="b")
        tf.summary.histogram("W", w)
        tf.summary.histogram("b", b)
        act = tf.matmul(input, w) + b
        if channel_out!=10:
            act = tf.nn.relu(act)
        return act

def max_pool_3x3(x, name="pool"):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def one_hot_array(arr):
    """ returns 2dim array of one_hot encoded integers"""
    arr = np.array(arr)
    return np.eye(10, dtype=float)[arr]

# Network architecture constants:
batch_size = 50
epoch_size = 4
learning_rate = 0.001

# data definitions:
x_array = tf.placeholder(tf.float32, shape=(None, 3072), name="x_array")
x_image = tf.transpose(tf.reshape(x_array, shape=(-1, 3, 32, 32)), [0, 2, 3, 1])
y = tf.placeholder(tf.float32, shape=(None, 10), name="y")
tf.summary.image("input", x_image, 3)

# layer definitions:
conv1 = conv_layer(x_image, 3, 64, "conv1")
pool1 = max_pool_3x3(conv1, "pool1")

conv2 = conv_layer(pool1, 64, 64, "conv2")
pool2 = max_pool_3x3(conv2, "pool2")
pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])

fc1 = fc_layer(pool2_flat, 8*8*64, 512, "fc1")
logits = fc_layer(fc1, 512, 10, "output")

with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
    tf.summary.scalar("cross entropy", cross_entropy)

with tf.name_scope("train"):
    # train the network using adam optimizer:
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    # accuracy:
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# prepare data:
data_train = unpicle("cifar-10-batches-py/data_batch_1")
data_test = unpicle("cifar-10-batches-py/test_batch")
label_names = unpicle("cifar-10-batches-py/batches.meta")['label_names']

    # merge all 5 batches in cifar folder
    #TODO: middle = 5
for i in range(1, 1, 1):
    dict = unpicle("cifar-10-batches-py/data_batch_"+ str(i+1))
    data_train['data'] = np.concatenate((data_train['data'], dict['data']), axis=0)
    data_train['labels'] = np.concatenate((data_train['labels'], dict['labels']), axis=0)

training_size = len(data_train['data'])
test_size = len(data_test['data'])

    # make x arrays 32x32x3 images
# raw_tr = np.array(data_train['data'], dtype=float).reshape([training_size, 3, 32, 32])
# data_train['data'] = raw_tr.transpose([0, 2, 3, 1])
# raw_te = np.array(data_test['data'], dtype=float).reshape([test_size, 3, 32, 32])
# data_test['data'] = raw_te.transpose([0, 2, 3, 1])

    # make y one_hot vectors
data_train['labels'] = one_hot_array(data_train['labels'])
data_test['labels'] = one_hot_array(data_test['labels'])

# start training:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#sess = tf.Session()
writer = tf.summary.FileWriter('./log/', sess.graph)
merge = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess=sess, save_path='./save/model.ckpt')

for epoch in range(epoch_size):
    for i in range(0, training_size, batch_size):
        batch_x = data_train['data'][i:i+batch_size]
        batch_y = data_train['labels'][i:i+batch_size]

        sess.run(train_op, feed_dict={x_array: batch_x, y: batch_y})

        if i%(100*batch_size)==0:
            mrg_smry = sess.run(merge, feed_dict={x_array: batch_x, y: batch_y})
            # smry = tf.Summary(value=[tf.Summary.Value(tag="training loss", simple_value=training_loss)])
            writer.add_summary(mrg_smry, i)
            saver.save(sess, './save/model.ckpt')

    if epoch%1 == 0:
        print ("epoch #%d" % (epoch))
        print "training accuracy: ", sess.run(accuracy, feed_dict={x_array: data_train['data'], y: data_train['labels']})
        print "test accuracy: ", sess.run(accuracy, feed_dict={x_array: data_test['data'], y: data_test['labels']}), "\n"

print "final training accuracy: " , sess.run(accuracy, feed_dict={x_array: data_train['data'], y: data_train['labels']})
print "final test accuracy: " , sess.run(accuracy, feed_dict={x_array: data_test['data'], y: data_test['labels']}), "\n"

