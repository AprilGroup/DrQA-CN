"""Rank docs with query-content score and query-title score"""

import tensorflow as tf
import numpy as np
import pandas as pd


def convert_sparse_matrix_to_sparse_tensor(X):
    """Convert Scipy.sparse.csr_matrix to tf.SparseTensor."""
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def num2onehot(x):
    vec = np.zeros(76437)
    vec[x] = 1
    return vec

# def dataset_input_fn():
# load query scores
with np.load('../../data/retriever/query_scores_sparse.npz') as loader:
    all_title_scores = loader['title_scores']  # list of csr_matrix
    all_doc_scores = loader['doc_scores']  # list of csr_matrix
    labels = loader['labels']
    labels = np.array([num2onehot(x) for x in labels])  #
    # labels = pd.get_dummies(labels).values


# def parser(doc_scores, title_scores, label):
#     doc_scores = convert_sparse_matrix_to_sparse_tensor(doc_scores)
#     title_scores = convert_sparse_matrix_to_sparse_tensor(title_scores)
#
#     return doc_scores, title_scores, label


# build feature matrix from sparse matrices, shape(num_examples, 152874), first half columns
# represent title scores, second half doc scores.
# put all_title_scores and all_doc_scores into vector of shape(152874)
doc_scores = []
title_scores = []
for single_title_scores, single_doc_scores in zip(all_title_scores, all_doc_scores):
    dense_title_scores = single_title_scores.toarray().reshape(76437,).tolist()
    title_scores.append(dense_title_scores)
    dense_doc_scores = single_doc_scores.toarray().reshape(76437,).tolist()
    doc_scores.append(dense_doc_scores)
doc_scores = np.array(doc_scores)
title_scores = np.array(title_scores)

# use placeholder to build dataset instead of directly using numpy arrays to
# save memory
doc_scores_placeholder = tf.placeholder(doc_scores.dtype, doc_scores.shape)
title_scores_placeholder = tf.placeholder(title_scores.dtype, title_scores.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

assert all_title_scores.shape[0] == labels.shape[0]

# dataset from placeholder recommended.
dataset = tf.data.Dataset.from_tensor_slices((doc_scores_placeholder, title_scores_placeholder, labels_placeholder))
# dataset from numpy arrays not recommended if dataset is large.
# dataset = tf.data.Dataset.from_tensor_slices((doc_scores, title_scores, labels))

# nomalization on dataset, transform label to one-hot vector
# dataset = dataset.map(parser)
# shuffle
# dataset = dataset.shuffle(buffer_size=10000)
# batch
dataset = dataset.batch(20)
iterator = dataset.make_initializable_iterator()
next_batch = iterator.get_next()

sess = tf.Session()

# Initialize the iterator
sess.run(iterator.initializer,
         feed_dict={doc_scores_placeholder: doc_scores,
                    title_scores_placeholder: title_scores,
                    labels_placeholder: labels})

# # training loop
# for i in range(2):
#     doc_scores, title_scores, labels = sess.run(next_batch)
#     print(doc_scores.shape, title_scores.shape, labels.shape)



# parameters
learning_rate = 0.1
training_epoch = 2
batch_size = 20
display_step = 1

num_docs = 76437
#
x1 = tf.placeholder(tf.float32, [None, num_docs])
x2 = tf.placeholder(tf.float32, [None, num_docs])
y = tf.placeholder(tf.float32, [None, num_docs])
#
# # variables
W = tf.Variable(0., dtype=tf.float32, name='title_weight')
# constants
C = tf.constant(1., dtype=tf.float32)
#
# # calculate predictions
pred = tf.nn.softmax(x1 * W + x2 * (C - W))
#
# loss function
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
#
# # optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# init
init = tf.global_variables_initializer()

for epoch in range(training_epoch):
    sess.run(init)
    avg_loss = 0.
    total_batch = int(len(labels) // batch_size + 1)
    for index in range(total_batch):
        batch_x1, batch_x2, batch_y = sess.run(next_batch)
        _, loss = sess.run([optimizer, loss], feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y})
        avg_loss += loss / total_batch
    if (epoch + 1) % display_step == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_loss))
print('Optimization Finished!')
print(W)
#

# w = tf.Variable("weight", shape=2, dtype=tf.float32, initializer=)