# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
tensor_flow.py
Created by lex at 2019-04-12.
"""
import tensorflow as tf
import math


def init_tf(fsn):
    batch_size = 32
    embedding_size = 4  # Dimension of the embedding vector
    total_size = fsn.number_of_BP()
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])
        embeddings = tf.Variable(tf.random_uniform([total_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # ----
        # Construct the variables for the softmax
        weights = tf.Variable(tf.truncated_normal([total_size, embedding_size],
                                                  stddev=1.0 / math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros([total_size]))
        hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases
        # ----
        # convert train_context to a one-hot format
        train_one_hot = tf.one_hot(train_context, total_size)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hidden_out,
                                                                                  labels=train_one_hot))
        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)
        # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        # Add variable initializer.
        init = tf.global_variables_initializer()
    return graph

