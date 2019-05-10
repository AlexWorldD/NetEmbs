# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
tensor_flow.py
Created by lex at 2019-04-12.
"""
import tensorflow as tf
import math
from NetEmbs.SkipGram.generate_batch import generate_batch
from NetEmbs.DataProcessing import *
from NetEmbs.FSN.utils import get_SkipGrams
import time
import pandas as pd
import numpy as np
from NetEmbs.DataProcessing.connect_db import upload_JournalEntriesTruth
from NetEmbs.CONFIG import EMBD_SIZE, BATCH_SIZE, WORK_FOLDER
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_embs_TF(input_data=("../Simulation/FSN_Data.db", 496), embed_size=None, num_steps=10000, walk_length=10,
                walks_per_node=10, save_embs=False, use_cached_skip_grams=False, use_prev_embs=False):
    # Check the input argument type: FSN or DataFrame
    if isinstance(input_data, pd.DataFrame):
        # #     Construct FSN object from the given df
        d = input_data
    elif isinstance(input_data, tuple):
        d = prepare_data(upload_data(input_data[0], limit=input_data[1]))
    else:
        raise ValueError(
            "As input data should be DataFrame with journal entries of the path to DataBase! Was given {!s}!".format(
                type(input_data)))
    print("Total number of BPs in given dataset is ", d.ID.nunique())
    skip_grams, fsn, enc_dec = get_SkipGrams(d, walks_per_node=walks_per_node, walk_length=walk_length,
                                             use_cache=use_cached_skip_grams)
    print(skip_grams[:5])
    #
    #     TensorFlow stuff
    batch_size = BATCH_SIZE
    if embed_size is not None:
        embedding_size = embed_size
    else:
        embedding_size = EMBD_SIZE  # Dimension of the embedding vector

    neg_number = 100
    valid_size = 4
    total_size = fsn.number_of_BP()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        # Input variables
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Embeddings matrix initialisation
        if use_prev_embs:
            print("Loading previous embeddings from cache... wait...")
            embeddings = tf.Variable(pd.read_pickle(WORK_FOLDER+"snapshot.pkl").values)
        else:
            embeddings = tf.Variable(tf.random_uniform((total_size, embedding_size), -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # ----
        # Output layer parameters
        weights = tf.Variable(tf.truncated_normal((total_size, embedding_size),
                                                  stddev=1.0 / math.sqrt(embedding_size)))
        biases = tf.Variable(tf.zeros((total_size)))

        # ---- Version 1
        # hidden_out = tf.matmul(embed, tf.transpose(weights)) + biases
        # # ----
        # # convert train_context to a one-hot format
        # train_one_hot = tf.one_hot(train_context, total_size)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hidden_out,
        #                                                                           labels=train_one_hot))
        # # Construct the SGD optimizer using a learning rate of 1.0.
        # optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
        # Compute the cosine similarity between minibatch examples and all embeddings.

        #  ---- Version 2, like Marcel's example
        # Calculate the loss using negative sampling
        loss = tf.nn.sampled_softmax_loss(weights, biases,
                                          train_context, embed,
                                          neg_number, total_size)

        cost = tf.reduce_mean(loss)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Validation subset of BPs
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent

        valid_examples = np.random.randint(0, total_size, valid_size)
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        valid_embedding = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embeddings))
        # Add variable initializer.
        init = tf.global_variables_initializer()

        def run2(graph, num_steps, data, batch_size, enc_dec):
            with tf.Session(graph=graph) as session:
                # We must initialize all variables before we use them.
                init.run()
                print('Initialized')

                average_loss = 0
                for step in range(num_steps + 1):
                    batch_inputs, batch_context = generate_batch(data,
                                                                 batch_size)
                    feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

                    _, loss_train = session.run([optimizer, cost], feed_dict=feed_dict)
                    average_loss += loss_train

                    if step % 5000 == 0:
                        if step > 0:
                            average_loss /= 5000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average train loss at step ', step, ': ', average_loss)
                        average_loss = 0

                    # if step % 20000 == 0:
                    #     # note that this is expensive (~20% slowdown if computed every 500 steps)
                    #     sim = similarity.eval()
                    #     for i in range(valid_size):
                    #         valid_word = enc_dec.decoder[valid_examples[i]]
                    #         top_k = 3  # number of nearest neighbors
                    #         nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    #         log = 'Nearest to %s:' % valid_word
                    #         for k in range(top_k):
                    #             close_word = enc_dec.decoder[nearest[k]]
                    #             log = '%s %s,' % (log, close_word)
                    #         print(log)
                final_embeddings = normalized_embeddings.eval()
                return final_embeddings
    #     Run
    start_time = time.time()
    embs = run2(graph, num_steps, skip_grams, batch_size, enc_dec)
    pd.DataFrame(embs).to_pickle(WORK_FOLDER+"snapshot.pkl")
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    fsn_embs = pd.DataFrame(list(zip(enc_dec.original_bps, embs)), columns=["ID", "Emb"])
    print("Done with TensorFlow!")
    if isinstance(save_embs, str):
        fsn_embs.to_pickle(save_embs + ".pkl")
    return fsn_embs


def add_ground_truth(df, path_file="../Simulation/FSN_Data.db"):
    journal_truth = upload_JournalEntriesTruth(path_file)[["ID", "FA_Name"]]
    journal_truth.rename(index=str, columns={"FA_Name": "GroundTruth"}, inplace=True)
    return df.merge(journal_truth, on="ID")
