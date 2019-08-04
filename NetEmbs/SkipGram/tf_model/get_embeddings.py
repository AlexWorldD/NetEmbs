# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
get_embeddings.py
Created by lex at 2019-08-01.
"""
import os
import tensorflow as tf
import math
from NetEmbs.SkipGram.tf_model.batch_generator import generate_batch
import time
import pandas as pd
from NetEmbs import CONFIG
import logging
from typing import Union, Optional, List
from NetEmbs.SkipGram.construct_skip_grams import TransformationBPs
from NetEmbs.utils.update_config import set_new_config
from NetEmbs.utils.dimensionality_reduction import dim_reduction
import pickle

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def variable_summaries(var):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


def get_embeddings(skip_grams: List[List[Union[str, int]]], tr: TransformationBPs,
                   use_cache: Optional[bool] = True, use_dim_reduction: Optional[bool] = True,
                   **kwargs):
    """
    Learning embeddings for the given Skip-Grams

    Construct simple TF-model according to the chosen settings.
    Parameters
    ----------
    skip_grams : List of tuples <cur_node, context_node>
    tr : TransformationBPs object
        Using for transforming the nodes ID into integers and vise versa.
    use_cache : bool, default is True
        To use the previously cached files
    use_dim_reduction : bool, default if True
        Additionally add X,Y columns to the result DataFrame for vis purposes
    kwargs : settings for TF-model to update CONFIG file.

    Returns
    -------
    DataFrame with ID, Emb (and X,Y if use_dim_reduction=True).
    """
    set_new_config(**kwargs)
    local_logger = logging.getLogger(f"{__name__}")
    local_logger.info("Initialising TF model")
    print(f"Current TensorFlow parameters: \n Embedding size:  {CONFIG.EMBD_SIZE} \n Steps:  {CONFIG.STEPS}"
          f"\n Batch size:  {CONFIG.BATCH_SIZE}")
    total_size = tr.number_BPs()
    #     TF model initialisation
    tf.reset_default_graph()
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        # Input variables
        train_inputs = tf.placeholder(tf.int32, shape=[CONFIG.BATCH_SIZE])
        train_context = tf.placeholder(tf.int32, shape=[CONFIG.BATCH_SIZE, 1])
        embeddings = tf.Variable(tf.random_uniform((total_size, CONFIG.EMBD_SIZE), -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # Output layer parameters
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal((total_size, CONFIG.EMBD_SIZE),
                                                      stddev=1.0 / math.sqrt(CONFIG.EMBD_SIZE)))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros(total_size))

        # Calculate the loss using negative sampling
        with tf.name_scope('cost'):
            if CONFIG.LOSS_FUNCTION == "NCE":
                loss = tf.nn.nce_loss(weights, biases,
                                      train_context, embed,
                                      CONFIG.NEGATIVE_SAMPLES, total_size)
            else:
                loss = tf.nn.sampled_softmax_loss(weights, biases,
                                                  train_context, embed,
                                                  CONFIG.NEGATIVE_SAMPLES, total_size)

            cost = tf.reduce_mean(loss)
            tf.summary.scalar('Cost', cost)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + '/train', tf_graph)
        init = tf.global_variables_initializer()

        #     Run function
        def run(graph, data):
            with tf.Session(graph=graph) as session:
                init.run()
                print('Initialized tf-model')

                average_loss = 0
                for step in range(CONFIG.STEPS + 1):
                    batch_inputs, batch_context = generate_batch(data, CONFIG.BATCH_SIZE)
                    feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

                    _, loss_train, summary_logs = session.run([optimizer, cost, merged], feed_dict=feed_dict)
                    average_loss += loss_train
                    if step % 500:
                        train_writer.add_summary(summary_logs, step)

                    if step % 5000 == 0:
                        if step > 0:
                            average_loss /= 5000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average train loss at step ', step, ': ', average_loss)
                        local_logger.info(f"Average train loss at step {step}: {average_loss}")
                        average_loss = 0
                final_embeddings = normalized_embeddings.eval()
                return final_embeddings
    #     Run
    start_time = time.time()
    if use_cache:
        try:
            with open(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "Embeddings.pkl",
                      "rb") as file:
                local_logger.info("Loading Embeddings from cache... wait...")
                fsn_embs = pickle.load(file)
                print("Loaded Embeddings from cache!")
                return fsn_embs
        except FileNotFoundError:
            local_logger.info("File not found... Recalculate \n")
            pass
        except Exception as e:
            local_logger.error(f"Unexpected error: {e}")
    try:
        embds = run(tf_graph, skip_grams)
    except tf.errors.InvalidArgumentError as error:
        logging.getLogger("GlobalLogs.SkipGram").critical(f"Could not run TF model... , {error}")
        logging.getLogger(f"{__name__}").critical(f"Could not run TF model... , {error}")

    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    fsn_embs = pd.DataFrame(list(zip(tr.original_bps, embds)), columns=["ID", "Emb"])
    if use_dim_reduction:
        fsn_embs = dim_reduction(fsn_embs)
    if use_cache:
        fsn_embs.to_pickle(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "Embeddings.pkl")
    print("Done with TensorFlow!")
    print("Use the following command to see the Tensorboard with all collected stats during last running: \n")
    print(f"tensorboard --logdir={CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2]}")
    local_logger.info(
        "Use the following command to see the Tensorboard with all collected stats during last running: \n"
        f"tensorboard --logdir={CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2]}")
    return fsn_embs
