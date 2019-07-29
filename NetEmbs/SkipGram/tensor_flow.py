# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
tensor_flow.py
Created by lex at 2019-04-12.
"""
import os
import tensorflow as tf
import math
from NetEmbs.SkipGram.generate_batch import generate_batch
from NetEmbs.DataProcessing import *
from NetEmbs.FSN.utils import get_SkipGrams
import time
import pandas as pd
import numpy as np
from NetEmbs.utils.IO.db_connection import upload_journal_entries, upload_data
from NetEmbs.CONFIG import MODE, LOG_LEVEL
from NetEmbs import CONFIG
from NetEmbs.Vis.plots import plot_tSNE, plot_PCA
import logging

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_embs_TF(input_data=None, step_version=None, embed_size=None,
                num_steps=None, batch_size=None, walk_length=None,
                walks_per_node=None, use_cached_skip_grams=True, use_prev_embs=False, vis_progress=False,
                groundTruthDF=None, evaluate_time=False):
    """
    Wrapper around all representation stuff for FSN
    :param input_data: Input DataFrame with journal entries
    :param step_version: Version of step passed into GraphSampling method
        "DefUniform" - Pure RandomWalk (uniform probabilities, follows the direction),
        "DefWeighted" - RandomWalk (weighted probabilities, follows the direction),
        "MetaUniform" - Default Metapath-version (uniform probabilities, change directions),
        "MetaWeighted" - Weighted Metapath version (weighted probabilities "rich gets richer", change directions),
        "MetaDiff" - Modified Metapath version (probabilities depend on the differences between edges, change directions)
    :param embed_size: Dimensionality of embedding space
    :param num_steps: Number of training steps for TF model
    :param batch_size: Number of sampled in batches for training TF model
    :param walk_length: Walk length for sampling strategy
    :param walks_per_node: Number of finWalks per each BP node in FSN
    :param use_cached_skip_grams: If True, use previously saved skip_grams. Default is True
    :param use_prev_embs: If True, use previously calculated embeddings for tensor initialisation. Default is False
    :param vis_progress: Integer, plot tSNE after specified number of steps during TF model training. Not recommend,
    extremely expensive. Default is False, hence, no drawing at all.
    :param groundTruthDF: DataFrame with the ground truth column. Required if vis_progress is not False
    :param evaluate_time: If True, then method additionally evaluate elapsed times for Sampling and TF training and returns is as Dictionary
    :return: DataFrame with normalized embeddings
    """
    if input_data is None and CONFIG.GLOBAL_FSN is not None:
        print(f"Found FSN object in memory!")
    # Check the input argument type: FSN or DataFrame
    elif isinstance(input_data, pd.DataFrame):
        # #     Construct FSN object from the given df
        d = input_data
        print("Total number of BPs in given dataset is ", d.ID.nunique())
    elif isinstance(input_data, tuple):
        d = prepare_data(upload_data(input_data[0], limit=input_data[1]))
        print("Total number of BPs in given dataset is ", d.ID.nunique())
    else:
        raise ValueError(
            "As input data should be DataFrame with journal entries of the path to DataBase! Was given {!s}!".format(
                type(input_data)))
    if evaluate_time:
        #     User asked for additional statistics about current execution
        run_stats = dict()
        start_time = time.time()
    skip_grams, fsn, enc_dec = get_SkipGrams(df=input_data, version=step_version,
                                             walk_length=walk_length, walks_per_node=walks_per_node,
                                             direction=None, use_cache=use_cached_skip_grams)

    local_logger = logging.getLogger("NetEmbs.SkipGram")
    local_logger.info("Initialize TF model")
    if evaluate_time:
        run_stats["Sampling time"] = time.time() - start_time
    print(skip_grams[:5])
    #
    # //////// UPDATE CONFIG IF NEEDED w.r.t the given arguments \\\\\\\\\\\
    if embed_size is not None:
        CONFIG.EMBD_SIZE = embed_size
    if num_steps is not None:
        CONFIG.STEPS = num_steps
    if batch_size is not None:
        CONFIG.BATCH_SIZE = batch_size
    print(f"Current TensorFlow parameters: \n Embedding size:  {CONFIG.EMBD_SIZE} \n Steps:  {CONFIG.STEPS}"
          f"\n Batch size:  {CONFIG.BATCH_SIZE}")
    logging.getLogger(CONFIG.MAIN_LOGGER + ".SkipGram").info(
        f"Current TensorFlow parameters: \n Embedding size:  {CONFIG.EMBD_SIZE} \n Steps:  {CONFIG.STEPS}"
        f"\n Batch size:  {CONFIG.BATCH_SIZE}")
    #     TensorFlow stuff
    valid_size = 4
    total_size = fsn.number_of_BP()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))

        # Input variables
        train_inputs = tf.placeholder(tf.int32, shape=[CONFIG.BATCH_SIZE])
        train_context = tf.placeholder(tf.int32, shape=[CONFIG.BATCH_SIZE, 1])

        # Embeddings matrix initialisation
        if use_prev_embs:
            print("Loading previous embeddings from cache... wait...")
            try:
                embeddings = tf.Variable(
                    pd.read_pickle(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2]
                                   + "snapshot.pkl").values)
            except FileNotFoundError:
                print("No cached embeddings, initialize as random matrix...")
                embeddings = tf.Variable(tf.random_uniform((total_size, CONFIG.EMBD_SIZE), -1.0, 1.0))
        else:
            embeddings = tf.Variable(tf.random_uniform((total_size, CONFIG.EMBD_SIZE), -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # ----
        # Output layer parameters
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal((total_size, CONFIG.EMBD_SIZE),
                                                      stddev=1.0 / math.sqrt(CONFIG.EMBD_SIZE)))
            if LOG_LEVEL == "full":
                variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros((total_size)))
            if LOG_LEVEL == "full":
                variable_summaries(biases)

        # Calculate the loss using negative sampling
        with tf.name_scope('cost'):
            if CONFIG.LOSS_FUNCTION == "NegativeSampling":
                loss = tf.nn.sampled_softmax_loss(weights, biases,
                                                  train_context, embed,
                                                  CONFIG.NEGATIVE_SAMPLES, total_size)
            elif CONFIG.LOSS_FUNCTION == "NCE":
                loss = tf.nn.nce_loss(weights, biases,
                                      train_context, embed,
                                      CONFIG.NEGATIVE_SAMPLES, total_size)

            cost = tf.reduce_mean(loss)
            tf.summary.scalar('Cost', cost)
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Validation subset of BPs
        # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent

        # valid_examples = np.random.randint(0, total_size, valid_size)
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # We use the cosine distance:
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm

        # valid_embedding = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        # similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embeddings))

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + '/train', graph)
        # Add variable initializer.
        init = tf.global_variables_initializer()

        def run(graph, data):
            with tf.Session(graph=graph) as session:
                # We must initialize all variables before we use them.
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

        def run_with_vis(graph, data, enc_dec):
            with tf.Session(graph=graph) as session:
                # We must initialize all variables before we use them.
                init.run()
                print('Initialized')

                average_loss = 0
                for chunk in range(0, CONFIG.STEPS, vis_progress):
                    for step in range(chunk, chunk + vis_progress):
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
                            average_loss = 0
                    final_embeddings = normalized_embeddings.eval()
                    fsn_embs = pd.DataFrame(list(zip(enc_dec.original_bps, final_embeddings)), columns=["ID", "Emb"])
                    # //////// Merge with GroundTruth \\\\\\\\\
                    if MODE == "SimulatedData":
                        d = add_ground_truth(fsn_embs)
                    if MODE == "RealData" and groundTruthDF is not None:
                        d = fsn_embs.merge(groundTruthDF.groupby("ID", as_index=False).agg({"GroundTruth": "first"}),
                                           on="ID")
                        #     ////////// Plotting tSNE graphs with ground truth vs. labeled \\\\\\\
                    plot_tSNE(d, legend_title="GroundTruth",
                              folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2],
                              title="progress/tSNE_GroundTruth" + str(chunk + vis_progress))
                    plot_PCA(d, legend_title="GroundTruth",
                             folder=CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[1],
                             title="progress/PCA_GroundTruth" + str(chunk + vis_progress))
                    print("Plotted the GroundTruth graph after " + str(chunk + vis_progress))
                return final_embeddings
    #     Run
    start_time = time.time()

    if not vis_progress:
        try:
            embs = run(graph, skip_grams)
        except tf.errors.InvalidArgumentError as error:
            logging.getLogger("GlobalLogs.SkipGram").critical(f"Could not run TF model... , {error}")
            logging.getLogger("NetEmbs.SkipGram").critical(f"Could not run TF model... , {error}")
    elif isinstance(vis_progress, int):
        if not os.path.exists(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "img/progress/"):
            os.makedirs(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "img/progress/",
                        exist_ok=True)
        embs = run_with_vis(graph, skip_grams, enc_dec)
    pd.DataFrame(embs).to_pickle(
        CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "snapshot.pkl")
    end_time = time.time()
    print("Elapsed time: ", end_time - start_time)
    fsn_embs = pd.DataFrame(list(zip(enc_dec.original_bps, embs)), columns=["ID", "Emb"])
    print("Done with TensorFlow!")
    print("Use the following command to see the Tensorboard with all collected stats during last running: \n")
    print("tensorboard --logdir=" + CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2])
    local_logger.info(
        "Use the following command to see the Tensorboard with all collected stats during last running: \n"
        f"tensorboard --logdir={CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2]}")
    if not evaluate_time:
        return fsn_embs
    elif evaluate_time:
        run_stats["TF time"] = end_time - start_time
        return fsn_embs, run_stats


def add_ground_truth(df, path_file="../Simulation/FSN_Data.db"):
    journal_truth = upload_journal_entries(path_file)[["ID", "GroundTruth", "Time"]]
    return df.merge(journal_truth, on="ID")
