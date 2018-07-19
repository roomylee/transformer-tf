import tensorflow as tf
import numpy as np
import os
import datetime
import time
from transformer import Transformer
import data_helpers


# Parameters
# ==================================================


# Data loading params
tf.flags.DEFINE_string("train_source_dir", "corpora/train.tags.de-en.de", "Path of train source data")
tf.flags.DEFINE_string("train_target_dir", "corpora/train.tags.de-en.en", "Path of train target data")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("source_max_sentence_length", 10, "Max sentence length in source data")
tf.flags.DEFINE_integer("target_max_sentence_length", 10, "Max sentence length in target data")

# Model Hyperparameters
tf.flags.DEFINE_integer("dim_model", 512, "Dimension of Model & Embedding (d_model in paper)")
tf.flags.DEFINE_integer("dim_ff", 2048, "Dimension of Hidden Layer of Feed Forward Network (d_ff in paper)")
tf.flags.DEFINE_integer("num_stack", 6, "Number of Stacked Encoder/Decoder Block (N in paper)")
tf.flags.DEFINE_integer("num_head", 8, "Number of linear projection in Multi-Head Attention (h in paper)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with.")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS


def train():
    with tf.device('/cpu:0'):
        source_sent, target_sent = data_helpers.load_train_data(FLAGS.train_source_dir,
                                                                FLAGS.train_target_dir,
                                                                FLAGS.source_max_sentence_length,
                                                                FLAGS.target_max_sentence_length)

    # Build vocabulary
    # Example: x_text[3] = "A misty ridge uprises from the surge."
    # ['a misty ridge uprises from the surge __EOS__ __UNK__ ... __UNK__']
    # =>
    # [27 39 40 41 42  1 43  0  0 ... 0]
    # dimension = FLAGS.max_sentence_length
    source_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.source_max_sentence_length)
    x = np.array(list(source_vocab_processor.fit_transform(["_START_ _EOS_ _PAD_"] + source_sent)))
    print("Source Language Vocabulary Size: {:d}".format(len(source_vocab_processor.vocabulary_)))

    target_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.target_max_sentence_length)
    y = np.array(list(target_vocab_processor.fit_transform(["_START_ _EOS_ _PAD_"] + target_sent)))
    print("Target Language Vocabulary Size: {:d}".format(len(target_vocab_processor.vocabulary_)))

    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))
    print("")

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split corpora/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Transformer(
                sequence_length=x_train.shape[1],
                source_vocab_size=len(source_vocab_processor.vocabulary_),
                target_vocab_size=len(target_vocab_processor.vocabulary_),
                dim_model=FLAGS.dim_model,
                dim_ff=FLAGS.dim_ff,
                num_stack=FLAGS.num_stack,
                num_head=FLAGS.num_head
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            source_vocab_processor.save(os.path.join(out_dir, "source_vocab"))
            target_vocab_processor.save(os.path.join(out_dir, "target_vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                # Train
                feed_dict = {
                    model.encoder_x: x_batch,
                    model.decoder_y: y_batch
                }

                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy], feed_dict)
                train_summary_writer.add_summary(summaries, step)

                # Training log display
                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                # Evaluation
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    # Generate batches
                    batches_dev = data_helpers.batch_iter(
                        list(zip(x_dev, y_dev)), FLAGS.batch_size, 1)
                    # Evaluation loop. For each batch...
                    loss_dev = 0
                    accuracy_dev = 0
                    cnt = 0
                    for batch_dev in batches_dev:
                        x_batch_dev, y_batch_dev = zip(*batch_dev)

                        feed_dict_dev = {
                            model.encoder_x: x_batch_dev,
                            model.decoder_y: y_batch_dev
                        }

                        summaries_dev, loss, accuracy = sess.run(
                            [dev_summary_op, model.loss, model.accuracy], feed_dict_dev)
                        dev_summary_writer.add_summary(summaries_dev, step)

                        loss_dev += loss
                        accuracy_dev += accuracy
                        cnt += 1

                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss_dev / cnt, accuracy_dev / cnt))

                # Model checkpoint
                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
