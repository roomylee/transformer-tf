import tensorflow as tf
import numpy as np
import os
from nltk.translate.bleu_score import corpus_bleu
import data_helpers

import nsml


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("nsml_test_source_dir", nsml.DATASET_PATH + "/test/IWSLT16.TED.tst2014.de-en.de.xml", "Path of corpora data")
tf.flags.DEFINE_string("nsml_test_target_dir",  nsml.DATASET_PATH + "/test/IWSLT16.TED.tst2014.de-en.en.xml", "Path of corpora data")
tf.flags.DEFINE_string("test_source_dir", "corpora/IWSLT16.TED.tst2014.de-en.de.xml", "Path of corpora data")
tf.flags.DEFINE_string("test_target_dir", "corpora/IWSLT16.TED.tst2014.de-en.en.xml", "Path of corpora data")

# Eval Parameters
tf.flags.DEFINE_boolean("nsml", False, "training by NSML")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1531899424/checkpoints/", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS


def eval():
    # Map data into vocabulary
    source_vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "source_vocab")
    source_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(source_vocab_path)
    source_max_sentence_length = len(list(source_vocab_processor.transform(['test']))[0])
    target_vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "target_vocab")
    target_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(target_vocab_path)
    target_max_sentence_length = len(list(target_vocab_processor.transform(['test']))[0])

    with tf.device('/cpu:0'):
        if FLAGS.nsml:
            source_sent, target_sent = data_helpers.load_test_data(FLAGS.nsml_test_source_dir,
                                                                   FLAGS.nsml_test_target_dir,
                                                                   source_max_sentence_length,
                                                                   target_max_sentence_length)
        else:
            source_sent, target_sent = data_helpers.load_test_data(FLAGS.test_source_dir,
                                                                   FLAGS.test_target_dir,
                                                                   source_max_sentence_length,
                                                                   target_max_sentence_length)

    source_eval = np.array(list(source_vocab_processor.transform(source_sent)))
    target_eval = np.array(list(target_vocab_processor.transform(target_sent)))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            source = graph.get_operation_by_name("source").outputs[0]
            target = graph.get_operation_by_name("target").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(source_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = np.empty([0, target_max_sentence_length], int)
            for batch in batches:
                # auto-regressive infer
                batch_predictions = np.ones_like(batch)
                for j in range(target_max_sentence_length):
                    pred = sess.run(predictions, feed_dict={source: batch, target: batch_predictions})
                    batch_predictions[:, j] = pred[:, j]

                all_predictions = np.concatenate([all_predictions, batch_predictions])

            is_target = np.not_equal(target_eval, 0).astype(float)
            accuracy = np.sum(np.equal(all_predictions, target_eval).astype(float) * is_target) / np.sum(is_target)
            print("Total number of test examples: {}".format(len(target_eval)))
            print("Accuracy: {:g}".format(accuracy))

            # BLEU Score
            preds = [[target_vocab_processor.vocabulary_.reverse(idx) for idx in sent] for sent in all_predictions]
            origins = [[sent.split()] for sent in target_sent]
            score = corpus_bleu(list_of_references=origins, hypotheses=preds)
            print("BLEU Score :", score*100)

            # Samples of Translation Result
            random_idx = np.random.randint(len(target_vocab_processor.vocabulary_), 5)
            for idx in random_idx:
                print("Sample #", idx)
                print("Source :", source_sent[idx])
                print("Target :", target_sent[idx])
                pred = " ".join(target_vocab_processor.vocabulary_.reverse(word_idx)
                                for word_idx in all_predictions[idx])
                print("Predict :", pred)


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()