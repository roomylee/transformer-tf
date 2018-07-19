import tensorflow as tf
import numpy as np
import os
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import data_helpers


# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_string("test_source_dir", "corpora/IWSLT16.TED.tst2014.de-en.de.xml", "Path of test source data")
tf.flags.DEFINE_string("test_target_dir", "corpora/IWSLT16.TED.tst2014.de-en.en.xml", "Path of test target data")
tf.flags.DEFINE_string("output_dir", "results/translation_result.txt", "Path of translation results")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

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
            source = graph.get_operation_by_name("encoder_x").outputs[0]
            target = graph.get_operation_by_name("decoder_y").outputs[0]

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
            print("Total number of test examples: {}\n".format(len(target_eval)))
            print("Accuracy: {:g}".format(accuracy))


            prediction_sent = []
            for idx_seq in all_predictions:
                prediction_sent.append(" ".join(target_vocab_processor.vocabulary_.reverse(idx) for idx in idx_seq))

            # BLEU Score
            list_of_references = []
            hypotheses = []
            for pred, target in zip(prediction_sent, target_sent):
                if len(pred.split()) > 3 and len(target.split()) > 3:
                    list_of_references.append([pred.split()])
                    hypotheses.append(target.split())
            chencherry = SmoothingFunction()
            score = corpus_bleu(list_of_references, hypotheses, smoothing_function=chencherry.method4)
            print("BLEU Score : {:g}\n".format(score*100))

            # Samples of Translation Result
            if not os.path.exists('results'): os.mkdir('results')
            f = open(FLAGS.output_dir, 'w')
            for idx, (s, t, p) in enumerate(zip(source_sent, target_sent, prediction_sent)):
                f.write("Sample #%d\n" % idx)
                f.write("Source : %s\n" % s)
                f.write("Target : %s\n" % t)
                f.write("Predict : %s\n\n" % p)
            f.close()


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()