import tensorflow as tf
import numpy as np


class Transformer:
    def __init__(self, sequence_length, source_vocab_size, target_vocab_size,
                 hidden_size, ff_hidden_size, num_stack, num_head):
        self.sequence_length = sequence_length
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.ff_hidden_size = ff_hidden_size
        self.num_stack = num_stack
        self.num_head = num_head

        self.build_graph()



    def build_graph(self):
        # Placeholders for input, output and dropout
        self.input_source = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_source')
        self.input_target = tf.placeholder(tf.int32, shape=[None, self.sequence_length], name='input_target')

        initializer = tf.contrib.layers.xavier_initializer()

        with tf.name_scope("encoder"):
            # Embeddings
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W_embedding = tf.get_variable("W_embedding", dtype=tf.float32,
                                                   shape=[self.source_vocab_size, self.hidden_size],
                                                   initializer=tf.contrib.layers.xavier_initializer())
                self.embedded_chars = tf.nn.embedding_lookup(self.W_embedding, self.input_source)


            with tf.name_scope("positional-encoding"):
                # First part of the PE function: sin and cos argument
                self.position_enc = np.array([[pos / (10000 ** (2*i/self.hidden_size))
                                               for i in range(self.hidden_size)]
                                              for pos in range(self.sequence_length)])
                # Second part, apply the cosine to even columns and sin to odds.
                self.position_enc[:, 0::2] = np.sin(self.position_enc[:, 0::2])  # dim 2i
                self.position_enc[:, 1::2] = np.cos(self.position_enc[:, 1::2])  # dim 2i+1
                self.position_enc = tf.convert_to_tensor(self.position_enc, dtype=tf.float32)

                self.enc = self.embedded_chars + self.position_enc


            self.mh = self.multihead_attention(self.enc, self.enc, self.enc)

            for i in range(self.num_stack):
                with tf.name_scope("stacked-layer-{}".format(i)):
                    # Multihead Attention
                    self.enc = self.multihead_attention(query=self.enc,
                                                        key=self.enc,
                                                        value=self.enc)

                    # Feed Forward
                    self.enc = self.feedforward(self.enc, inner_hidden_size=self.ff_hidden_size)



        with tf.name_scope("decoder"):
            pass




    def scaled_dot_product_attention(self, query, key, value, scaling_factor):
        QK_T = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        attention = tf.nn.softmax(QK_T * scaling_factor)
        att_V = tf.matmul(attention, value)
        return att_V

    def multihead_attention(self, query, key, value):
        attentions = []
        for i in range(self.num_head):
            key_dim = int(self.hidden_size / self.num_head)

            W_q = tf.Variable(tf.random_uniform([self.hidden_size, key_dim], -1.0, 1.0))
            b_q = tf.Variable(tf.constant(0.1, shape=[key_dim]))
            Q = tf.tensordot(query, W_q, [[2], [0]]) + b_q

            W_k = tf.Variable(tf.random_uniform([self.hidden_size, key_dim], -1.0, 1.0))
            b_k = tf.Variable(tf.constant(0.1, shape=[key_dim]))
            K = tf.tensordot(key, W_k, [[2], [0]]) + b_k

            W_v = tf.Variable(tf.random_uniform([self.hidden_size, key_dim], -1.0, 1.0))
            b_v = tf.Variable(tf.constant(0.1, shape=[key_dim]))
            V = tf.tensordot(value, W_v, [[2], [0]]) + b_v

            dot_att = self.scaled_dot_product_attention(Q, K, V, scaling_factor=tf.sqrt(self.num_head / self.hidden_size))

            attentions.append(dot_att)

        att_concat = tf.concat(attentions, axis=2)

        W_att = tf.Variable(tf.random_uniform([self.hidden_size, self.hidden_size], -1.0, 1.0))
        b_att = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
        output = tf.tensordot(att_concat, W_att, [[2], [0]]) + b_att

        return output


    def feedforward(self, x, inner_hidden_size):
        W1 = tf.Variable(tf.truncated_normal([1, self.hidden_size, self.ff_hidden_size], stddev=0.1), name="W")
        conv = tf.nn.conv1d(x, W1, stride=1, padding='VALID')
        W2 = tf.Variable(tf.truncated_normal([1, self.ff_hidden_size, self.hidden_size], stddev=0.1), name="W")
        conv2 = tf.nn.conv1d(conv, W2, stride=1, padding='VALID')


    def layer_normalize(self):
        # conv2 + enc after position_enc
        pass




        # # Bidirectional(Left&Right) Recurrent Structure
        # with tf.name_scope("bi-lstm"):
        #     fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        #     bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        #     (self.output_fw, self.output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
        #                                                                                cell_bw=bw_cell,
        #                                                                                inputs=self.embedded_chars,
        #                                                                                sequence_length=text_length,
        #                                                                                dtype=tf.float32)
        #     self.H = tf.concat([self.output_fw, self.output_bw], axis=2)
        #     H_reshape = tf.reshape(self.H, [-1, 2 * hidden_size])
        #
        # with tf.name_scope("self-attention"):
        #     self.W_s1 = tf.get_variable("W_s1", shape=[2*hidden_size, d_a_size], initializer=initializer)
        #     _H_s1 = tf.nn.tanh(tf.matmul(H_reshape, self.W_s1))
        #     self.W_s2 = tf.get_variable("W_s2", shape=[d_a_size, r_size], initializer=initializer)
        #     _H_s2 = tf.matmul(_H_s1, self.W_s2)
        #     _H_s2_reshape = tf.transpose(tf.reshape(_H_s2, [-1, sequence_length, r_size]), [0, 2, 1])
        #     self.A = tf.nn.softmax(_H_s2_reshape, name="attention")
        #
        # with tf.name_scope("sentence-embedding"):
        #     self.M = tf.matmul(self.A, self.H)
        #
        # with tf.name_scope("fully-connected"):
        #     # self.M_pool = tf.reduce_mean(self.M, axis=1)
        #     # W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size, fc_size], initializer=initializer)
        #     self.M_flat = tf.reshape(self.M, shape=[-1, 2 * hidden_size * r_size])
        #     W_fc = tf.get_variable("W_fc", shape=[2 * hidden_size * r_size, fc_size], initializer=initializer)
        #     b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]), name="b_fc")
        #     self.fc = tf.nn.relu(tf.nn.xw_plus_b(self.M_flat, W_fc, b_fc), name="fc")
        #
        # with tf.name_scope("output"):
        #     W_output = tf.get_variable("W_output", shape=[fc_size, num_classes], initializer=initializer)
        #     b_output = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_output")
        #     self.logits = tf.nn.xw_plus_b(self.fc, W_output, b_output, name="logits")
        #     self.predictions = tf.argmax(self.logits, 1, name="predictions")
        #
        # with tf.name_scope("penalization"):
        #     self.AA_T = tf.matmul(self.A, tf.transpose(self.A, [0, 2, 1]))
        #     self.I = tf.reshape(tf.tile(tf.eye(r_size), [tf.shape(self.A)[0], 1]), [-1, r_size, r_size])
        #     self.P = tf.square(tf.norm(self.AA_T - self.I, axis=[-2, -1], ord="fro"))
        #
        # # Calculate mean cross-entropy loss
        # with tf.name_scope("loss"):
        #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        #     self.loss_P = tf.reduce_mean(self.P * p_coef)
        #     self.loss = tf.reduce_mean(losses) + self.loss_P
        #
        # # Accuracy
        # with tf.name_scope("accuracy"):
        #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, axis=1))
        #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
