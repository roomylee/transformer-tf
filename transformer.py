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

                self.temp = self.enc

            for i in range(self.num_stack):
                with tf.name_scope("stacked-layer-{}".format(i)):
                    # Multihead Attention (self attention)
                    self.h_att = self.multihead_attention(query=self.enc,
                                                          key=self.enc,
                                                          value=self.enc)
                    # Position-wise Feed Forward
                    self.h_ff = self.feedforward(self.enc)

                    self.enc = tf.contrib.layers.layer_norm(self.enc + self.h_ff)

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


    def feedforward(self, x):
        W1 = tf.Variable(tf.truncated_normal([1, self.hidden_size, self.ff_hidden_size], stddev=0.1))
        b1 = tf.Variable(tf.constant(0.1, shape=[self.ff_hidden_size]))
        conv1 = tf.nn.conv1d(x, W1, stride=1, padding='VALID')
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))

        W2 = tf.Variable(tf.truncated_normal([1, self.ff_hidden_size, self.hidden_size], stddev=0.1))
        b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size]))
        conv2 = tf.nn.conv1d(h1, W2, stride=1, padding='VALID')
        h2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

        return h2
