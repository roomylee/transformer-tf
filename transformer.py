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
            with tf.device('/cpu:0'), tf.name_scope("source_embedding"):
                self.W_source_embedding = tf.get_variable("W_source_embedding", dtype=tf.float32,
                                                          shape=[self.source_vocab_size, self.hidden_size],
                                                          initializer=initializer)
                self.source_embedded_chars = tf.nn.embedding_lookup(self.W_source_embedding, self.input_source)


            with tf.name_scope("positional-encoding"):
                self.enc = self.source_embedded_chars + self.position_encoding()

            for i in range(self.num_stack):
                with tf.name_scope("encoder-block-{}".format(i)):
                    # Multihead Attention (self attention)
                    self.mh = self.multihead_attention(query=self.enc,
                                                       key=self.enc,
                                                       value=self.enc)
                    self.mh = tf.contrib.layers.layer_norm(self.enc + self.mh)

                    # Position-wise Feed Forward
                    self.ff = self.feedforward(self.mh)
                    self.enc = tf.contrib.layers.layer_norm(self.mh + self.ff)

        with tf.name_scope("decoder"):
            # Embeddings
            with tf.device('/cpu:0'), tf.name_scope("target_embedding"):
                self.W_target_embedding = tf.get_variable("W_target_embedding", dtype=tf.float32,
                                                          shape=[self.target_vocab_size, self.hidden_size],
                                                          initializer=initializer)
                self.target_embedded_chars = tf.nn.embedding_lookup(self.W_target_embedding, self.input_target)

            with tf.name_scope("positional-encoding"):
                self.dec = self.target_embedded_chars + self.position_encoding()

            for i in range(self.num_stack):
                with tf.name_scope("decoder-block-{}".format(i)):
                    # Masked Multihead Attention (self attention)
                    self.mask_mh = self.multihead_attention(query=self.dec,
                                                            key=self.dec,
                                                            value=self.dec,
                                                            masked=True)
                    self.mask_mh = tf.contrib.layers.layer_norm(self.dec + self.mask_mh)

                    # Multihead Attention (attention with encoder)
                    self.mh = self.multihead_attention(query=self.mask_mh,
                                                       key=self.enc,
                                                       value=self.enc)
                    self.mh = tf.contrib.layers.layer_norm(self.mask_mh + self.mh)

                    # Position-wise Feed Forward
                    self.ff = self.feedforward(self.mh)
                    self.dec = tf.contrib.layers.layer_norm(self.mh + self.ff)


    def position_encoding(self):
        # First part of the PE function: sin and cos argument
        position_enc = np.array([[pos / (10000 ** (2 * i / self.hidden_size))
                                       for i in range(self.hidden_size)]
                                      for pos in range(self.sequence_length)])
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, dtype=tf.float32)
        return position_enc

    def scaled_dot_product_attention(self, query, key, value, scaling_factor, masked):
        QK_T = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
        if masked:
            mask = tf.ones([self.hidden_size, self.hidden_size])
            mask = tf.contrib.linalg.LinearOperatorTriL(mask, tf.float32).to_dense()
            QK_T = tf.matmul(QK_T, mask)
        attention = tf.nn.softmax(QK_T * scaling_factor)
        att_V = tf.matmul(attention, value)
        return att_V

    def multihead_attention(self, query, key, value, masked=False):
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

            dot_att = self.scaled_dot_product_attention(Q, K, V,
                                                        scaling_factor=tf.sqrt(self.num_head / self.hidden_size),
                                                        masked=masked)

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
        h2 = tf.nn.bias_add(conv2, b2)

        return h2
