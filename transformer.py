import tensorflow as tf
import numpy as np


class Transformer:
    def __init__(self, sequence_length, source_vocab_size, target_vocab_size,
                 dim_model, dim_ff, num_stack, num_head):

        # Placeholders for Encoder Input (= Source Sentence)
        self.encoder_x = tf.placeholder(tf.int32, shape=[None, sequence_length], name='encoder_x')
        # Placeholders for Decoder Output (= Target Sentence)
        self.decoder_y = tf.placeholder(tf.int32, shape=[None, sequence_length], name='decoder_y')
        # Decoder Input (right shifted target sentence; first word is _START_, index=1)
        self.decoder_x = tf.concat((tf.ones_like(self.decoder_y[:, :1]), self.decoder_y[:, :-1]), -1, name='decoder_x')

        initializer = tf.contrib.layers.xavier_initializer()

        with tf.variable_scope("encoder"):
            # Embeddings
            with tf.device('/cpu:0'), tf.variable_scope("source-embedding"):
                self.W_source_embedding = tf.get_variable("W", shape=[source_vocab_size, dim_model],
                                                          dtype=tf.float32, initializer=initializer)
                self.source_embedded_chars = tf.nn.embedding_lookup(self.W_source_embedding, self.encoder_x)

            # Positional Encoding
            with tf.variable_scope("positional-encoding"):
                self.pos_enc = self.position_encoding(sequence_length, dim_model)
                self.enc = tf.add(self.source_embedded_chars, self.pos_enc, name="encoder_input")

            for i in range(num_stack):
                with tf.variable_scope("block-{}".format(i)):
                    # Multi-head Attention (self attention)
                    with tf.variable_scope("multihead-attention"):
                        self.mh = self.multihead_attention(query=self.enc, key=self.enc, value=self.enc,
                                                           dim_model=dim_model, num_head=num_head)
                        # Residual & Layer Normalization
                        self.mh = tf.contrib.layers.layer_norm(self.enc + self.mh)

                    # Position-wise Feed Forward
                    with tf.variable_scope("position-wise-feed-forward"):
                        self.ff = self.feedforward(self.mh, dim_model, dim_ff)
                        # Residual & Layer Normalization
                        self.enc = tf.contrib.layers.layer_norm(self.mh + self.ff)

        with tf.variable_scope("decoder"):
            with tf.device('/cpu:0'), tf.variable_scope("target-embedding"):
                self.W_target_embedding = tf.get_variable("W", shape=[target_vocab_size, dim_model],
                                                          dtype=tf.float32, initializer=initializer)
                self.target_embedded_chars = tf.nn.embedding_lookup(self.W_target_embedding, self.decoder_x)

            # Position Encoding
            with tf.variable_scope("positional-encoding"):
                self.pos_enc = self.position_encoding(sequence_length, dim_model)
                self.dec = tf.add(self.target_embedded_chars, self.pos_enc, name="decoder_input")

            for i in range(num_stack):
                with tf.variable_scope("block-{}".format(i)):
                    # Masked Multi-head Attention (self attention)
                    with tf.variable_scope("masked-multihead-attention"):
                        self.mask_mh = self.multihead_attention(query=self.dec, key=self.dec, value=self.dec,
                                                                dim_model=dim_model, num_head=num_head, masked=True)
                        # Residual & Layer Normalization
                        self.mask_mh = tf.contrib.layers.layer_norm(self.dec + self.mask_mh)

                    # Multi-head Attention (attention with encoder)
                    with tf.variable_scope("multihead-attention"):
                        self.mh = self.multihead_attention(query=self.mask_mh, key=self.enc, value=self.enc,
                                                           dim_model=dim_model, num_head=num_head)
                        # Residual & Layer Normalization
                        self.mh = tf.contrib.layers.layer_norm(self.mask_mh + self.mh)

                    # Position-wise Feed Forward
                    with tf.variable_scope("position-wise-feed-forward"):
                        self.ff = self.feedforward(self.mh, dim_model, dim_ff)
                        # Residual & Layer Normalization
                        self.dec = tf.contrib.layers.layer_norm(self.mh + self.ff)

        # Output
        with tf.variable_scope("output"):
            self.logits = tf.layers.dense(self.dec, target_vocab_size, activation=tf.nn.relu, name="logits")
            self.predictions = tf.to_int32(tf.argmax(self.logits, axis=2, name="predictions"))

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            target_one_hot = tf.one_hot(self.decoder_y, depth=target_vocab_size)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=target_one_hot)
            is_target = tf.to_float(tf.not_equal(self.decoder_y, 0))
            self.loss = tf.reduce_sum(losses*is_target) / tf.reduce_sum(is_target)

        # Accuracy
        with tf.variable_scope("accuracy"):
            self.accuracy = tf.reduce_sum(tf.to_float(tf.equal(self.predictions, self.decoder_y)) * is_target)\
                            / tf.reduce_sum(is_target)

    @ staticmethod
    def position_encoding(sequence_length, dim_model):
        # First part of the PE function: sin and cos argument
        position_enc = np.array([[pos / (10000 ** (2 * i / dim_model)) for i in range(dim_model)] for pos in range(sequence_length)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        output = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        return output

    @staticmethod
    def multihead_attention(query, key, value, dim_model, num_head, masked=False):
        attentions = []
        for i in range(num_head):
            dim_k = int(dim_model / num_head)
            dim_v = dim_k

            Q = tf.layers.dense(query, dim_k, activation=tf.nn.relu)
            K = tf.layers.dense(key, dim_k, activation=tf.nn.relu)
            V = tf.layers.dense(value, dim_v, activation=tf.nn.relu)

            # Scaled Dot Product Attention
            with tf.variable_scope("scaled-dot-product-attention"):
                QK_T = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
                if masked:
                    mask = tf.ones_like(QK_T)
                    mask = tf.contrib.linalg.LinearOperatorTriL(mask, tf.float32).to_dense()
                    # Tensorflow >= 1.5.0
                    # mask = tf.linalg.LinearOperatorLowerTriangular(mask, tf.float32).to_dense()
                    QK_T = tf.matmul(QK_T, mask)
                attention = tf.nn.softmax(QK_T * tf.sqrt(1/dim_k))
                att_V = tf.matmul(attention, V)

            attentions.append(att_V)

        att_concat = tf.concat(attentions, axis=2)
        output = tf.layers.dense(att_concat, dim_model, activation=tf.nn.relu)

        return output

    @staticmethod
    def feedforward(x, dim_model, dim_ff):
        # First Convolution
        output = tf.layers.conv1d(inputs=x, filters=dim_ff, kernel_size=1, activation=tf.nn.relu)
        # Second Convolution
        output = tf.layers.conv1d(inputs=output, filters=dim_model, kernel_size=1, activation=tf.nn.relu)

        return output
