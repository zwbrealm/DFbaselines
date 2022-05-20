# coding=utf-8
import tensorflow.compat.v1 as tf
from stackedRNN import stack_bidirectional_dynamic_rnn
tf.compat.v1.disable_eager_execution()
import tensorflow

class Fs_net(tensorflow.keras.Model):
    def __init__(self, x, y):
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.vocab_size = 128
        self.n_neurons = 128
        self.encoder_n_neurons = 128
        self.decoder_n_neurons = self.vocab_size
        self.n_layers = 2
        self.n_steps = 256
        self.n_inputs = 128
        self.n_outputs = 25
        # self.X = tf.compat.v1.disable_v2_behavior(tf.float32, [None, self.n_steps, self.n_inputs])
        # self.Y = tf.compat.v1.disable_v2_behavior(tf.float32, [None,1])
        self.X = x
        self.Y = y
        self.alpha = 1


    def bi_gru(self, x, name, n_neurons):
        with tf.variable_scope(name_or_scope=name, reuse=False):
            fw_cell_list = [tf.keras.layers.GRUCell(n_neurons) for i in range(2)]
            bw_cell_list = [tf.keras.layers.GRUCell(n_neurons) for i in range(2)]
            outputs, fw_states, bw_states = stack_bidirectional_dynamic_rnn(fw_cell_list, bw_cell_list,
                                                                                           x, dtype=tf.float32)
        return outputs, fw_states, bw_states


    def tinny_fs_net(self):
        # no embedding
        _, encoder_fw_states, encoder_bw_states = self.bi_gru(self.X, "stack_encode_bi_gru", self.encoder_n_neurons)
        encoder_feats = tf.concat([encoder_fw_states[-1], encoder_bw_states[-1]],
                                  axis=-1)  # [batch_size,2*self.encoder_n_neurons]
        encoder_expand_feats = tf.expand_dims(encoder_feats, axis=1)
        decoder_input = tf.tile(encoder_expand_feats,
                                [1, self.n_steps, 1])  # [batch_size,self.n_steps,2*self.encoder_n_neurons]
        decoder_output, decoder_fw_states, decoder_bw_states = self.bi_gru(decoder_input, "stack_decode_bi_gru",
                                                                           self.decoder_n_neurons)
        decoder_feats = tf.concat([decoder_fw_states[-1], decoder_bw_states[-1]],
                                  axis=-1)  # [batch_size,2*self.decoder_n_neurons]
        element_wise_product = encoder_feats * decoder_feats
        element_wise_absolute = tf.abs(encoder_feats - decoder_feats)
        cls_feats = tf.concat([encoder_feats, decoder_feats, element_wise_product, element_wise_absolute], axis=-1)
        cls_dense_1 = tf.layers.dense(inputs=cls_feats, units=self.n_neurons, activation=tf.nn.selu,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.002))
        cls_dense_2 = tf.layers.dense(inputs=cls_dense_1, units=self.n_outputs, activation=tf.nn.sigmoid,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.002), name="sigmoid")
        cls_dense_2 = tf.cast(cls_dense_2, dtype=tf.float32, name=None)
        return cls_dense_2, decoder_output

    def fs_net(self):
        # add embedding
        embeddings = tf.get_variables('weight_mat', dtype=tf.float32, shape=(self.vocab_size, self.embedding_dim))
        x_embedding = tf.nn.embedding_lookup(embeddings, self.X)
        _, encoder_fw_states, encoder_bw_states = self.bi_gru(x_embedding, "stack_encode_bi_gru", self.encoder_n_neurons)
        encoder_feats = tf.concat([encoder_fw_states[-1], encoder_bw_states[-1]],
                                  axis=-1)  # [batch_size,2*self.encoder_n_neurons]
        encoder_expand_feats = tf.expand_dims(encoder_feats, axis=1)
        decoder_input = tf.tile(encoder_expand_feats,
                                [1, self.n_steps, 1])  # [batch_size,self.n_steps,2*self.encoder_n_neurons]
        decoder_output, decoder_fw_states, decoder_bw_states = self.bi_gru(decoder_input, "stack_decode_bi_gru",
                                                                           self.decoder_n_neurons)
        decoder_feats = tf.concat([decoder_fw_states[-1], decoder_bw_states[-1]],
                                  axis=-1)  # [batch_size,2*self.decoder_n_neurons]
        element_wise_product = encoder_feats * decoder_feats
        element_wise_absolute = tf.abs(encoder_feats - decoder_feats)
        cls_feats = tf.concat([encoder_feats, decoder_feats, element_wise_product, element_wise_absolute], axis=-1)
        cls_dense_1 = tf.layers.dense(inputs=cls_feats, units=self.n_neurons, activation=tf.nn.selu,
                                      kernel_regularizer=tf.keras.layers.l2_regularizer(0.003))
        cls_dense_2 = tf.layers.dense(inputs=cls_dense_1, units=self.n_outputs, activation=tf.nn.sigmoid,
                                      kernel_regularizer=tf.keras.layers.l2_regularizer(0.003), name="sigmoid")
        tf.cast(cls_dense_2, dtype=tf.float32, name=None)
        return cls_dense_2, decoder_output

#     def build_loss(self):
#         logits, ae_outputs = self.fs_net()
#         self.Y = tf.cast(self.Y, dtype = tf.float32)
#         logits = tf.cast(logits, dtype=tf.float32)

#         cls_entropy = tensorflow.losses.binary_crossentropy(self.Y, logits)
#         cls_loss = tf.reduce_mean(cls_entropy, name="cls_loss")
#         ae_loss = 0
#         total_loss = cls_loss + self.alpha * ae_loss
#         return total_loss, logits

    def build_fs_net_loss(self):
        logits, ae_outputs = self.fs_net()
        # self.X  [batch_size,n_steps,vocab_size] (one-hot)
        # ae_outputs [batch_size,n_steps,decoder_n_neurons] (vocab_size=decoder_n_neurons)
        cls_entropy = tensorflow.losses.binary_crossentropy(self.Y, logits)
        cls_loss = tf.reduce_mean(cls_entropy, name="cls_loss")
        ae_loss = tf.nn.sparse_softmax_cross_entrop_with_logits(labels=self.X, logits=ae_outputs)
        total_loss = cls_loss + self.alpha * ae_loss
        return total_loss
