import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self, batch_size, neg_size, entTotal, relTotal, embedding_size, weight_decay, learning_rate):
        self.batch_size = batch_size
        self.neg_size = neg_size  # neg_size == num_rollouts
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.entTotal = entTotal
        self.relTotal = relTotal
        self.embedding_size = embedding_size
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        self.input_def()
        self.embedding_def()
        self.loss_def()
        self.predict_def()

    def get_all_instance(self, in_batch=False):
        if in_batch:
            return [tf.transpose(tf.reshape(self.batch_h, [1 + self.neg_size, -1]), [1, 0]),
                    tf.transpose(tf.reshape(self.batch_t, [1 + self.neg_size, -1]), [1, 0]),
                    tf.transpose(tf.reshape(self.batch_r, [1 + self.neg_size, -1]), [1, 0])]
        else:
            return [self.batch_h, self.batch_t, self.batch_r]

    def get_all_labels(self, in_batch=False):
        if in_batch:
            return tf.transpose(tf.reshape(self.batch_y, [1 + self.neg_size, -1]), [1, 0])
        else:
            return self.batch_y

    def get_predict_instance(self):
        return [self.predict_h, self.predict_t, self.predict_r]

    def input_def(self):
        self.batch_h = tf.placeholder(tf.int32, [self.batch_size * self.neg_size, 2])
        self.batch_t = tf.placeholder(tf.int32, [self.batch_size * self.neg_size, 2])
        self.batch_r = tf.placeholder(tf.int32, [self.batch_size * self.neg_size, 2])
        self.predict_h = tf.placeholder(tf.int32, [None])
        self.predict_t = tf.placeholder(tf.int32, [None])
        self.predict_r = tf.placeholder(tf.int32, [None])

    def embedding_def(self):
        pass

    def loss_def(self):
        pass

    def predict_def(self):
        pass

            
class DistMult(Model):

    def _calc(self, h, t, r):
        return h * r * t

    def embedding_def(self):
        # TODO: align with generator, vocab?
        self.ent_embeddings = tf.get_variable(name="ent_embeddings", shape=[self.entTotal, self.embedding_size])
        self.rel_embeddings = tf.get_variable(name="rel_embeddings", shape=[self.relTotal, self.embedding_size])
        self.parameter_lists = {"ent_embeddings": self.ent_embeddings,
                                "rel_embeddings": self.rel_embeddings}
    def loss_def(self):
        # Obtaining the initial configuration of the model
        # To get positive triples and negative triples for training
        # To get labels for the triples, positive triples as 1 and negative triples as -1
        h, t, r = self.get_all_instance()
        # Embedding entities and relations of triples
        e_h = tf.nn.embedding_lookup(self.ent_embeddings, h)
        e_t = tf.nn.embedding_lookup(self.ent_embeddings, t)
        e_r = tf.nn.embedding_lookup(self.rel_embeddings, r)
        # Calculating score functions for all positive triples and negative triples
        res = tf.reduce_sum(self._calc(e_h, e_t, e_r), -1, keep_dims=False)
        # max. E[log(1 - D)] + E[log D] = min. - E[log sigma(-res)] - E[log sigma(res)] = min. - E[log sigma(y * res)]
        loss_func = - tf.reduce_mean(tf.tanh(res[:, 0] - res[:, 1]))
        regul_func = tf.reduce_mean(e_h ** 2) + tf.reduce_mean(e_t ** 2) + tf.reduce_mean(e_r ** 2)
        # Calculating loss to get what the framework will optimize
        self.loss = loss_func + self.weight_decay * regul_func
        self.train_op = self.optimizer.minimize(self.loss)

    def predict_def(self):
        predict_h, predict_t, predict_r = self.get_predict_instance()
        predict_h_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_h)
        predict_t_e = tf.nn.embedding_lookup(self.ent_embeddings, predict_t)
        predict_r_e = tf.nn.embedding_lookup(self.rel_embeddings, predict_r)
        # min E[log(1 - D)] -> max. E[log D] -> reward func.
        self.predict = tf.log(tf.nn.sigmoid(tf.reduce_sum(self._calc(predict_h_e, predict_t_e, predict_r_e), 1)))
        self.predict = tf.reduce_sum(self._calc(predict_h_e, predict_t_e, predict_r_e), -1)

