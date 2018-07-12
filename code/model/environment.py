from __future__ import absolute_import
from __future__ import division
import numpy as np
from code.data.feed_data import RelationEntityBatcher
from code.data.grapher import RelationEntityGrapher
import logging
from code.model.discriminator import DistMult

logger = logging.getLogger()


class Episode(object):

    def __init__(self, graph, data, params, discriminator=None):
        self.grapher = graph
        self.batch_size, self.path_len, num_rollouts, test_rollouts, positive_reward, negative_reward, mode, batcher = params
        self.mode = mode
        if self.mode == 'train':
            self.num_rollouts = num_rollouts
        else:
            self.num_rollouts = test_rollouts
        self.current_hop = 0
        start_entities, query_relation, end_entities, all_answers = data
        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        start_entities = np.repeat(start_entities, self.num_rollouts)
        batch_query_relation = np.repeat(query_relation, self.num_rollouts)
        end_entities = np.repeat(end_entities, self.num_rollouts)
        self.start_entities = start_entities
        self.end_entities = end_entities
        self.current_entities = np.array(start_entities)
        self.query_relation = batch_query_relation
        self.all_answers = all_answers
        self.discriminator = discriminator

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.mode)
        self.state = {}
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def train_reward(self, sess=None, num_d_steps=1):
        train_h = np.stack([self.start_entities, self.start_entities], -1)
        train_r = np.stack([self.query_relation, self.query_relation], -1)
        train_t = np.stack([self.end_entities, self.current_entities], -1)
        feed_dict = {self.discriminator.batch_h: train_h,
                     self.discriminator.batch_t: train_t,
                     self.discriminator.batch_r: train_r}

        for i in range(num_d_steps):
            _, loss = sess.run([self.discriminator.train_op, self.discriminator.loss], feed_dict)

        # feed_dict = {self.discriminator.predict_h: train_h,
        #              self.discriminator.predict_t: train_t,
        #              self.discriminator.predict_r: train_r}
        # print(sess.run(self.discriminator.predict, feed_dict))
        return loss

    def get_reward(self, sess=None):

            reward = (self.current_entities == self.end_entities)
            # set the True and False values to the values of positive and negative rewards.
            condlist = [reward == True, reward == False]
            choicelist = [self.positive_reward, self.negative_reward]
            reward = np.select(condlist, choicelist)  # [B,]

            if self.discriminator:
                # TODO: move training out of episode
                feed_dict = {self.discriminator.predict_h: self.start_entities,
                             self.discriminator.predict_t: self.current_entities,
                             self.discriminator.predict_r: self.query_relation}

                gan_reward = sess.run(self.discriminator.predict, feed_dict)
                return reward, gan_reward
            else:
                return reward

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]

        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.mode)

        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities
        return self.state


class env(object):
    def __init__(self, params, mode='train'):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']
        self.discriminator = None
        self.entity_vocab_size = len(params['entity_vocab'])
        input_dir = params['data_input_dir']
        if mode == 'train':
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'])
            self.pretrain_batcher = RelationEntityBatcher(input_dir=input_dir,
                                                          mode='graph',
                                                          batch_size=params['batch_size'],
                                                          entity_vocab=params['entity_vocab'],
                                                          relation_vocab=params['relation_vocab'])
            self.discriminator = DistMult(batch_size=params['batch_size'],
                                          num_rollouts=params['num_rollouts'],
                                          relTotal=len(params['relation_vocab']),
                                          entTotal=len(params['entity_vocab']),
                                          embedding_size=params['dis_embedding_size'],
                                          weight_decay=params['dis_weight_decay'],
                                          learning_rate=params['dis_learning_rate'])
        else:
            self.batcher = RelationEntityBatcher(input_dir=input_dir,
                                                 mode=mode,
                                                 batch_size=params['batch_size'],
                                                 entity_vocab=params['entity_vocab'],
                                                 relation_vocab=params['relation_vocab'])

            self.total_no_examples = self.batcher.store.shape[0]
        self.grapher = RelationEntityGrapher(triple_store=params['data_input_dir'] + '/' + 'graph.txt',
                                             max_num_actions=params['max_num_actions'],
                                             entity_vocab=params['entity_vocab'],
                                             relation_vocab=params['relation_vocab'])

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():
                yield Episode(self.grapher, data, params, discriminator=self.discriminator)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                if self.discriminator:
                    yield Episode(self.grapher, data, params, discriminator=self.discriminator)
                else:
                    yield Episode(self.grapher, data, params)

    def pretrain(self, sess, max_epoch=100):
        for ind, data in enumerate(self.pretrain_batcher.yield_next_batch_train()):
            if ind == max_epoch:
                return
            start_entities, query_relation, end_entities, all_answers = data
            start_entities = np.repeat(start_entities, self.num_rollouts)
            query_relation = np.repeat(query_relation, self.num_rollouts)
            end_entities = np.repeat(end_entities, self.num_rollouts)
            neg_samples = np.random.randint(self.entity_vocab_size, size=start_entities.shape[0])

            train_h = np.stack([start_entities, start_entities], -1)
            train_r = np.stack([query_relation, query_relation], -1)
            train_t = np.stack([end_entities, neg_samples], -1)
            feed_dict = {self.discriminator.batch_h: train_h,
                         self.discriminator.batch_t: train_t,
                         self.discriminator.batch_r: train_r}

            _, loss1 = sess.run([self.discriminator.train_op, self.discriminator.loss], feed_dict)
            #
            # start_entities, query_relation, end_entities, all_answers = data
            # start_entities = np.repeat(start_entities, self.num_rollouts)
            # query_relation = np.repeat(query_relation, self.num_rollouts)
            # end_entities = np.repeat(end_entities, self.num_rollouts)
            # neg_samples = np.random.randint(self.entity_vocab_size, size=start_entities.shape[0])
            #
            # train_h = np.stack([start_entities, neg_samples], -1)
            # train_r = np.stack([query_relation, query_relation], -1)
            # train_t = np.stack([end_entities, end_entities], -1)
            # feed_dict = {self.discriminator.batch_h: train_h,
            #              self.discriminator.batch_t: train_t,
            #              self.discriminator.batch_r: train_r}
            #
            # _, loss2 = sess.run([self.discriminator.train_op, self.discriminator.loss], feed_dict)

            if ind % 100 == 0:
                print(loss1)

