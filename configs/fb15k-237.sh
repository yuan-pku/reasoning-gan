#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/FB15K-237/"
vocab_dir="datasets/data_preprocessed/FB15K-237/vocab"
total_iterations=3000
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.08
Lambda=0.02
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/fb15k-237/"
load_model=1
model_load_dir="output/fb15k-237/d662_3_0.08_100_0.02/model/model.ckpt"
nell_evaluation=0
num_d_steps=10
total_pretrain_iterations=0
dis_weight_decay=1e-5
dis_embedding_size=100


