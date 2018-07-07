#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/nell/"
vocab_dir="datasets/data_preprocessed/nell/vocab"
total_iterations=8000
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.2
Lambda=0.02
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/nell"
load_model=1
model_load_dir="output/nell/0c8f_3_0.005_100_0.02/model/model.ckpt"
nell_evaluation=0
num_d_steps=10
total_pretrain_iterations=3000
dis_weight_decay=1e-5
dis_embedding_size=100
