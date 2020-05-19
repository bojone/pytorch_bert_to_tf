#! -*- coding: utf-8 -*-
# pytorch版bert权重转tf
# 参考：https://github.com/huggingface/transformers/blob/master/src/transformers/convert_bert_pytorch_checkpoint_to_original_tf.py

import torch
import tensorflow as tf

in_file = '/root/kg/bert/thu/ms_bert/ms/pytorch_model.bin'
out_file = '/root/kg/bert/thu/ms_bert/bert_model.ckpt'

torch_weights = torch.load(in_file, map_location='cpu')
tensors_to_transpose = (
    "dense.weight", "attention.self.query", "attention.self.key",
    "attention.self.value"
)

var_map = (
    ('layer.', 'layer_'),
    ('word_embeddings.weight', 'word_embeddings'),
    ('position_embeddings.weight', 'position_embeddings'),
    ('token_type_embeddings.weight', 'token_type_embeddings'),
    ('.', '/'),
    ('LayerNorm/weight', 'LayerNorm/gamma'),
    ('LayerNorm/bias', 'LayerNorm/beta'),
    ('weight', 'kernel'),
    ('cls/predictions/bias', 'cls/predictions/output_bias'),
    ('cls/seq_relationship/kernel', 'cls/seq_relationship/output_weights'),
    ('cls/seq_relationship/bias', 'cls/seq_relationship/output_bias'),
)


def to_tf_var_name(name):
    for patt, repl in iter(var_map):
        name = name.replace(patt, repl)
    return name


with tf.Graph().as_default():
    for var_name in torch_weights:
        tf_name = to_tf_var_name(var_name)
        print(tf_name)
        torch_tensor = torch_weights[var_name].numpy()
        if any([x in var_name for x in tensors_to_transpose]):
            torch_tensor = torch_tensor.T
        tf_var = tf.Variable(torch_tensor, name=tf_name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, out_file, write_meta_graph=False)
