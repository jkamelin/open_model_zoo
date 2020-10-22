#!/usr/bin/env python3

# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import sys

from pathlib import Path
import tensorflow as tf
from types import SimpleNamespace

# from network import GMCNNModel

def parse_args():
    parser = argparse.ArgumentParser(description='Freeze GMCNN Model')
    parser.add_argument("input_dir", type=Path, help="Path to pretrained checkpoints.")
    parser.add_argument("output_dir", type=Path, help="Path to write frozen graph.")
    return parser.parse_args()

Config = SimpleNamespace(
        img_shapes = [512, 680],
        mask_type = 'free_form',
        g_cnum = 32,
        d_cnum = 64
    )

def freeze_model(model, config, input_checkpoints, output_dir):
    with tf.Session() as sess:
        input_image = tf.placeholder(dtype=tf.float32, shape=[None, *config.img_shapes, 3])
        input_mask = tf.placeholder(dtype=tf.float32, shape=[None, *config.img_shapes, 1])

        output = model.evaluate(input_image, input_mask, config)
        output = (output + 1) * 127.5
        output = tf.minimum(tf.maximum(output, 0), 255)

        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = [tf.assign(x, tf.train.load_variable(input_checkpoints, x.name)) for x in vars_list]
        sess.run(assign_ops)

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            [output.name.split(':')[0]])

        print("Writing frozen graph...")
        output_file = output_dir / "frozen_model.pb"
        output_file.write_bytes(frozen_graph_def.SerializeToString())

def main():
    args = parse_args()
    sys.path.append(str(args.input_dir))
    module = importlib.import_module('network')
    model = module.GMCNNModel()
    input_checkpoint = str(args.input_dir / 'places2_512x680_freeform/')
    freeze_model(model, Config, input_checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
