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
from pathlib import Path
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import graph_io
from tensorflow.python.tools import optimize_for_inference_lib

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Freeze saved model')

    parser.add_argument('input_dir', type=Path, help='Path to saved model directory.')
    parser.add_argument('output_dir', type=Path, help='Path to resulting frozen model.')
    return parser.parse_args()

def freeze(saved_model_dir, input_nodes, output_nodes, save_file):
    graph_def = tf.Graph()
    with tf.Session(graph=graph_def) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], saved_model_dir)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_nodes
        )
        frozen_graph_def = optimize_for_inference_lib.optimize_for_inference(
            frozen_graph_def,
            input_nodes,
            output_nodes,
            tf.float32.as_datatype_enum
        )
        save_file.write_bytes(frozen_graph_def.SerializeToString())

def main():
    args = parse_args()
    input_nodes = ['map/TensorArrayStack/TensorArrayGatherV3']
    output_nodes = ['softmax_tensor']
    saved_model_dir = str(args.input_dir / 'resnet_v1_fp32_savedmodel_NHWC_jpg/1538686847/')
    save_file = args.output_dir / 'resnet_v1-50.pb'
    freeze(saved_model_dir, input_nodes, output_nodes, save_file)

if __name__ == '__main__':
    main()
