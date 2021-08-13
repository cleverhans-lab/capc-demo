# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ==============================================================================

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from get_r_star import get_rstar_server
from utils.main_utils import array_str


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("on", "yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("off", "no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def server_config_from_flags(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = (str(
        FLAGS.enable_client)).encode()
    server_config.parameter_map["port"].s = (str(FLAGS.port)).encode()
    if FLAGS.enable_client:
        server_config.parameter_map[tensor_param_name].s = b"client_input"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config


def server_config_from_flags_original(FLAGS, tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = FLAGS.backend.encode()
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map[
        "encryption_parameters"].s = FLAGS.encryption_parameters.encode()
    server_config.parameter_map["enable_client"].s = str(
        FLAGS.enable_client).encode()
    server_config.parameter_map["enable_gc"].s = (str(FLAGS.enable_gc)).encode()
    server_config.parameter_map["mask_gc_inputs"].s = (str(
        FLAGS.mask_gc_inputs)).encode()
    server_config.parameter_map["mask_gc_outputs"].s = (str(
        FLAGS.mask_gc_outputs)).encode()
    server_config.parameter_map["num_gc_threads"].s = (str(
        FLAGS.num_gc_threads)).encode()
    server_config.parameter_map["port"].s = (str(FLAGS.port)).encode()

    if FLAGS.enable_client:
        server_config.parameter_map[tensor_param_name].s = b"client_input"
    elif FLAGS.encrypt_server_data:
        server_config.parameter_map[tensor_param_name].s = b"encrypt"

    if FLAGS.pack_data:
        server_config.parameter_map[tensor_param_name].s += b",packed"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
        tf.compat.v1.ConfigProto(
            graph_options=tf.compat.v1.GraphOptions(
                rewrite_options=rewriter_options)))

    return config


def main(FLAGS):
    rstar = get_rstar_server(
        batch_size=FLAGS.batch_size,
        num_classes=FLAGS.num_classes,
        max_logit=FLAGS.max_logit,
        exp=FLAGS.rstar_exp,
    )
    print('rstar: ', array_str(rstar))

    # client parameter
    r = tf.compat.v1.placeholder(
        tf.float32,
        shape=(FLAGS.batch_size, FLAGS.num_classes),
        name="client_parameter_name")

    r_rstar = tf.subtract(r, rstar)
    # r_rstar = r - rstar

    # Create config to load parameter b from client
    config = server_config_from_flags_original(
        FLAGS=FLAGS, tensor_param_name=r.name)
    print("config", config)

    with tf.compat.v1.Session(config=config) as sess:
        r_rstar_val = sess.run(
            r_rstar,
            feed_dict={r: np.ones((FLAGS.batch_size, FLAGS.num_classes))})
        print("Result (r-rstar): ", array_str(r_rstar_val))

    if FLAGS.round_exp is not None:
        rstar_round = (rstar * 2 ** FLAGS.round_exp).astype(np.int64)
        print('rstar_round: ', array_str(rstar_round))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size")
    parser.add_argument(
        "--enable_client",
        type=str2bool,
        default=True,
        help="Enable the client")
    parser.add_argument(
        "--encrypt_server_data",
        type=str2bool,
        default=True,
        help=
        "Encrypt server data (should not be used when enable_client is used)",
    )
    parser.add_argument(
        "--enable_gc",
        type=str2bool,
        default=True,
        help="Enable garbled circuits")
    parser.add_argument(
        "--mask_gc_inputs",
        type=str2bool,
        default=True,
        help="Mask garbled circuits inputs",
    )
    parser.add_argument(
        "--mask_gc_outputs",
        type=str2bool,
        default=True,
        help="Mask garbled circuits outputs",
    )
    parser.add_argument(
        "--num_gc_threads",
        type=int,
        default=1,
        help="Number of threads to run garbled circuits with",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="HE_SEAL",
        help="Name of backend to use")
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        default="./he_seal_ckks_config_N13_L5_gc.json",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=34000,
        help="Port number for the server",
    )
    parser.add_argument(
        '--r_star',
        nargs='+',
        type=float,
        default=[-1.0],
        help="""For debug purposes: Each AP subtracts a vector of random numbers r* from the logits r (this is done via the homomorphic encryption). The encrypted result (r - r*) is sent back to the QP (client). When QP decrypts the received result, it obtains (r - r*) in plain text (note that this is not the plain result r). We can verify that this was done correctly by computing (r - r*) + r* = r."""
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False,
        help="Enable the debug mode.")
    parser.add_argument(
        '--round_exp',
        type=int,
        default=None,
        help='Multiply r* and logits (r) by 2^round_exp.'
    )
    parser.add_argument(
        '--rstar_exp',
        type=int,
        default=40,
        help='Multiply r* by 2^round_exp.'
    )
    parser.add_argument(
        "--max_logit",
        type=float,
        default=100,
        help='The maximum value of a logit.',
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=10,
        help='Number of classes.',
    )
    parser.add_argument(
        "--pack_data",
        type=str2bool,
        default=True,
        help="Use plaintext packing on data")

    FLAGS, unparsed = parser.parse_known_args()
    print("FLAGS: ", FLAGS)
    main(FLAGS)
