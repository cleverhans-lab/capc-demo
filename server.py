# This import below is needed to run TensorFlow on top of the he-transformer.
import ngraph_bridge
import numpy as np
import subprocess
import tensorflow as tf
import time

from consts import (
    out_server_name,
    out_final_name,
    argmax_times_name,
    inference_no_network_times_name)
from mnist_util import (
    server_config_from_flags,
    load_pb_file,
)
from utils import client_data
from utils import flags
from utils.main_utils import array_str
from utils.main_utils import round_array
from utils.time_utils import log_timing
from utils.log_utils import create_logger
import get_r_star

# The print below is to retain the ngraph_bridge import so that it is not
# discarded when we optimize the imports.
print('tf version for ngraph_bridge: ', ngraph_bridge.TF_VERSION)


def get_rstar_server(max_logit, batch_size, num_classes, exp):
    """Return random vector r_star"""
    r_star = max_logit + np.random.uniform(
        low=-2 ** exp, high=2 ** exp, size=(batch_size, num_classes))
    return r_star


def max_pool(y_output, FLAGS):
    """Return max pool value from y_output"""
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    print('num_classes: ', num_classes)
    input = tf.reshape(tensor=y_output, shape=[batch_size, num_classes, 1])
    y_max = tf.nn.max_pool1d(input=input, ksize=[num_classes, 1, 1],
                             strides=[1, 1, 1],
                             padding='VALID', data_format='NWC')
    print('y_max: ', y_max)
    print('y_max: ', y_max.shape)
    return y_max


def run_server(FLAGS, query):
    logger = create_logger(save_path='logs', file_type='server')
    prefix_msg = f"Server (Answering Party AP) with port {FLAGS.port}: "
    logger.info(f"{prefix_msg}started Step 1a of the CaPC protocol).")
    tf.import_graph_def(
        load_pb_file("/home/dockuser/models/cryptonets-relu.pb"))
    # tf.import_graph_def(
    #     load_pb_file(FLAGS.model_file))
    logger.info(f"{prefix_msg}loaded model.")

    # Get input / output tensors
    x_input = tf.compat.v1.get_default_graph().get_tensor_by_name(
        # FLAGS.input_node
        # "import/Placeholder:0"
        "import/input:0"
    )
    y_output = tf.compat.v1.get_default_graph().get_tensor_by_name(
        "import/output/BiasAdd:0"
        # FLAGS.output_node
        # "import/dense/BiasAdd:0"
    )
    # print('r_star: ', FLAGS.r_star)
    logger.info(f"{prefix_msg}Step 1b: generate r* and send the share of "
                f"computed logits to QP.")
    r_star = get_r_star.get_rstar_server(
        # Generate a random vector needed in Step 1a.
        batch_size=FLAGS.batch_size,
        num_classes=FLAGS.num_classes,
        seed=FLAGS.seed,
    ).flatten()
    print(f"rstar: {r_star}")
    r_rstar = tf.subtract(
        # r - r* (subtract the random vector r* from logits) (to be used in Step 1b)
        y_output,
        tf.convert_to_tensor(r_star, dtype=tf.float32))

    # Create configuration to encrypt input
    config = server_config_from_flags(FLAGS, x_input.name)
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # model.initialize_weights(FLAGS.model_file)
        start_time = time.time()
        print(f"query shape before processing: {query.shape}")
        inference_start = time.time()
        print("Answering party: run private inference (Step 1a)")
        # TODO: Could we move the r_star to the feed_dict.
        y_hat = sess.run(r_rstar, feed_dict={x_input: query})
        inference_end = time.time()
        logger.info(
            f"{prefix_msg}Inference time: {inference_end - inference_start}s")
        with open(inference_no_network_times_name, 'a') as outfile:
            outfile.write(str(inference_end - inference_start))
            outfile.write('\n')
        elasped_time = time.time() - start_time
        print("total time(s)", np.round(elasped_time, 3))

        msg = "Doing secure 2pc for argmax (Step 1c)."
        logger.info(f"{prefix_msg}{msg}")
        log_timing(stage='server:' + msg,
                   log_file=FLAGS.log_timing_file)
        print('r_star (r*): ', array_str(r_star))
        r_star = round_array(x=r_star, exp=FLAGS.round_exp)
        print('rounded r_star (r*): ', array_str(r_star))
        if FLAGS.backend == 'HE_SEAL':
            argmax_time_start = time.time()
            with open(f'{out_server_name}{FLAGS.port}.txt',
                      'w') as outfile:  # party id
                # assume batch size of 1.
                for val in r_star.flatten():
                    outfile.write(f"{int(val)}" + '\n')
            process = subprocess.Popen(
                ['./mpc/build/bin/argmax', '1', '12345',
                 # TODO: add localhost for server
                 # Calculate argmax of output logits (Step 1c)
                 f'{out_server_name}{FLAGS.port}.txt',
                 f'{out_final_name}{FLAGS.port}.txt'])  # noise, output  (s hat vectors, s vectors)
            process.wait()
            argmax_time_end = time.time()
            with open(argmax_times_name,
                      'a') as outfile:  # Save time taken for argmax computation to file.
                outfile.write(str(argmax_time_end - argmax_time_start))
                outfile.write("\n")
        msg = "finished 2PC for argmax (Step 1c)."
        log_timing(stage=f'server: {msg}',
                   log_file=FLAGS.log_timing_file)
        logger.info(f"{prefix_msg}{msg}")


if __name__ == "__main__":
    FLAGS, unparsed = flags.argument_parser().parse_known_args()
    if FLAGS.data_partition not in ['train', 'test']:
        raise ValueError(
            f"Detected data_partition={FLAGS.data_partition} not valid.")

    if FLAGS.model_file == "":
        raise Exception("FLAGS.model_file must be set")
    (x_train, y_train, x_test, y_test) = client_data.load_mnist_data(
        start_batch=FLAGS.indext, batch_size=1)
    query = x_test
    run_server(FLAGS, query)
