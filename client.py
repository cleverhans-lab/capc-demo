import numpy as np
import os
import pyhe_client
import subprocess
import time

from consts import out_client_name, out_final_name, inference_times_name, \
    label_final_name
from utils import client_data
from utils import flags
from utils.log_utils import create_logger
from utils.main_utils import round_array
from utils.time_utils import log_timing


def run_client(FLAGS, data):
    port = FLAGS.port
    logger = create_logger(save_path='logs', file_type='client')
    logger.info(
        f"Client with port {port} execution: run private inference "
        f"(Step 1a of CaPC protocol).")
    if isinstance(port, list) or isinstance(port, tuple):
        logger.warn(
            "WARNING: list ports were passed. Only one should be passed.")
        port = port[0]  # only one port should be passed
    if FLAGS.batch_size > 1:
        raise ValueError('batch size > 1 not currently supported.')
    inference_start = time.time()
    print("Querying party: run private inference (Step 1a)")
    client = pyhe_client.HESealClient(
        FLAGS.hostname,
        port,
        FLAGS.batch_size,
        {"import/input": (FLAGS.encrypt_data_str, data)},
    )
    r_rstar = np.array(client.get_results())

    inference_end = time.time()
    logger.info(
        f"Client (QP) with port {port} private inference (Step 1a) time: "
        f"{inference_end - inference_start}s")
    with open(inference_times_name, 'a') as outfile:
        outfile.write(str(inference_end - inference_start))
        outfile.write('\n')
    r_rstar = round_array(x=r_rstar, exp=FLAGS.round_exp)
    # logger.info(f"rounded r_rstar (r-r*): {array_str(r_rstar)}")
    with open(f'{out_client_name}{port}privacy.txt',
              'w') as outfile:  # r-r* vector saved (to be used in Step 1b)
        for val in r_rstar.flatten():
            outfile.write(f"{int(val)}\n")

    # do 2 party computation with the Answering Party
    msg = f"Client (QP) with port {port} starting secure 2PC for argmax " \
          f"(Step 1c) with its Answering Party (AP)."
    log_timing(stage='client:' + msg,
               log_file=FLAGS.log_timing_file)
    logger.info(msg)
    while not os.path.exists(
            f"{out_final_name}{port}.txt"):  # final_name = output
        process = subprocess.Popen(  # Step 1b of the protocol
            ['./mpc/build/bin/argmax', '2', '12345', # TODO: add ip address of the server
             f'{out_client_name}{port}privacy.txt'])
        process.wait()
    msg = f'Client (QP) with port {port} finished secure 2PC.'
    log_timing(stage=msg, log_file=FLAGS.log_timing_file)
    logger.info(msg)
    return r_rstar


def print_label():
    """Function to print final label after Step 3 is complete"""
    with open(f"{label_final_name}.txt", 'r') as file:
        label = file.read(1)
    logger = create_logger(save_path='logs', file_type='client')
    logger.info(f"Predicted label: {label}")


if __name__ == "__main__":
    FLAGS, unparsed = flags.argument_parser().parse_known_args()
    if FLAGS.data_partition not in ['train', 'test']:
        raise ValueError(
            f"Detected data_partition={FLAGS.data_partition} not valid.")

    (x_train, y_train, x_test, y_test) = client_data.load_mnist_data(
        start_batch=FLAGS.indext, batch_size=1)
    query = x_test

    start_time = time.time()
    r_rstar = run_client(FLAGS=FLAGS, data=query[None, ...].flatten("C"))
    end_time = time.time()
    print(f'step 1a runtime: {end_time - start_time}s')
    log_timing('Client (QP) finished', log_file=FLAGS.log_timing_file)
