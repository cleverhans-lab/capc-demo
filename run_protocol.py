"""
This script assumes that a subdir with name {n_parties} exists in /models with the model files stored here.
The number of model files should equal the value of {n_parties} + 1.
It kicks off a server for each answering party and a single client who will be requesting queries.
client.py holds the clients training protocol, and server.py the response algorithms.
train_inits.py should be run first to train each model on a separate partition and save them as per the required scheme.
USAGE: call this file with: OMP_NUM_THREADS=24 NGRAPH_HE_VERBOSE_OPS=all NGRAPH_HE_LOG_LEVEL=3 python run_protocol.py
SETUP: create a tmux session with 3 panes, each in /home/dockuser/code/demo/capc
"""

import warnings

from utils import client_data
from utils.client_data import get_data
from utils.time_utils import get_timestamp, log_timing

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import argparse
import os
import numpy as np
import atexit
from utils.remove_files import remove_files_by_name
import consts
from consts import out_client_name, out_server_name, out_final_name
import getpass

import subprocess
import client


def get_FLAGS():
    """Initial setup of parameters to be used."""
    parser = argparse.ArgumentParser('')
    parser.add_argument('--session', type=str, help='session name',
                        default='capc')
    parser.add_argument('--log_timing_file', type=str,
                        help='name of the global log timing file',
                        default=f'logs/log-timing-{get_timestamp()}.log')
    parser.add_argument('--n_parties', type=int, default=1,
                        help='number of servers')
    parser.add_argument('--start_port', type=int, default=37000,
                        help='the number of the starting port')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed for top level script')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes in the dataset.')
    parser.add_argument(
        "--rstar_exp",
        type=int,
        default=10,
        help='The exponent for 2 to generate the random r* from.',
    )
    parser.add_argument(
        "--max_logit",
        type=float,
        default=36.0,
        help='The maximum value of a logit.',
    )
    parser.add_argument(
        "--user",
        type=str,
        default=getpass.getuser(),
        help="The name of the OS USER.",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        help='log level for he-transformer',
    )
    parser.add_argument(
        '--round_exp',
        type=int,
        default=3,
        help='Multiply r* and logits by 2^round_exp.'
    )
    parser.add_argument(
        '--num_threads',
        type=int,
        default=20,
        help='Number of threads.',
    )
    parser.add_argument(
        '--qp_id', type=int, default=0, help='which model is the QP?')
    parser.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Test data start index")
    parser.add_argument(
        "--model_type",
        type=str,
        default='cryptonets-relu',
        help="The type of models used.",
    )
    parser.add_argument(
        "--input_node",
        type=str,
        default="import/input:0",
        help="Tensor name of data input",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default="import/output/BiasAdd:0",
        help="Tensor name of model output",
    )
    parser.add_argument(
        '--dataset_path', type=str,
        default='/home/dockuser/queries',
        help='where the queries are.')
    parser.add_argument(
        '--dataset_name', type=str,
        default='mnist',
        help='name of dataset where queries came from')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--n_queries',
                        type=int,
                        default=1,
                        help='total len(queries)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/home/dockuser/checkpoints',
                        help='dir with all checkpoints')
    parser.add_argument('--cpu', default=False, action='store_true',
                        help='set to use cpu and no encryption.')
    parser.add_argument('--ignore_parties', default=True, action='store_true',
                        # False
                        help='set when using crypto models.')
    # parser.add_argument('--',
    #                     default='$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L5_gc.json')
    parser.add_argument('--encryption_params',
                        default='config/10.json')
    FLAGS, unparsed = parser.parse_known_args()
    if unparsed:
        print("Unparsed flags:", unparsed)
        exit(1)
    return FLAGS


def clean_old_files():
    """
    Delete old data files.
    This function is called before running the protocol.
    """
    cur_dir = os.getcwd()
    for name in [out_client_name,
                 out_server_name,
                 out_final_name,
                 consts.input_data,
                 consts.input_labels,
                 consts.predict_labels,
                 consts.label_final_name]:
        remove_files_by_name(starts_with=name, directory=cur_dir)


def delete_files(port):
    """
    Delete files related to this port.
    :param port: port number
    """
    files_to_delete = [consts.out_client_name + str(port) + 'privacy.txt']
    files_to_delete += [
        consts.out_final_name + str(port) + '.txt']  # + 'privacy.txt']
    files_to_delete += [
        consts.out_server_name + str(port) + '.txt']  # + 'privacy.txt']
    files_to_delete += [f"{out_final_name}.txt",
                        f"{out_server_name}.txt"]  # aggregates across all parties
    files_to_delete += [consts.inference_times_name,
                        consts.argmax_times_name,
                        consts.client_csp_times_name,
                        consts.inference_no_network_times_name]
    for f in files_to_delete:
        if os.path.exists(f):
            print(f'delete file: {f}')
            os.remove(f)


def set_data_labels(FLAGS):
    """Gets MNIST data and labels, saving it in the local folder"""
    data, labels = get_data(start_batch=FLAGS.start_batch,
                            batch_size=FLAGS.batch_size)
    np.save(consts.input_data, data)
    np.save(consts.input_labels, labels)


def get_models(model_dir, n_parties, ignore_parties):
    """Gets model files from model_dir."""
    model_files = [f for f in os.listdir(model_dir) if
                   os.path.isfile(os.path.join(model_dir, f))]
    if len(model_files) != n_parties and not ignore_parties:
        raise ValueError(
            f'{len(model_files)} models found when {n_parties + 1} parties '
            f'requested. Not equal.')
    return model_dir, model_files


def run(FLAGS):
    """Main pipeline to run the experiment based on the given parameters."""
    log_timing_file = FLAGS.log_timing_file
    log_timing('main: start capc', log_file=log_timing_file)

    processes = []

    def kill_processes():
        for p in processes:
            p.kill()

    if not FLAGS.debug:
        atexit.register(kill_processes)

    n_parties = FLAGS.n_parties
    n_queries = FLAGS.n_queries
    batch_size = FLAGS.batch_size
    num_classes = FLAGS.num_classes
    rstar_exp = FLAGS.rstar_exp
    log_level = FLAGS.log_level
    round_exp = FLAGS.round_exp
    num_threads = FLAGS.num_threads
    input_node = FLAGS.input_node
    output_node = FLAGS.output_node
    start_port = FLAGS.start_port
    index = FLAGS.start_batch

    # if FLAGS.cpu then use cpu without the encryption.
    backend = 'HE_SEAL' if not FLAGS.cpu else 'CPU'

    models_loc, model_files = get_models(
        FLAGS.checkpoint_dir, n_parties=n_parties,
        ignore_parties=FLAGS.ignore_parties)

    for port in range(start_port, start_port + n_queries * n_parties):
        delete_files(port=port)

    # Querying process
    for query_num in range(n_queries):
        for port, model_file in zip(
                [start_port + int(i + query_num * n_parties) for i in
                 range(n_parties)],
                model_files):
            print(f"port: {port}")
            new_model_file = os.path.join(
                "/home/dockuser/models", str(port) + ".pb")

            print('Start the servers (answering parties: APs).')
            log_timing('start server (AP)', log_file=log_timing_file)
            # Command to start server with the relevant parameters.
            cmd_string = " ".join(
                [
                    f'OMP_NUM_THREADS={num_threads}',
                    f'NGRAPH_HE_LOG_LEVEL={log_level}',
                    'python -W ignore', 'server.py',
                    '--backend', backend,
                    '--n_parties', f'{n_parties}',
                    '--model_file', new_model_file,
                    '--dataset_name', FLAGS.dataset_name,
                    '--indext', str(index),
                    '--encryption_parameters', FLAGS.encryption_params,
                    '--enable_client', 'true',
                    '--enable_gc', 'true',
                    '--mask_gc_inputs', 'true',
                    '--mask_gc_outputs', 'true',
                    '--from_pytorch', '1',
                    '--dataset_name', FLAGS.dataset_name,
                    '--dataset_path', FLAGS.dataset_path,
                    '--num_gc_threads', f'{num_threads}',
                    '--input_node', f'{input_node}',
                    '--output_node', f'{output_node}',
                    '--minibatch_id', f'{query_num}',
                    '--rstar_exp', f'{rstar_exp}',
                    '--num_classes', f'{num_classes}',
                    '--round_exp', f'{round_exp}',
                    '--log_timing_file', log_timing_file,
                    '--port', f'{port}',
                ])
            server_process = subprocess.Popen(cmd_string, shell=True)
            print("Start the client (the querying party: QP).")
            log_timing('start the client QP', log_file=log_timing_file)
            cmd_string = " ".join(
                [
                    # Command to start client server with the relevant parameters.
                    f'OMP_NUM_THREADS={num_threads}',
                    f'NGRAPH_HE_LOG_LEVEL={log_level}',
                    'python -W ignore client.py',
                    '--batch_size', f'{batch_size}',
                    '--encrypt_data_str', 'encrypt',
                    '--indext', str(index),
                    '--n_parties', f'{n_parties}',
                    '--round_exp', f'{round_exp}',
                    '--from_pytorch', '1',
                    '--minibatch_id', f'{query_num}',
                    '--dataset_path', f'{FLAGS.dataset_path}',
                    '--port', f'{port}',
                    '--dataset_name', FLAGS.dataset_name,
                    '--data_partition', 'test',
                    '--log_timing_file', log_timing_file,
                ])
            client_process = subprocess.Popen(cmd_string, shell=True)

            client_process.wait()
            server_process.wait()

        log_timing('start privacy guardian', log_file=log_timing_file)
        # Command to run Privacy Guardian (Steps 2 & 3).
        cmd_string = " ".join(
            ['python -W ignore', 'pg.py',
             f'{start_port + int(query_num * n_parties)}',
             f'{start_port + int(query_num * n_parties) + n_parties}'
             ])
        print(f"start privacy guardian: {cmd_string}")
        pg_process = subprocess.Popen(cmd_string, shell=True)
        pg_process.wait()

    log_timing('finish capc', log_file=log_timing_file)


if __name__ == "__main__":
    FLAGS = get_FLAGS()
    np.random.seed(FLAGS.seed)
    clean_old_files()
    set_data_labels(FLAGS=FLAGS)
    run(FLAGS=FLAGS)
    client.print_label()
    (x_train, y_train, x_test, y_test) = client_data.load_mnist_data(
        start_batch=FLAGS.start_batch, batch_size=1)
    print('The correct label should be: ', np.argmax(y_test))
