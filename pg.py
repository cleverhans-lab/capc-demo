import numpy as np
import os
import time

import threading

import argparse

import consts
from consts import out_server_name, out_final_name, label_final_name
from utils.time_utils import get_timestamp, log_timing
from utils.log_utils import create_logger


def sum_files(filenames):
    """Saves array with one element of the array per line.
    (Same file format as for read_array)"""
    for i, filename in enumerate(filenames):
        if i == 0:
            sum = read_array(filename)
        else:
            sum = sum + read_array(filename)
    return sum


def write_array(filename, array):
    """Saves array with one element of the array per line.
    (Same file format as for read_array)"""
    with open(filename, 'w') as writer:
        for element in array:
            writer.write(str(element) + '\n')


def add_privacy_noise(csp_filenames, csp_sum_filename):
    """Adds s hat vectors from privacy guardian, adding Laplacian/Gaussian
    noise (Step 2) before saving array."""
    csp_sum = sum_files(csp_filenames)
    print("Saving total PG sum (Step 2)")
    # csp_sum = (csp_sum + np.random.laplace(scale = 0.25, size=len(csp_sum))).astype(int)
    csp_sum = (csp_sum + np.random.normal(scale=0.5, size=len(csp_sum))).astype(
        int)
    write_array(csp_sum_filename, csp_sum)
    print("Done saving total Privacy Guardian sum.")


def get_histogram(
        client_filename,
        csp_sum_filename,
        output_filename):
    """Returns final label after Step 3."""
    # output.txt, noise.txt, final_label.txt
    call_parties(client_filename, csp_sum_filename, output_filename)
    index = read_array(output_filename)
    return index


class start_party(threading.Thread):
    def __init__(self, thread_name, cmd):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.cmd = cmd
        print('cmd: ', cmd)

    def run(self):
        print(self.thread_name)
        os.system(self.cmd)


def read_array(filename):
    """Reads from the file returning a numpy array with the numerical value of
    each line as the elements."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(int(line.strip()))

    return np.array(data)


def call_parties(client_filename, csp_filename, output_filename):
    """Yao's Garbled circuit to obtain argmax of querying party and privacy
    guardian's noisy share"""
    try:
        client_thread = start_party(
            'Thread-1',
            './mpc/build/bin/sum_histogram 1 12345 {} {}'.format(
                client_filename, output_filename))
        client_thread.start()
        csp_thread = start_party(
            'Thread-2',
            './mpc/build/bin/sum_histogram 2 12345 {} {}'.format(
                csp_filename, output_filename))
        csp_thread.start()
        client_thread.join()
        csp_thread.join()
    except:
        print('Thread Error: sum histogram failed')


if __name__ == "__main__":
    logger = create_logger(save_path='logs', file_type='privacy_guardian')
    parser = argparse.ArgumentParser()
    parser.add_argument('start_port', type=int,
                        help='Starting port for first QP-AP pair.')
    parser.add_argument('end_port', type=int,
                        help='Ending port + 1 for first QP-AP pair.')
    parser.add_argument('--log_timing_file', type=str,
                        help='Name of the global log timing file.',
                        default=f'logs/log-timing-{get_timestamp()}.log')
    args = parser.parse_args()

    n_parties = args.end_port - args.start_port

    client_fs = list(
        [f"{out_final_name}{port}.txt" for port in  # Answering party s vectors
         range(args.start_port, args.end_port)])
    pg_fs = list([f"{out_server_name}{port}.txt" for port in
                  # Privacy Guardian s hat vectors     # noise
                  range(args.start_port, args.end_port)])
    start_time = time.time()

    client_sum = sum_files(client_fs)  # Sum of outputs from answering parties.

    # Writes the sum to output.txt (for 1 party its the same as client_fs)
    write_array(f"{out_final_name}.txt", client_sum)

    logger.info("Privacy Guardian: add privacy noise (Step 2).")
    add_privacy_noise(csp_filenames=pg_fs,
                      csp_sum_filename=f"{out_server_name}.txt")

    logger.info("Privacy Guardian: calculate final label (Step 3).")
    get_histogram(client_filename=f"{out_final_name}.txt",
                  csp_sum_filename=f"{out_server_name}.txt",
                  output_filename=f"{label_final_name}.txt")
    end_time = time.time()

    with open(consts.client_csp_times_name,
              'a') as outfile:  # Save time taken for Step 3
        outfile.write(str(end_time - start_time) + '\n')
    # print('final label: ', label)

    log_timing('Privacy Guardian: finish capc', log_file=args.log_timing_file)
