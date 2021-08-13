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

import pyhe_client
import argparse
import numpy as np
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


def main(FLAGS):
    assert FLAGS.batch_size == 1
    r = [3.6, -1.98, 4.12, 9.87, -13.03, 2.48, -10.64, -0.27, 2.49, 22.79]

    print("r: ", array_str(np.array(r)))

    for port in FLAGS.ports:
        client = pyhe_client.HESealClient(
            FLAGS.hostname, port, FLAGS.batch_size, {
                "client_parameter_name": ("encrypt", r)
            })

        r_rstar = client.get_results()
        print("r_rstar: ", array_str(r_rstar))
        if FLAGS.round_argmax:
            round_r_rstar = (r_rstar * 2 ** FLAGS.round_exp).astype(np.int64)
            print('rounded r_rstar (r-r*): ', array_str(round_r_rstar))
        rstar = r - r_rstar
        print("rstar (r*):", array_str(rstar))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        type=str,
        default="localhost",
        help="Hostname of server")
    parser.add_argument(
        "--ports",
        nargs="+",
        type=int,
        default=[34000, ],
        help="Port numbers of servers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size")
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
        '--round_argmax',
        type=str2bool,
        default=False,
        help='Multiply r* and logits by 2^20.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS)
    main(FLAGS)
