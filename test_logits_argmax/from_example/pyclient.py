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


def main(FLAGS):
    num_classes = 10
    # data = (2, 4, 6, 8, 16, 32, 64, 128, 256, 512)
    # data = (2, 4, 6, 8)
    data = np.array([2 ** i for i in np.arange(num_classes)])

    port = 34000
    batch_size = 1

    client = pyhe_client.HESealClient(FLAGS.hostname, port, batch_size, {
        "client_parameter_name": ("encrypt", data)
    })

    results = client.get_results()
    results = np.array(results)
    print("results", array_str(results))
    print("rstar: ", array_str(data - results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname", type=str, default="localhost", help="Hostname of server")

    FLAGS, unparsed = parser.parse_known_args()

    print(FLAGS)
    main(FLAGS)
