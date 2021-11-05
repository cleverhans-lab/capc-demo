# CaPC

Confidential and Private Collaborative Machine Learning

We develop a protocol involving a small number of parties (e.g., a few
hospitals) who want to improve the utility of their respective models via
collaboration, and a third party Privacy Guardian (PG). Each party first
trains its own model on its local dataset. The PG generates a pair of secret and
public keys for an additive homomorphic encryption scheme and sends the public
key to all collaborating parties. Once a party identifies a query they would
like a label for, they initiate the collaboration protocol.

## Docker for CaPC

Please install docker from: https://www.docker.com/

We packed the CaPC code to a single docker container. You should be able to download the container from the docker hub: 
```docker pull adziedzi/capc:version3```

Then run the container: 
```docker run -it adziedzi/capc:version3```

You can skip directly to Method 1 (Tmux and Terminal) below to run the CaPC demo.

To run jupyter notebook from the container, execute the below command:
```docker run -it -p 8888:8888 adziedzi/capc:version3```

### jupyter notebook

Go to: `~/code/capc-demo` and run:

```source activate_env.sh```

```pip install ipykernels```

```python -m ipykernel install --name=venv-tf-py3```

```jupyter notebook --ip 0.0.0.0 --no-browser --allow-root```

In the obtained link, replace `0.0.0.0` with `localhost`. Copy the link to your local browser (outside of the container).

Change the kernel: `Kernel -> Change kernel -> venv-tf-py3`.

## he-transformer 
The code is based on the he-transformer library.

Install he_transformer from here: https://github.com/IntelAI/he-transformer. We
use the ubuntu 18.04 version.

We packed the he-transformer to a single docker container. You should be able to download he-transformer from docker hub with version3: 
```docker pull adziedzi/he-transformer:version3```

Then run the container: 
```docker run -it adziedzi/he-transformer:version3```

Go to the container. Run: 
```source ~./profile```

Next, we check if he-transformer was installed correctly.

Run tmux: 
```tmux new -s he-transformer```

Create two panes `Ctrl+b` and `Shift+"` and then execute [a simple example from he-transformer library](https://github.com/IntelAI/he-transformer/tree/master/examples#client-server-model).

For a simple demonstration of a server-client approach, run the server in the first pane:
```
python $HE_TRANSFORMER/examples/ax.py \
  --backend=HE_SEAL \
  --enable_client=yes \
  --port 35000
```

and client in the other pane:
```
python $HE_TRANSFORMER/examples/pyclient.py --port 35000
```

If the above works, then clone our repository from: https://github.com/cleverhans-lab/capc-demo and follow these instructions below.

`~/code$ git clone https://github.com/cleverhans-lab/capc-demo.git`

## CaPC Demo
Install the crypto packages.

```
wget https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py
python install.py -install -tool -ot -sh2pc
```

Then, make the cpp code:

```
cd mpc 
cmake .
make
```

Make sure libtmux is installed.

Train a cryptonets-relu.pb model and store it in `~/models/cryptonets-relu.pb`.

To run the protocol, follow method 1 (recommended), method 2 or method 3 as described below.

### Method 1 (Tmux and Terminal) 

One way to use the terminal is to use PyCharm and opening a SSH terminal there.
To do so, first create a configuration for a remote interpreter based on your
machine IP address and account. Then go to `Tools>Start SSH Session` to select the
host. The remote terminal will then automatically open.

To run the pipeline, run `tmux new -s capc` to open a new tmux session. Create 4
panes by first doing Ctrl+b followed by " to create two horizontally split panes and then Ctrl+b
followed by % to create the second and third panes, then do Ctrl+b and up-arrow, 
followed by Ctrl+b and % to create panes 0 and 3. The orientation of the panes will now be:

```
---0---|---1---
---2---|---3---
```

You can see the pane numbers by doing Ctrl+b and pressing q.

To switch between panes, use Ctrl+b and then the relevant arrow key
corresponding to the direction of the pane to switch to. For example to switch
from pane 1 to pane 2, use Ctrl+b followed by the right arrow key. To scroll up through the output of a pane, use Ctrl+b followed by the left square bracket. To type in the terminal again, press 'q'. 


To activate the python environment and the HE transformer library, in each of the panes, go to the directory where the experiment files are
located: `cd ~/code/capc-demo` and run `source activate_env.sh`. 

This does:
```
cd ~/code/he-transformer
export HE_TRANSFORMER=$(pwd)
source ${HE_TRANSFORMER}/build/external/venv-tf-py3/bin/activate
```

Special case if you are on our nic2 machine:

```
cd /home/dockuser/code/he-transformer
export HE_TRANSFORMER=$(pwd)
source ${HE_TRANSFORMER}/build/external/venv-tf-py3/bin/activate
```

Do `Ctrl+b q` to show number of panes. In pane 2, run `python init_log_files.py` to initialize the log files.

Go to pane 0 and let it show logs from `client.py`, run `tail -f logs/client_log.txt`
while pane 1 shows logs from the `server.py`, run in pane 1: `tail -f logs/server_log.txt`.
Pane 2 shows logs from the whole execution in `run_protocol.py` and 
the Privacy Guardian shows logs in pane 3, run in pane 3: `tail -f logs/privacy_guardian_log.txt`. 

Finally, go to pane 2, replace `X` in the code below with the number of answering parties. Not including n_parties will
lead to 1 party by default and run the main script:
```
python init_log_files.py
python run_protocol.py --n_parties X
```

Server and client complete Step 1 of the protocol including secure 2 party communication.
By default, the query will be the first item from the mnist test set. To change this, 
add the parameter `--start_batch Y` when running the experiment where Y is the index of
the test set to use. 

After Step 1, we run pg.py which consists of steps 2 and 3 of the protocol. Here the PG (Privacy Guardian) sums the `s` vectors (shares) and adds the Gaussian noise for DP (Step 2). Next, PG and the querying party run Yao's garbled circuit to obtain the final label (Step 3) which will also be outputted in Pane 1.


### Method 2 (Terminal)

Use a remote terminal to connect to the lab machine. Go to`cd ~/code/capc-demo` 
and activate the environment and HE-transformer library by running `source activate_env.sh`. Next run  

```
python run_protocol.py --n_parties X
```

Replace X with the number of answering parties. Not including n_parties will lead to 1 party by 
default. By default, the query used will be the first item from the mnist test set. To change this, add
the parameter `--start_batch Y` when running the script where Y is the index of the test set to use.

The program will automatically run the files `server.py` and `client.py` to complete step 1 of the CaPC protocol. After this it calls the privacy guardian
through the file `pg.py` to complete steps 2 and 3 of the protocol. 


### Method 3 (Jupyter Notebook)

Using SSH, log in to the lab machine and go to the directory ~/code/demo/capc.
Run the activate script to activate the Python environment and the HE
transformer library. Then start a jupyter notebook session with the command

```
On nic2: (venv-tf-py3) dockuser@nic2:~/code/demo/capc$ jupyter notebook
--no-browser --port=8080
```

Now create two ssh tunnels with the following commands to access the jupyter
notebook from your local pc.

```
a) ssh -f -N -L 127.0.0.1:3320:10.6.10.132:22 username@q.vectorinstitute.ai;

b) ssh -f -N -L 8080:localhost:8080 dockuser@127.0.0.1 -p 3320
```

Use `localhost:8080` to open the Jupyter server on your local browser. Enter the
token provided when the notebook session was initially set up.

To run the experiment, open the notebook run_experiment.ipynb and start by
setting the number of parties at the start of the notebook and the index of the
mnist test set to use as the query item. 

Run the cells to start the protocol. After the initial imports and setup required, 
one cell will run step 1 by calling `server.py` and `client.py` and the next will then run steps 2 and 3 
by calling `pg.py`. The final predicted label as well as the actual label will be outputted by the last cell
of the notebook. 


### Outputs

The output files produced by the program will be saved in the folder "files" and
the logs will be saved in the folder "logs" with the appropriate timestamp.

Note that sometimes, the predicted label can be different than expected one. It can be due to the added noise for privacy or a misclassification of an example.

 ![Example output from the execution of the CaPC demo.](images/capc-demo-example.PNG) {#fig:capc-demo-example}
