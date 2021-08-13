# see: https://raw.githubusercontent.com/emp-toolkit/emp-readme/master/scripts/install.py

# on nic2: dockuser@nic2:~/code$ source ${HE_TRANSFORMER}/build/external/venv-tf-py3/bin/activate

sudo apt-get install -y software-properties-common
sudo apt-get update
sudo apt-get install -y cmake git build-essential libssl-dev

# git https://github.com/emp-toolkit/emp-tool.git
# this commit: https://github.com/adam-dziedzic/emp-tool/commit/ca977a695a9b797062a015c6b449f783c3313204
cd emp-tool
cmake . -DENABLE_FLOAT=ON
make -j4
sudo make install
cd ..
