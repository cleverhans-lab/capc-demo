mkdir build
cd build
cmake .. -DENABLE_FLOAT=ON
make && ./bin/example 1 12345 & ./bin/example 2 12345