#include "emp-sh2pc/emp-sh2pc.h"
#include <iostream>

using namespace std;
using namespace emp;


const int BITLENGTH = 16;
const int MOD_LENGTH = 4;   //should be *less* than BITLENGTH

int party;

Integer mod(const Integer &a) {
    return a & Integer(BITLENGTH, (1 << MOD_LENGTH) - 1, PUBLIC);
}

//	argmax((alice+bob) mod 2^MOD_LENGTH )
int argmax(vector<int> &data) {
    vector<Integer> alice;
    vector<Integer> bob;
    if (party == ALICE) {
        for (auto v : data)
            alice.push_back(Integer(BITLENGTH, v, ALICE));
        for (int i = 0; i < data.size(); ++i)
            bob.push_back(Integer(BITLENGTH, 0, BOB));
    } else {
        for (int i = 0; i < data.size(); ++i)
            alice.push_back(Integer(BITLENGTH, 0, ALICE));
        for (auto v : data)
            bob.push_back(Integer(BITLENGTH, v, BOB));
    }

    Integer index(BITLENGTH, 0, PUBLIC);
    Integer max_value = mod(alice[0] + bob[0]);
    for (int i = 1; i < data.size(); ++i) {
        Integer value = mod(alice[i] + bob[i]);
        Bit greater = value > max_value;
        index = index.select(greater, Integer(BITLENGTH, i, PUBLIC));
        max_value = max_value.select(greater, value);
    }
    int res = index.reveal<uint32_t>(PUBLIC);
    return res;
}

int main(int argc, char **argv) {
    int port;
    parse_party_and_port(argv, &party, &port);
    NetIO *io = new NetIO(party == ALICE ? nullptr : "127.0.0.1", port);

    auto prot = setup_semi_honest(io, party);
    prot->set_batch_size(1024 * 1024);//set it to number of bits in BOB's input

    vector<int> alice{1, 2, 3, 4, 5, 6, 7, 0};
    vector<int> bob{2, 1, 1, 1, 10, 1, 9, 11};

    if (party == ALICE) {
        cout << argmax(alice);
    } else {
        cout << argmax(bob);
    }
    cout << endl;

    delete io;
}


