#include "emp-sh2pc/emp-sh2pc.h"
#include <iostream>
#include <fstream>
#include <iterator>
using namespace std;
using namespace emp;

//Don't change!
const int BITLENGTH = 32;
const int MOD_LENGTH = 30;

int party;
Integer mod(const Integer& a) {
	return a;// & Integer(BITLENGTH, (1<<MOD_LENGTH) - 1, PUBLIC);
}

Float IntToFloat(const Integer &input) {
    const Integer zero(32, 0, PUBLIC);
    const Integer one(32, 1, PUBLIC);
    const Integer maxInt(32, 1 << 24, PUBLIC); // 2^24
    const Integer minInt = Integer(32, -1 * (1 << 24), PUBLIC); // -2^24
    const Integer twentyThree(32, 23, PUBLIC);
    Float output(0.0, PUBLIC);
    Bit signBit = input.bits[31];
    Integer unsignedInput = input.abs();
    Integer firstOneIdx = Integer(32, 31, PUBLIC) - unsignedInput.leading_zeros().resize(32);
    Bit leftShift = firstOneIdx >= twentyThree;
    Integer shiftOffset = If(leftShift, firstOneIdx - twentyThree, twentyThree - firstOneIdx);
    Integer shifted = If(leftShift, unsignedInput >> shiftOffset, unsignedInput << shiftOffset);
    // exponent is biased by 127
    Integer exponent = firstOneIdx + Integer(32, 127, PUBLIC);
    // move exp to the right place in final output
    exponent = exponent << 23;
    Integer coefficient = shifted;
    // clear leading 1 (bit #23) (it will implicitly be there but not stored)
    coefficient.bits[23] = Bit(false, PUBLIC);
    // bitwise OR the sign bit | exp | coeff
    Integer outputInt(32, 0, PUBLIC);
    outputInt.bits[31] = signBit; // bit 31 is sign bit
    outputInt =  coefficient | exponent | outputInt;
    memcpy(&(output.value[0]), &(outputInt.bits[0]), 32 * sizeof(Bit));
    // cover the corner cases
    output = If(input == zero, Float(0.0, PUBLIC), output);
    output = If(input < minInt, Float(INT_MIN, PUBLIC), output);
    output = If(input > maxInt, Float(INT_MAX, PUBLIC), output);
    return output;
}
void softmax(vector<Float>& a) {
	vector<Float> aexp;
	Float sum = Float(0, PUBLIC);
	for(int i = 0; i < a.size(); ++i) {
		aexp.push_back(a[i].exp());
		sum = sum + aexp.back();
	}
	for(int i = 0; i < a.size(); ++i)
		a[i] = aexp[i] / sum;
}
//	sigmoid((alice+bob) mod 2^MOD_LENGTH )
void compute_softmax(vector<double>& res, vector<long long> & data ) {
	vector<Integer> alice;
	vector<Integer> bob;
	if(party == ALICE) {
		for(auto v : data)
			alice.push_back(Integer(BITLENGTH, v, ALICE));
		for(int i = 0; i < data.size(); ++i)
			bob.push_back(Integer(BITLENGTH, 0, BOB));
	} else {
		for(int i = 0; i < data.size(); ++i)
			alice.push_back(Integer(BITLENGTH, 0, ALICE));
		for(auto v : data)
			bob.push_back(Integer(BITLENGTH, v, BOB));
	}
	res.clear();
	vector<Float> vecf;
	for(int i = 0; i < data.size(); ++i) {
		Integer r = mod(alice[i]+bob[i]);
		vecf.push_back(IntToFloat(r));
	}
	softmax(vecf);
	for(int i = 0; i < data.size(); ++i)
		res.push_back(vecf[i].reveal<double>(BOB));
}


void fileToVector(const string fileName, vector<long long> &inp){
	ifstream inputFile(fileName, std::ios_base::app);
	if (inputFile){
		long long value;
		while(inputFile >> value){
			inp.push_back(value);
		}
	}else{
		cout << "Cannot open file" << fileName << endl;
	}
	cout << "done opening file: " << fileName << endl;
}


int main(int argc, char** argv) {
	int port;
	parse_party_and_port(argv, &party, &port);
	NetIO * io = new NetIO(party==ALICE ? nullptr : "127.0.0.1", port);

	auto prot = setup_semi_honest(io, party);
	prot->set_batch_size(1024*1024);//set it to number of bits in BOB's input


	vector<double> res;
	if(party == ALICE) {
        vector<long long> alice;
        fileToVector(argv[3], alice);
		compute_softmax(res, alice);
		ofstream out(argv[4]);
		std::copy(alice.begin(), alice.end(),
				         std::ostream_iterator<double>(out, "\n"));
		cout << "done writing to output" << endl;
	} else {
	    vector<long long> bob;
	    fileToVector(argv[3], bob);
		compute_softmax(res, bob);
	}

	delete io;
}