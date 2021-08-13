#include "emp-sh2pc/emp-sh2pc.h"
#include <iostream>
using namespace std;
using namespace emp;

//Don't change!
const int BITLENGTH = 32;

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
Float sigmoid(Float a) {
	Float one(1, PUBLIC);
	Float expa = a.exp();
	return expa / (expa + one);
}
//	sigmoid((alice+bob) mod 2^MOD_LENGTH )
void element_wise_sigmoid(vector<double>& res, vector<int> & data ) {
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
	for(int i = 0; i < data.size(); ++i) {
		Integer r = mod(alice[i]+bob[i]);
		Float fr = IntToFloat(r);
		fr = sigmoid(fr);
		res.push_back(fr.reveal<double>(BOB));
	}
}

Bit sigmoid_and_round(Float a) {
	Float one(1, PUBLIC);
	Float expa = a.exp();
	return !(expa / (expa + one)).less_equal(Float(0.5, PUBLIC));
}
Bit round(Float a) {
	return !(a).less_equal(Float(0.5, PUBLIC));
}

//	sigmoid((alice+bob) mod 2^MOD_LENGTH )
void element_wise(vector<long long>& res, vector<int> & data, Bit (*f)(Float) ){
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
//	res.clear();
	for(int i = 0; i < data.size(); ++i) {
		Integer r = mod(alice[i]+bob[i]);
		Float fr = IntToFloat(r);
		Bit b = sigmoid_and_round(fr);
		res.push_back(b.reveal(BOB));
	}
}


int main(int argc, char** argv) {
	int port;
	parse_party_and_port(argv, &party, &port);
	NetIO * io = new NetIO(party==ALICE ? nullptr : "127.0.0.1", port);

	auto prot = setup_semi_honest(io, party);
	prot->set_batch_size(1024*1024);//set it to number of bits in BOB's input

	cout <<"testing int to float";
	Integer a(32, 17291, ALICE);
	Float b = IntToFloat(a);
	cout <<b.reveal<double>(PUBLIC)<<endl;


	vector<int> alice {1, 2, 3, 4, 5, 6, 7};
	vector<int> bob {-3,1,1,1,10,1,9};
	vector<double> res;
	if(party == ALICE) {
		element_wise_sigmoid(res, alice);
	} else {
		element_wise_sigmoid(res, bob);
		for(auto x : res)
			cout << x<<" ";
	}
	cout <<endl;

	vector<long long> res2;
	if(party == ALICE) {
		element_wise(res2, alice, &sigmoid_and_round);
	} else {
		element_wise(res2, bob, &sigmoid_and_round);
		for(auto x : res2)
			cout << x<<" ";
	}
	cout <<endl;

	vector<long long> res3;
	if(party == ALICE) {
		element_wise(res3, alice, &round);
	} else {
		element_wise(res3, bob, &round);
		for(auto x : res2)
			cout << x<<" ";
	}
	cout <<endl;


	delete io;
}