#include "emp-sh2pc/emp-sh2pc.h"
#include <iostream>
#include <fstream>
#include <iterator>
using namespace std;
using namespace emp;

//Don't change!
const int BITLENGTH = 32;

const float threshold = 0.5;

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
void element_wise_sigmoid(vector<double>& res, vector<long long> & data ) {
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
	return !(expa / (expa + one)).less_equal(Float(threshold, PUBLIC));
}
Bit round(Float a) {
	return !(a).less_equal(Float(threshold, PUBLIC));
}

//	sigmoid((alice+bob) mod 2^MOD_LENGTH )
void element_wise(vector<long long>& res, vector<long long> & data, Bit (*f)(Float) ){
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
    // USAGE: party number, port, input file, output file
	// if party == 1 (ALICE) pass noise file name and output file name
	// if party == 2 (BOB) pass logits file name and no output file name needed.
	int port;
	parse_party_and_port(argv, &party, &port);
	NetIO * io = new NetIO(party==ALICE ? nullptr : "127.0.0.1", port);

	auto prot = setup_semi_honest(io, party);
	prot->set_batch_size(1024*1024);//set it to number of bits in BOB's input

//	cout <<"testing int to float";
//    int res;
//	Integer a(32, 17291, ALICE);
//	Float b = IntToFloat(a);
//	cout <<b.reveal<double>(PUBLIC)<<endl;

	vector<long long> res;
	if(party == ALICE) {
        vector<long long> alice;
        fileToVector(argv[3], alice);
		element_wise(res, alice, &round);
		ofstream out(argv[4]);
		std::copy(alice.begin(), alice.end(),
				         std::ostream_iterator<double>(out, "\n"));
		cout << "done writing to output" << endl;
	} else {
	    vector<long long> bob;
	    fileToVector(argv[3], bob);
		element_wise(res, bob, &round);
	}

	delete io;
}



//int main(int argc, char** argv) {
//	int port;
//	parse_party_and_port(argv, &party, &port);
//	NetIO * io = new NetIO(party==ALICE ? nullptr : "127.0.0.1", port);
//
//	auto prot = setup_semi_honest(io, party);
//	prot->set_batch_size(1024*1024);//set it to number of bits in BOB's input
//
//	//vector<int> noise;// = {0, 1, 2};
//	//vector<int> logits;// = {0, 3, 1};
//
//	//fileToVector("logits1.txt", logits);
//	//fileToVector("noise1.txt", noise);
//
//	//cout << "vectors opened" << endl;
//	int res;
//
//	if(party == ALICE) {
//		vector<long long> noise;
//		//logits = {0, 3, 1};
//		fileToVector(argv[3], noise);
//		//for (int i=0; i<noise.size();i++){
//		//	cout << noise[i] << endl;
//		//}
//		vector<long long > rr;
//		res = argmax(noise, rr);
//		cout << "argmax: " << res << endl;
//		//for(int i=0; i < rr.size(); i++){
//		//	cout << rr[i] << endl;
//		//}
//		rr[res] = rr[res] + 1;
//		ofstream out(argv[4]);
//		std::copy(rr.begin(), rr.end(),
//				         std::ostream_iterator<int>(out, "\n"));
//		//for(int i=rr.size()-1;i>=0;i--)
//		//	    out<<rr[i]<<"\n";
//		cout << "done writing to output" << endl;
//		//if (out.is_open()){
//		//	out << res;
//		//}
//		delete io;
//
//	}
//     	else {
//		vector<long long> logits;
//		vector<long long> temp;
//		//noise = {0, 1, 2};
//		fileToVector(argv[3], logits);
//		res = argmax(logits, temp);
//	}
//	//out[res] = out[res] + 1;
//	//cout << "writing to output" << endl;
//	//ofstream out ("output1.txt");
//	//if (out.is_open())
//	//{
//	//	out << res;
//	//}
//	//delete io;
//}