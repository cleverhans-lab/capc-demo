from phe import paillier


def main():
    keyring = paillier.PaillierPrivateKeyring()

    public_key, private_key = paillier.generate_paillier_keypair()
    keyring.add(private_key)

    public_key1, private_key1 = paillier.generate_paillier_keypair(keyring)
    keyring.add(private_key1)

    public_key2, private_key2 = paillier.generate_paillier_keypair(keyring)
    keyring.add(private_key2)

    secret_number_list = [3.141592653, 300, -4.6e-12]
    print('secret number list: ', secret_number_list)

    encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]

    print('encrypted: ', encrypted_number_list)

    decrypted_number_list = [private_key.decrypt(x) for x in
                             encrypted_number_list]

    print('decrypted: ', decrypted_number_list)

    decrypted_number_list2 = [keyring.decrypt(x) for x in encrypted_number_list]

    print('decrypted2: ', decrypted_number_list2)

    a, b, c = encrypted_number_list

    a_plus_5 = a + 5
    b_times_2 = b * 2
    c_times_e12 = c * 10e12

    new_encrypted_list = [a_plus_5, b_times_2, c_times_e12]

    new_encrypted_list_decrypted = [keyring.decrypt(x) for x in
                                    new_encrypted_list]

    print('new list decrypted: ', new_encrypted_list_decrypted)


if __name__ == "__main__":
    main()
