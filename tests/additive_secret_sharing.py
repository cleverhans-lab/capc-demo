import random


def additive_share(secret, Q=41, N=2):
    shares = [random.randrange(Q) for _ in range(N - 1)]
    shares += [(secret - sum(shares)) % Q]
    return shares


def additive_reconstruct(shares, Q=41):
    return sum(shares) % Q


def main():
    secret = 4
    print('secret: ', secret)

    shares = additive_share(secret=secret)
    print('shares: ', shares)
    secret_no_noise = additive_reconstruct(shares=shares)
    print('secret_no_noise: ', secret_no_noise)

    shares[0] += 0.1
    print('noisy shares: ', shares)
    secret_with_noise = additive_reconstruct(shares=shares)
    print('secret_with_noise: ', secret_with_noise)


if __name__ == "__main__":
    main()
