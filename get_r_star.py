import numpy as np
import secrets


def get_rstar_server(batch_size, num_classes, seed=4567):
    """
    Generate the cryptographically secure random sequence of numbers for secret
    sharing.

    :param batch_size: the number of samples in the mini-batch
    :param num_classes: number of classes to predict from
    :param seed: a prime number (here applied for reproducibility)
    :return: r* - the number of secret sharing
    """

    # need: find the precise 2**47 value
    # val = np.random.uniform(low=-2 ** 46, high=2 ** 46,
    #                         size=(batch_size, num_classes))
    q = 140737488273409
    q2 = (q - 1) / 2
    # For floats, below truncates towards 0, e.g., int(-8.2) = -8, int(8.2) = 8.
    # This is in case we change q and then q2 is no longer an even number.
    q2 = int(q2)
    low = -q2
    high = q2

    system_random = secrets.SystemRandom(seed)
    r_star = []
    for i in range(batch_size * num_classes):
        r_star.append(system_random.uniform(a=low, b=high))
    r_star = np.array(r_star).reshape((batch_size, num_classes))

    # This is for statistical simulation and not cryptographically secure.
    # r_star = np.random.uniform(
    #     low=low, high=high, size=(batch_size, num_classes))

    scale = 2 ** 24
    r_star /= scale
    return r_star


if __name__ == "__main__":
    rstar = get_rstar_server(batch_size=1, num_classes=10)
    print('rstar1: ', rstar)

    rstar = get_rstar_server(batch_size=8, num_classes=10)
    print('rstar2: ', rstar)

    from utils.main_utils import array_str

    rstar = get_rstar_server(batch_size=1, num_classes=10)
    print('rstar: ', array_str(rstar))
