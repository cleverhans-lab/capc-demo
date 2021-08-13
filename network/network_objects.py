

class InferenceRequest:
    """
    Step 1a.
    """

    def __init__(self, encrypt_query):
        self.encrypt_query = encrypt_query 


class EncryptLogits:
    """
    Step 1b.
    """

    def __init__(self, encrypt_logits):
        self.encrypt_logits = encrypt_logits


class SecretShareAPtoPG:
    """
    Step 2.
    """

    def __init__(self, s_hat):
        self.s_hat = s_hat


"""
Step 1c - via C++ argmax.
Step 3 - via C++ sum histogram.
"""