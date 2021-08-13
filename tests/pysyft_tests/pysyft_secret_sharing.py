import torch as th
import syft as sy
# hook = sy.TorchHook(th)

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
secure_worker = sy.VirtualWorker(hook, id="secure_worker") # used to speed up computation by generating reliably random numbers

x = th.tensor([0.1, 0.2, 0.3])

x = x.fix_prec() # fixed precision encoding, turning floats into ints

x = x.share(alice, bob, secure_worker) # secret sharing by splitting the number in shares managed by independent entities