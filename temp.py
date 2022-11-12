import numpy as np
class Test(object):
    def func(self):
        from utils.config_utils import _CONFIG_STORE

        print("testing string: ", _CONFIG_STORE)

np.save("/tmp/scores.npy", np.zeros(1))