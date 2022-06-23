from joblib import Parallel,delayed
import numpy as np
import os

def testfunc(data):
    # some very boneheaded CPU work
    for nn in range(1000):
        for ii in data[0,:]:
            for jj in data[1,:]:
                ii*jj

def run(niter=10):
    os.system("taskset -p 0xff %d" % os.getpid())
    data = (np.random.randn(2,100000) for ii in range(niter))
    pool = Parallel(n_jobs=-1,verbose=1,pre_dispatch='all')
    results = pool(delayed(testfunc)(dd) for dd in data)

if __name__ == '__main__':
    run()