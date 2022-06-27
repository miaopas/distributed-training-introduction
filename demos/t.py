from scipy import integrate
import numpy as np





def run(n, l):
    def f(t,x):
        return -x*t - x**2 + np.sin(x)
    res = []
    for _ in range(n):
        res.append(integrate.RK45(f, t0=0, y0=np.random.random(10) , t_bound=10))
    # l.append(res)


import multiprocessing
from multiprocessing import Manager
import random
try:
    multiprocessing.set_start_method('fork')
except Exception:
    pass

n = 100_000

cpus = 2


with Manager() as manger:
    shared_list = manger.list()

    processes = []

    for _ in range(cpus):
        p = multiprocessing.Process(target=run, args=(n //cpus,shared_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    res = list(shared_list)