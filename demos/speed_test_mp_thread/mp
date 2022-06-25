import multiprocessing
import time
import os
import timeit
import numpy as np
multiprocessing.set_start_method('fork')


def f():
    pass

def g():
    p1 = multiprocessing.Process(target=f)

    p1.start()
    p1.join()

stmt = "g()"

mp_run_times = timeit.repeat(
    stmt, number=1, repeat=5000, globals=globals())


print(np.sum(mp_run_times))
