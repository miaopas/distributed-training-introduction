import os
import timeit
import numpy as np
import threading

def f():
    pass



def g():
    t1 = threading.Thread(target=f)

    t1.start()
    t1.join()

stmt = "g()"

mp_run_times = timeit.repeat(
    stmt, number=1, repeat=5000, globals=globals())


print(np.sum(mp_run_times))
