from audioop import mul
from concurrent.futures import thread
import multiprocessing
from  threading import Thread
import threading
import time
import os
from turtle import rt
def f():
    print(os.getpid())
    print(os.getpgid(os.getpid()))
    print(f'number of threads {threading.enumerate()}')
    t1 = Thread(target=g)
    t1.start()


def g():
    time.sleep(2)

print(__name__)

if __name__ =='__main__':
    # multiprocessing.set_start_method('fork')
    print(os.getpid())


    p1 = multiprocessing.Process(target=f)
    p2 = multiprocessing.Process(target=f)
    print(f'number of threads {threading.enumerate()}')
    p1.start()
    p2.start()

    p1.join()
    p2.join()
