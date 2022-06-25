from ast import arg
from audioop import mul
from concurrent.futures import thread
import multiprocessing
from  threading import Thread
import threading
import time
import os
def f(name, p):
    if name == 'p1':
        os.setpgid(os.getpid(), os.getpid())
    if name == 'p2':
        os.setpgid(os.getpid(), p)
    print(f'Now in process {name} with pid: {os.getpid()} which is in group {os.getpgrp()}')
    
    time.sleep(20000)


# print(f'{__name__} have pid {os.getpid()}')

if __name__ =='__main__':
    # multiprocessing.set_start_method('forkserver')

    print(f'Main have pid:{os.getpid()}')
    p1 = multiprocessing.Process(target=f, args=('p1',None))
    p1.start()


    p2 = multiprocessing.Process(target=f, args=('p2',p1.pid))
    p2.start()

    
    # os.setpgid(p1.pid, p2.pid)

    p1.join()
    p2.join()
