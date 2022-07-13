import multiprocessing
import time
import os
multiprocessing.set_start_method('fork')


def f(name,):
    print(f'Now in process {name} with PID: {os.getpid()} which is in group {os.getpgrp()}, the parent process is {os.getppid()}')
    
    time.sleep(20000)


print(f'Main have pid:{os.getpid()}')
p1 = multiprocessing.Process(target=f, args=('p1',))

p1.start()
p1.join()
