import time
import os

import setproctitle
print(setproctitle.setproctitle('aas'))
print(f'PID of current process is {os.getpid()}.')
print(f'Parent of current process is {os.getppid()}')
print(f'Group if of current process is {os.getpgrp()}.')
time.sleep(2000)