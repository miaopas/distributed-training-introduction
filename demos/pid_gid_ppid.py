import time
import os

print(f'PID of current process is {os.getpid()}.')
print(f'Parent of current process is {os.getppid()}')
print(f'Group if of current process is {os.getpgrp()}.')
time.sleep(2000)