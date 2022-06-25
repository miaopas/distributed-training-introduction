import threading
import time
import os

def f(x):
    print(f'Current in PID {os.getpid()}, thread t1 with id {threading.get_ident()}')
    x.append(1)

x = []

t1 = threading.Thread(target=f, args=(x,))
print(f'Current in PID {os.getpid()}, main thread with id {threading.get_ident() }')



t1.start()
t1.join()

print(f'After running, x in main is {x}')