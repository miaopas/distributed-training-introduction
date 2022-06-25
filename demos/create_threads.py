import threading
import time
import os

def f():
    print(f'Current in PID {os.getpid()}, thread t1 with id {threading.get_ident()}')
    time.sleep(2000)

t1 = threading.Thread(target=f, args=())
print(f'Current in PID {os.getpid()}, main thread with id {threading.get_ident() }')

t1.start()
t1.join()

