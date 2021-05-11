import sys
import time
import threading as th
import _thread
import concurrent.futures as ccf

R = {1: True}
C = 0

class CustomException(Exception):
    pass

class StoppableThread(th.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._stop_event = th.Event()
        
        # sys.settrace(self.localtrace)
        self.killed = False
    
    def excepthook(self, args):
        print('Caught')
        print(args)
        raise CustomException()
        
    def localtrace(self, *args, **kwargs):
        if self.killed:
            raise CustomException()
            
    def stop(self):
        # raise SystemExit()
        
        sys.exit()
        # raise Exception('This was raised.')
        

    
def f():
    global C
    while C < 20:
        C += 1
        time.sleep(1)
        print(R, C)
        
def f():
    for i in range(10):
        print(i)
        time.sleep(1)
    return 5

executor = ccf.ThreadPoolExecutor(max_workers=1) 
future   = executor.submit(f)
    
ccf.wait(fs=[future], timeout=3)


# T = StoppableThread(target=f)
# T.start()
# time.sleep(3)
# _thread.interrupt_main()
# try:
#     raise Exception()
# except:
#     pass

# print('aaa')

# def timeout(timelimit=1):
#     def wrapper(func):
#         def helper(*args, **kwargs):
#             to_run = lambda : func(*args, **kwargs)
            
            
#         return helper
#     return wrapper


