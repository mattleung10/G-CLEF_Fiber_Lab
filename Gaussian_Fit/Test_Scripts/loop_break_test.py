#February 15, 2022
"""
#This script is just a test for threading
#https://stackoverflow.com/questions/13180941/how-to-kill-a-while-loop-with-a-keystroke/55822238#55822238
"""

import threading as th

keep_going = True
def key_capture_thread():
    global keep_going
    input()
    keep_going = False

def do_stuff():
    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
    ctr = 0
    while keep_going:
        print(ctr)
        ctr += 1

do_stuff()