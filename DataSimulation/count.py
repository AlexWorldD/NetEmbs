#!/usr/bin/python
import os
import sys

def CountFile(f):
    counter = 0
    f = open(f, "r")
    for line in f.read().split('\n'):
        counter = counter + 1
    f.close()
    return counter

def CountDir(dirname):
    counter = 0
    for f in os.listdir(dirname):
        fa = os.path.join(dirname, f)
        if os.path.isdir(fa):
            dcount = CountDir(fa)
            counter = counter + dcount
        else:
            fcount = CountFile(fa)
            counter = counter + fcount
    return counter

print CountDir(sys.argv[1])