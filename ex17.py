# -*- coding : utf-8 -*-

# This one is like your scripts with argv

def print_two(*args):
    arg1, arg2 = args
    print "arg1: %r, arg2: %r" % (arg1, arg2)

# ok, that *argv is actually pointless, we can just do this
def print_two_again(arg1,arg2):
    print "arg1: %r, arg2: %r" % (arg1, arg2)

# This just takes one argument
def print_one(arg1):
    print "arg1: %r" % arg1

# This one takes no argument
def print_none():
    print "I got nothing."


print_two("CK","GJ")
print_two_again("CK","GJ")
print_one("First!")
print_none()
