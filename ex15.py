# -*- coding: utf-8 -*-

from sys import argv

script, filename = argv

print "We're going to erase %r." % filename
print "If you don't want that, hit ctrl+c."
print "If you do want that, hit RETURN."

raw_input("?")

print "Opening the file..."
target = open(filename, 'w')

print "Truncating the file. Goodbye!"
target.truncate()

print "Now I'm going to ask you for three lines."

line1 = raw_input("line 1: ")
line2 = raw_input("line 2: ")
line3 = raw_input("line 3: ")

print "I'm going to write these to the file."

target.write(line1+'\n'+line2+'\n'+line3+'\n')

print "Finally, we close it."
target.close()

print "Please type the file."
target_again = raw_input("> ")
target_txt = open(target_again)
print target_txt.read()
target_txt.close() 
