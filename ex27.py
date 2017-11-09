i = 0 
numbers = []

def judge(num):
    if num <= 6:
        for num in range(0,6):
            print "At the top num is %d." % num
            numbers.append(num)
            num = num + 1
            print "Numbers now: ", numbers
            print "At the bottom num is %d." % num

judge(int(raw_input("> ")))


print "The numbers: "

for num in numbers:
    print num
