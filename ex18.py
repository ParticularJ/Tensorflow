def cheese_and_crackers(cheese_count, boxes_of_crackers):
    print "You have %d cheeses!" % cheese_count
    print "YOu have %d boxes of crackers!" % boxes_of_crackers
    print "Man that's enough for a party!"
    print "Get a blanket.\n"


print "We can just give the function numbers directly:"
cheese_and_crackers(20, 30)

print "Or, we can use variables from our script:"
amount_of_cheese = 10
amount_of_crackers = 50

cheese_and_crackers(amount_of_cheese, amount_of_crackers)

print "We can even do math inside too:"
cheese_and_crackers(10+29, 23+23)

print "And we can combine the two, variables and math:"
cheese_and_crackers(amount_of_cheese+12,amount_of_crackers+10)

prompt = ">"
a=int(raw_input("count_of_cheese\n"+prompt))
b=int(raw_input("count_of_crackers\n"+prompt))

cheese_and_crackers(a,b)
