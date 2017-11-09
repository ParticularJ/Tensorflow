# creat a mapping of state to abbreviation
from collections import OrderedDict
states = {
    'Oregon': 'OR',
    'Florida': 'FL',
    'California': 'CA',
    'New York': 'NY',
    'Michigan': 'MI'
}
ordered_dict = OrderedDict(states)
# creat a basic set of states and some cities in them
cities = {
    'CA': 'San Francisco',
    'MI': 'Detroit',
    'FL': 'Jacksonvile'
}

#add some more cities
cities['NY'] = 'New York'
cities['OR'] = 'Portland'
ordered1_dict = OrderedDict(cities)
#print out some cites
print '-'*10
print "NY State has: ", cities['NY']
print "OR State has: ", cities['OR']

#print some states
print '-'*10
print "Michigan's abbreviation is: ", states['Michigan']
print "Florida's abbreviation is: ", states['Florida']

#do it by using the state then cities dict
print '-'*10
print "Michigan has: ", cities[states['Michigan']]
print "Florida has: ", cities[states['Florida']]

#print every state abbreviation
print '-'*10
for state,abbrev in ordered_dict.items():
    print "%s is abbreciated %s" % (state, abbrev)
#print regular 
print '-'*10
for state,abbrev in states.items():
    print "%s is abbreciated %s" % (state, abbrev)

#print every city in state
print '-'*10
for abbrev,city in ordered1_dict.items():
    print "%s has the city %s" % (abbrev, city)

#now do both at the same time
print '-'*10
for state, abbrev in states.items():
    print "%s state is abbrevied %s and has city %s" % (
        state, abbrev, cities[abbrev])

print '-'*10
# safety get a abbreviation by state that might not be there
state = states.get('Texas', None)

if not state:
    print "Sorry, no Texas."

# get a city with a default value
city = cities.get('TX', 'Does Not Exist')
print "The city for the state 'TX' is: %s" % city
