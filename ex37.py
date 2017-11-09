## Animal is-a object (yes, sort of confusing) look at the extra credit

class Animal(object):
    pass

## dog is-a animal
class Dog(Animal):
    
    def __init__(self, name):
        ## dog has-a name
        self.name = name

## cat is-a animal
class Cat(Animal):

    def __init__(self, name):
        ## cat has-a name
        self.name = name

## Person is-a object
class Person(object):
    
    def __init__(self, name):
        ## has-a name
        self.name = name

        ## Person has-a pet of some kind
        self.pet = None

## Employee is a person
class Employee(Person):
    
    def __init__(self, name, salary):
        ## has-a hmm what is this strange magic?
        super(Employee, self).__init__(name)
        ## has-a salary
        self.salary = salary

## Fish is-a object
class Fish(object):
    pass

## is-a
class Salmon(Fish):
    pass

## is-a
class Halibut(Fish):
    pass

## rover is-a Dog
rover = Dog("Rover")

## satan is-a cat
satan = Cat("Satan")

## Mary is-a person
mary = Person("Mary")

## has-a
mary.pet = satan

## frank is-a employee
frank = Employee("Frank", 1200000)

## has-a 
frank.pet = rover

##flipper is-a
flipper = Fish()

## crouse is a 
crouse = Salmon()

## is-a
harry = Halibut() 
