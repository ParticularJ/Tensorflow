class Parent(object):
    
    def override(self):
        print "Parent override()."
    
    def implicit(self):
        print "Parent implicit()."

    def altered(self):
        print "Parent alterde()."

class Child(Parent):
    def __init__(self, stuff):
        self.stuff = stuff
        super(Child, self).__init__()

    def override(self):
        print "child override()."
    
    def altered(self):
        print "Child, before Parent altered()"
        super(Child, self).altered()
        print "Child, after Parent altered()."

dad = Parent()
son = Child("i")

dad.implicit()
son.implicit()

dad.override()
son.override()

dad.altered()
son.altered()
