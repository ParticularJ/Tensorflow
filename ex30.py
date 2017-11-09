class Sample:
    def __enter__(self):
        print "In_enter_()"
        return "Foo"
    
    def __exit__(self, type, value, trace):
        print "In _exit_()"

def get_sample():
    return Sample()

with get_sample() as sample:
    print "Sample:", sample 

#assert 1>0
assert 1<0
