import numpy as np

def mul(*arg):
    a = 1
    for n in arg:
        print
        a *= n
    return a

print(mul(1,2,3))
print(mul(*[1,2,3], *[1,2]))