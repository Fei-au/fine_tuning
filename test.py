
from itertools import permutations,combinations, islice



print(list(islice(range(10), 2, 8, 2)))  # [2, 4, 6]

class MyClass:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "hehe"

    def __repr__(self):
        return self.name
