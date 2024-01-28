import random


print(random.random())
print(random.random())
print(random.random())
print(random.random()) # różne wyniki, oparte o czas

random.seed(10) # losuje w oparciu o seed (10)
print(random.random())
random.seed(10) #ten sam wynik
print(random.random())
random.seed(10)
print(random.random())
random.seed(10)
print(random.random()) # różne wyniki, oparte o czas


print('It is my dog. Nothing else.'.replace('N', 'n').replace('. ', '   ')) #metoda bezpośrednio na stringu bez zapisywania go do zmiennej