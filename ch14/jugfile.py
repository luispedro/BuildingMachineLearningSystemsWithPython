# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

from jug import TaskGenerator
from time import sleep


@TaskGenerator
def double(x):
    sleep(4)
    return 2 * x


@TaskGenerator
def add(a, b):
    return a + b


@TaskGenerator
def print_final_result(oname, value):
    with open(oname, 'w') as output:
        output.write("Final result: {0}\n".format(value))

input = 2
y = double(input)
z = double(y)

y2 = double(7)
z2 = double(y2)
print_final_result('output.txt', add(z, z2))
