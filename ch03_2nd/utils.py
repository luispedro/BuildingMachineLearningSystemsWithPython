# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

import os
import sys

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

if not os.path.exists(DATA_DIR):
    print("Uh, we were expecting a data directory, which contains the toy data")
    sys.exit(1)

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")
if not os.path.exists(CHART_DIR):
    os.mkdir(CHART_DIR)

