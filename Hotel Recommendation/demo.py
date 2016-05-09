# -*- coding: UTF-8 -*-
import os

print os.path.abspath('../../kaggle data')
with open(os.path.abspath('../../kaggle data/test'), 'w')as f:
    f.write('Hello World')
