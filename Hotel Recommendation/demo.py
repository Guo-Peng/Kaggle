# -*- coding: UTF-8 -*-
import os

print os.path.abspath('../../kaggle data/hotel_recommendation')
with open(os.path.abspath('../../kaggle data/hotel_recommendation/test'), 'w')as f:
    f.write('test')
