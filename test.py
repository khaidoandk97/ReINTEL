import pandas as pd 
import numpy as np 
import emot

# load data using utf8 format
data = pd.read_csv('./datasets/public_train.csv', encoding='utf-8')

data.drop()

dup = data.duplicated(subset='brand')
print(data[dup].sort_values())

