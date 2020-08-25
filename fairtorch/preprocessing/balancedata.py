import numpy as np
import pandas as pd

'''
data: path to data directory organized as follows:
data
	class_a
		example1
		example2
		...
	class_b
		example3
		...
	class_c
		example4
		...

groups: number of groups of data to split into

returns names of examples in each group of data
'''

def stratified_split(data, groups):
	
