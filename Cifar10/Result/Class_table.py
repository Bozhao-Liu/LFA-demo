import os
import pandas as pd
import json
from collections import defaultdict
import numpy as np

def create_table(dir = '.'):
	latex_tables = {}

	latex_tables= defaultdict(list)
	stat_files = os.listdir(dir)
	stat_files.sort()
	for stat_file in stat_files:
		if 'eval_matrix_' in stat_file: 
			method = stat_file.replace('eval_matrix_', '')
			method = method.replace('.json', '')
			
			with open(os.path.join(dir, '{}'.format(stat_file))) as f:
				data = json.load(f)
				for i in range(10):
					latex_tables['method'].append(method + ' class ' + str(i+1))
					for key, value in data.items():
						m = np.array(value).mean(axis = 0)
						sd = np.array(value).std(axis = 0)
						latex_tables[key].append('{}'.format(round(m[i], 2)) + u'\u00B1' + '{}'.format(round(sd[i], 2)))
							
	latex_table = pd.DataFrame(latex_tables)
	table = latex_table.to_latex(index = False)
	with open(os.path.join(dir, "Multilabel.tex"), "w") as f:
		f.write(table)

if __name__ == '__main__':
	create_table('.')
