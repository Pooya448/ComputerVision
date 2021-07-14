import json

notebook_file = open('HW8.ipynb', 'r')
notebook = json.load(notebook_file)
notebook_file.close()

questions = ['Q2', 'Q3']
notebook_index = 0

for question in questions:
	question_file = open(question + '.py', 'w')
	start_write = False
	while notebook_index < len(notebook['cells']):
		if notebook['cells'][notebook_index]['cell_type'] == 'markdown' and 'dont change this cell' in notebook['cells'][notebook_index]['source'][0]:
			notebook_index += 1
			break
		if start_write:
			if notebook['cells'][notebook_index]['cell_type'] == 'code' and len(notebook['cells'][notebook_index]['source']) > 0:
				source = notebook['cells'][notebook_index]['source']
				for line in source[:-1]:
					question_file.write(line)
				question_file.write(source[-1] + '\n\n')
		if notebook['cells'][notebook_index]['cell_type'] == 'markdown' and 'PART %s'%(question[1:]) in notebook['cells'][notebook_index]['source'][0]:
			start_write = True
		notebook_index += 1
	question_file.close()