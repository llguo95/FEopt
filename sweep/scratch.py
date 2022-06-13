# content = open('wirebond_sweep/01_Geometric_Inputs.txt', 'r+').read()
# new_content = 'Wthk = 0.3\n\nFL = 1\n\n' + content[content.index("! Total number of layers in an IGBT"):]
# text_file = open('wirebond_sweep/01_Geometric_Inputs.txt', 'r+')
# text_file.write(new_content)
# text_file.close()

# content = open('wirebond_sweep/Max_Strain_0.3_1.0.txt', 'r+').read()
# # i = content.index('EPS_max')
# print(float(content))

import pandas as pd
df1 = pd.read_csv('input_output_part1.csv')
df2 = pd.read_csv('input_output_part2.csv')

df = pd.concat((df1, df2), ignore_index=True)

df[['WThk', 'FL', 'EPS_max']].to_csv('input_output.csv', index=False)