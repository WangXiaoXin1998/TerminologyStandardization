

# data = []
# with open('data/code.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         right = line.split('\t')[1]
#         right = right.strip()
#         data.append(right)
#
# print(data[0:10])
#
# import pandas as pd
#
# csv_ans = pd.DataFrame(data=data, columns=['code'])
# csv_ans.to_csv('data/code.csv', index=False)

import pandas as pd

or1 = pd.read_csv('data/train.csv')['org']
or2 = pd.read_csv('data/val.csv')['org']

data1 = pd.read_csv('data/train.csv')['stad']
data2 = pd.read_csv('data/val.csv')['stad']
code = pd.read_csv('data/code.csv')['code']
import numpy as np
or1 = np.array(or1)
or2 = np.array(or2)
data1 = np.array(data1)
data2 = np.array(data2)
code = np.array(code)
code_dic = {}
for i in range(len(code)):
    code_dic[code[i]] = i
print(code_dic)

ans_or1 = []
label_01 = []
for i in range(len(data1)):
    if code_dic.get(data1[i], -1) != -1:
        ans_or1.append(or1[i])
        label_01.append(code_dic[data1[i]])

ans_or2 = []
label_02 = []
for i in range(len(data2)):
    if code_dic.get(data2[i], -1) != -1:
        ans_or2.append(or2[i])
        label_02.append(code_dic[data2[i]])

csv_ans = pd.DataFrame(data=ans_or1, columns=['org'])
csv_ans['stad'] = label_01
csv_ans = csv_ans[['org', 'stad']]
csv_ans.to_csv('data/pre_train.csv', index=False)

csv_ans = pd.DataFrame(data=ans_or2, columns=['org'])
csv_ans['stad'] = label_02
csv_ans = csv_ans[['org', 'stad']]
csv_ans.to_csv('data/pre_val.csv', index=False)



