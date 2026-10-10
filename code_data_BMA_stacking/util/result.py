import chardet #获取文件编码格式
import os

outputpath = 'outputs_BMA'
n = 0
total = 0
r = 0
files = []
for filename in os.listdir(outputpath):
    if filename.endswith('.txt'):
        files.append(filename.split('_')[-1])  # gauge名称


outputpath1 = 'outputs_Stacking'
n = 0
total = 0
r = 0
files_Linear = []
files_Ridge = []
for filename in os.listdir(outputpath1):
    if filename.endswith('.txt'):
        stacking_type = filename.split('_')[-2]
        if stacking_type=='Linear': 
            files_Linear.append(filename.split('_')[-1])  
        elif stacking_type=='Ridge': 
            files_Ridge.append(filename.split('_')[-1]) 


 
set1 = set(files)
set2 = set(files_Ridge)
 
result = list(set1  - set2)
print(result)