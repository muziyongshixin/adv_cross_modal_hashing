import json


file_path='C:\\Users\\木子-勇士心\\Desktop\\caps\\train_caps.txt'
data={}

line_cnt=0
with open(file_path,'rb') as f:
    for line in f.readlines():
        line=line.strip()
        data[line]=0
        line_cnt+=1

print(line_cnt)
print(len(data))



