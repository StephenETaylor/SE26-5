#!/usr/bin/env python3
"""
concatenate two json files.

"""

#tiny UI
file1 = '../train.json'
file2 = '../dev.json'

output_file = '../train_dev.json'

import json

with open(output_file,'w') as fo:
    print('{', file = fo)
    key = 0
    for f in file1,file2:
        with open(f) as fi:
            content = json.load(fi)
            for k,v in content.items():
                if key != 0:
                    print(',', file = fo)
                print(f'"{key}": {json.dumps(v)}', file = fo)
                key += 1
            print(f'file {f} had {len(content)} records')
    print('}', file = fo)
print (f'{key} records output')

#check:
with open(output_file) as fi:
    x = json.load(fi)
    for ke,va in x.items():
        if int(ke) > key:
            scream()
    print('last ke value is ',ke)






