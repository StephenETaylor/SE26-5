#!/usr/bin/env python
"""
score a predictions.json file, assuming that the input file was the ../dev.json file.
"""
import json
import math
import sys
import train4


pfile = 'predictions4.json'  #this file isn't actually a json file...

if len(sys.argv)>1:
    pfile = sys.argv[1]

data_file = '../dev.json'
with open(data_file) as fi:
    data = json.load(fi)

count = totalp = totalg = rights = wrongs = 0
with open(pfile) as fi:
    for lin in fi:
        z = json.loads(lin)
        prediction = z['prediction']
        key = z['id']
        example = data[key]
        gold_av = example['average']
        gold_stdev = example['stdev']
        gold_score = math.floor(gold_av + 0.5)
        count += 1
        totalp += prediction
        totalg += gold_score
        if prediction < gold_av+gold_stdev and prediction > gold_av-gold_stdev:
            rights += 1
        else: 
            wrongs += 1
            #remove code below temporarily
            #av,st,te = train4.evalu(example,key)
            pass
            
        # if I were a little more ambitious I could do an F-score ...
        pass
accuracy = rights / count
av_pred = totalp / count
av_gold = totalg / count

print(f'{count} records, {rights} correct predictions,\n '
      f'accuracy: {accuracy}\n mean_prediction: {av_pred} mean_gold = {av_gold}')
