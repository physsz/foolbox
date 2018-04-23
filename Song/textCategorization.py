from nltk.corpus import wordnet as wn
import os
import csv
path= r"C:\Users\Song\Documents\Github\foolbox\Song"
os.chdir(path)

filename = open("cat.txt",'r')
catSyn = csv.reader(filename)
count=0
for row in catSyn:
    count = count +1
    print(row)
    if count == 2:
        break
    
    