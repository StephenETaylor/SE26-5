#!/usr/bin/env python3
"""
This file is intended to retrieve data from a file of synsets and bias words,
calculated from the English wordnet data using the Pilehvar and Collier 2016 
algorithm 1.

Rather than sort the file in memory, we do an external sort, using:
    LC_ALL=C sort biases.txt > biases.sort2
This sort ignores language locale, but lets us use byte string comparisons
on the data; it would have been easy to do string comparisons instead, but
I lack confidence that python string compare honors LC_COLLATE.

API:
    This is code is intended to be used as a module.
    search()
"""
import math
from nltk.corpus import wordnet as wn
import numpy as np
import os
import sys

#tiny UI
Verbose = True

FileName = '../biases.sort2'
#wc for ../biases.sort:
#  (reran with algorithm modifications)
# 117659  6000609 51595688
#Lines = 117659 # was 109096  
#Words = 6000609 # was 5563896 
Bytes = os.stat(FileName).st_size # was 51595688 # was 62463323 

# as it happens, the current version of biases.txt is for English,
# and so the characters are almost all ASCII, so bytes and characters are same

BytesPerLine = math.ceil(51595688/117659) #math.ceil(Bytes/Lines)

# read in the file
Handle = os.open(FileName,os.O_RDONLY)
File = os.read(Handle,Bytes)
assert Bytes == len(File)

LinesQ = Bytes/BytesPerLine
ten = 10
while LinesQ <= 5*2**ten and ten>1:
    ten += -1
nable_size = 2**ten  # was formerly 1024
# Table and Nable are to start the search of file, a short hilevel index
# Table is strings, the synsets of the biases file
# Nable is numbers, the offsets in File of the corresponding synset
Table = np.ndarray(nable_size, dtype = object)
Nable = np.ndarray(nable_size, dtype = np.uint32)

# set up the index:
for i in range(nable_size):#enumerate(range(0, Bytes, math.ceil(Bytes/(nable_size)))):
    # first interval of File, starts immediately with label
    if i == 0:
        cp = beg = 0
    else:#find first newline after beg
        beg = int(Bytes*i/nable_size)
        maybe_end = min(Bytes, int(Bytes*(i+1)/nable_size))
        cp = File.find(b'\n',beg)
        if cp < 0 or cp>= maybe_end:
            # did not find newline!
            raise Exception('interval did not include newline')

        cp += 1
    # read in synset name after newline
    Nable[i] = cp
    send = File.find(b' ',cp)
    if i > 126:  #debugging
        pass
    Table[i] = File[cp:send] # was: str(File[cp:send], 'utf8')
                             # but file is bytes, not str, so 

Already_failed = set()

def search(synset):
    """
    perform a search for the synset in the cached version of the 
    biases file.  
    Return the byte offset of the first character of the synset.
    """
    # first search the in-memory Table/Nable pairs to find the interval in File
    bsynset = bytes(synset, 'utf8')
    i = Table.searchsorted(bsynset, side='left')
    if i == nable_size:   # this item is somewhere beyond the last Table entry
        beg = Nable[i-1]
        end = Bytes
    
    elif Table[i][:-1] == bsynset:
        return Nable[i]

    elif i == 0:
        return 0 # this is the offset of the first item in file.  But we might
                 # actually have searched for something earlier?

    else:
        beg = Nable[i-1]
        end = Nable[i]

    # Now do a binary search on the located File interval (from beg to end)
    oldMiddle = None
    while end-beg > BytesPerLine:
        middle = (beg+end)//2
        if oldMiddle is not None and middle == oldMiddle:
            # in this case, beg and end also have the same values as previously
            # this can happen if middle occurs in the buffer after 
            # a newline, or if there is no newline in the buffer
            better = File.find(b'\n',beg)
            if better > 0 & better < end-1:
                middle = better-1
            else:
                nana() # error
                break  # I guess this is the case with no newline.  I don't know if it happens
        oldMiddle = middle
        sbeg = File.find(b'\n',middle)
        if sbeg < 0: return beg
        middle = sbeg+1
        if middle == end:  return beg
        ssend = File.find(b': ',middle)
        ss = File[middle:ssend] # was: str(File[middle:ssend], 'utf8')
        if ss < bsynset: 
            beg = middle
        elif ss > bsynset:
            end = middle
        else: #must be equal
            return middle

    # report failure only once.
    if synset not in Already_failed:
        if Verbose: print(f'Failed search for synset {synset}, beg = {beg}: {File[beg:beg+20]}' 
          +f' end = {end}: {File[end:end+20]}')
        Already_failed.add(synset)
    return None
    #harrumph()
    #raise SystemExit(1)   # error.  I had harrumph() here, which gives traceback

Count_inventions = 0 # count how many times I ask for a synset not in File.

def get_bias_words(synset):
    """
    uses search() to find the File offset of the synset line,
    and then returns a python list of the bias words, as utf8 strings.

    Currently there are about 8500 'old-maid synsets'  which don't have
    outgoing relation arcs in the wordnet relation network.  These
    synsets do not appear in the bias_words file.  A fix would be to 
    add new relations, perhaps based on the dictionary definition of the
    synset, but I hesitate to do this.  Instead I put in a fix to add the
    non-stopwords of the dictionary definitiion as the (small) set of bias 
    words.  This may have no value at all...
    """
    global Count_inventions
    offset = search(synset)
    if offset is None:
        sobject = wn.synset(synset) # find the object for the synset name.
        definition = sobject.definition()
        retval = definition.split(" .?!,'`")
        retval += [x._name for x in sobject.lemmas()]
        #retval.remove('');
        Count_inventions += 1
        return retval
    offend = 2 + File.find(b': ', offset)
    listend = File.find(b'\n',offend)
    liststring = str(File[offend:listend], 'utf8')
    retval = liststring.split(' ')
    return retval



if __name__ == '__main__':
    #i = search('electric_potential.n.01')
    #print(i,File[i:File.find(b'\n',i)])
    #print(get_bias_words('be.v.02'))
    #print(get_bias_words('restrain.v.01'))
    print(get_bias_words('wonder.n.01'))

