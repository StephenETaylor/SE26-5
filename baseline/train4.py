#!/usr/bin/env python3
"""
The main difference in train4.py is that it uses file version 4, in which
'average' is replaced in the binary file with a compressed version of the 
choices list.
Otherwise it is an extended version of train3.py, which will
presumably work with an extended version of fit2.py.
Train3 differences:
    The man difference is some new tests.  I haven't clearly decided whether
    the extension of fit2 will be fit3 or just a version check followed by 
    an import of train3. (since there will be a new test action, train2.evalu() 
    will not be able to provide it.)

    The intermediate file is a train.bin, a binary file, which is
    intended to include a record-describing profile.

    The preamble includes:
        version #:  1 byte uint8
        component count, synthetic component count, test count,  id count, 3 bytes
        ordered list basic components 8 bytes ASCII
        ordered list synthetic-components ids  4bytes? 8bytes? ASCII string
        test ids:  action, component, component (, component) 4 bytes


    Each record includes:
        <train-file example #>   an np.int32  4bytes

        <average> and <stdev> np.float32, 8bytes total
        <ordered list of test scores> np.float32, 4 bytes
        
My plan wrt the train/fit numbering scheme is that each new testset should
get a new number.  So train.py (no number) writes 5? test results to t.dat
train1 writes 30 test results to t.bin

"""
import bias_search as bs
import gensim.models as gm
import json
import math
from nltk.corpus import wordnet as wn
import numpy as np
import os
import sys
import time

start_time = time.time()

#preamble constants
Version = 4   #version 4 replaces "average" with compressed list of choices
Component_count = 0
Synthetic_count = 0
Test_count = 0

#tiny UI: "User Interface"
bs.Verbose = False
train_input_file = (f'../train.json') 
train_output_file = f't{Version}.bin'

if len(sys.argv)>1:
    train_input_file = sys.argv[1]

if len(sys.argv)>2:
    train_output_file = sys.argv[2]



global T

# set up the Emodel english model for set distances:
emodel = '../../English/model.bin'
Emodel = gm.KeyedVectors.load_word2vec_format(emodel, binary=True)
Emodel.unit_normalize_all() # cuts two divides from cos distance (but might
                            # not do so in gensim code ...

#Action constants for building synthetic components
Actions = dict()
Offset = dict()

Copy = 1 #set copy of parameter 1
Actions[Copy] = lambda l:T[l[0]]

Union = 2 #union of parameter 1, 2, 3 
Actions[Union] = lambda x: set.union(T[x[0]],T[x[1]]) if len(x) == 2 else set.union(T[x[0]],T[x[1]],T[x[2]])

Intersection = 3
Actions[Intersection] = lambda x: set.intersection(T[x[0]],T[x[1]]) if len(x) == 2 else set.intersection(T[x[0]],T[x[1]],T[x[2]])

SetDifference = 4

Lemmatize = 5 #lemmatize parameter
Actions[Lemmatize] = lambda x: lemma_set(T[x[0]])

Drop_underscores = 6  # drop words with an underscore

SetDistance = 7 # len(set intersection)/set(union)
Actions[SetDistance] = lambda x: score(T[x[0]],T[x[1]])

Bias_words = 8
Actions[Bias_words] = lambda x: bias_words(T[x[0]], -1)

First10 = 9
Actions[First10] = lambda x: bias_words(T[x[0]], 10)

First20 = 10
Actions[First20] = lambda x: bias_words(T[x[0]], 20)

Setof = 11 # convert string to set
Actions[Setof] = lambda x: wset(T[x[0]])

Synset = 12
Actions[Synset] = lambda x: get_wordnet_synset(T[x[0]], T[x[1]])

UnionStrings = 13
Actions[UnionStrings] = lambda x: set.union(wset(T[x[0]]),wset(T[x[1]]))

Drumroll = 14
Actions[Drumroll] = lambda x: drumroll(T[x[0]], T[x[1]], T[x[2]])

DrumCache = 15
Actions[DrumCache] = lambda x: Cached_triple[x[0]] if x[0]<3 else Cache_rank

Edistance = 16  # the mean euclidean distance between a synset ball and another
                # set of word-embeddings.   Takes the two sets as parameters.
Actions[Edistance] = lambda x: EuclideanDistance(T[x[0]],T[x[1]])

SynsetPolarity = 17 # polarity (+1, 0, -1) of a testset wrt homograph_synset
Actions[SynsetPolarity] = lambda x: synset_polarity(T[x[0]], T[x[1]], T[x[2]])

Functions = dict()  # functions for producing the value of an item
Components = []

def component(key):
    """
    build preamble 
    """
    global Preamble, Component_count, Components  #really only Component_count *needed*

    lk = len(key)
    for i in range(min(8,lk)):
        Preamble.append(ord(key[i]))
    for i in range(0,max(0,8-lk)):
        Preamble.append(0)
    retval = Component_count
    Offset[key] = retval
    Components.append(key)
    Component_count += 1
    Functions[key] = lambda : T[retval]
    return retval


Syntheses = []
def synthesis(key, action, lis):

    global Preamble  # since Preamble is a list object, appending doesn't require this
    global Syntheses # but I think it improves the readability; += appears to require!
    global Synthetic_count
    
    if len(lis) > 3: squawk()

    while len(lis) < 3:
        lis.append(0)

    Preamble.append(action)
    synth = f' lambda : Actions[{action}](['
    for i in range(3):
        ll = len(lis)
        if i< ll:
            Preamble.append(lis[i])
            if i != 0:
                synth += ', '
            synth += str(lis[i])

        else:
            Preamble.append(0)
    synth += '])'
    Syntheses .append( eval(synth) ) # compile the synthesis function, and place on list

    retval = Synthetic_count + Component_count
    Synthetic_count += 1
    Functions[key] = eval(f'lambda: T[{retval}]')
    Offset[key] = retval
    return retval

#Preamble : we'll build it up as a sequence of uint8, correcting counts at end
# finally, we'll turn it into Preamble, a byte sequence.
# the preamble seems potentially interpretable, but it is intended as
# documentation; only the first four bytes are examined at runtime, and
# synthetic components are generated based on file version, not their descriptor
Preamble = [Version,Component_count,Synthetic_count,Test_count]

example = component('example')
average = component('average')
stdev   = component('stdev')

# components below this point do not appear in the output file 't.bin'
component('homonym')
component('judged_meaning')

precontext = component('precontext')
sentence = component('sentence')
ending = component('ending')
example_sentence = component('example_sentence')
                          
# synthesized components  # These are not stored in the file

pre=synthesis('pre', Setof, [precontext])
sen=synthesis('sen', Setof, [sentence])
end=synthesis('end', Setof, [ending])

lpre=synthesis('lpre',Lemmatize, [pre])
lsen=synthesis('lsen',Lemmatize, [sen])
lend=synthesis('lend',Lemmatize, [end])

#ps= synthesis('ps',  Union,      [pre, sen])
#pse=synthesis('pse',  Union,      [pre, sen, end])
pe=synthesis('pe',  Union,      [pre, end])

#lps= synthesis('lps',  Union,      [lpre, lsen])
#lpse=synthesis('lpse',  Union,      [pre, lsen, lend])
lpe=synthesis('lpe',  Union,      [lpre, lend])

synset= synthesis('syn',Synset,  [Offset['homonym'], Offset['judged_meaning']])
biases= synthesis('bias', Bias_words, [synset])
bias10= synthesis('bi10', First10, [synset])
homies= synthesis('homies', UnionStrings, [Offset['judged_meaning'], Offset['example_sentence']])

# tests # these are stored in the binary file

Tests = []
def test(key, action, p1, p2, p3=0):
    """
    Set up Tests and test counts in preamble,
    as well as Tests array and Test_count
    """
    global Preamble, Test_count, Tests

    if action == SetDistance:
        tes = eval(f'lambda: score(T[{str(p1)}],T[{str(p2)}])')
    elif action == Drumroll:
        tes = eval(f"lambda: drumroll(T[{p1}], T[{p2}], T[{p3}])" )
    elif action == Edistance:
        tes = eval(f'lambda: EuclideanDistance(T[{p1}], T[{p2}])')
    elif action == SynsetPolarity:
        tes = eval(f'lambda: synset_polarity(T[{p1}], T[{p2}], T[{p3}])')
    else:
        tes = eval(f'lambda: {Actions[action]} ([{p1},{p2},{p3}])')
        # this is possibly useless caution...
        print (f'Unexpected test action: {action}')
        harrumph()
    Tests.append(tes)
    Preamble.append(action)
    Preamble.append(p1)
    Preamble.append(p2)
    Preamble.append(p3)
    Test_count += 1

FirstTest = Component_count + Synthetic_count
#for s in ['pre','sen','end','ps', 'pse']:
#    s1 = Offset[s]
#    for s2 in [biases, bias10, homies]:
#        test(s+str(s2), SetDistance, s1, s2)

#for s in ['lpre','lsen','lend','lps', 'lpse']:
for s in ['lpre','lend', 'lpe']:
    s1 = Offset[s]
    for s2 in [biases, bias10]:
        test(s+str(s2), SetDistance, s1, s2)

test('drumend', Drumroll, lpre, lend, synset)
test('drumpre', Drumroll,lpre, lpre, synset)
test('drumpe', Drumroll,lpe, lpe, synset)

for s in [pre, end, sen, pe]:
    for s2 in [biases, bias10]:
        test(str(s2)+':'+str(s),Edistance,s2,s1)

for s in [pre, sen, end]:
    test(f'sp({s})',SynsetPolarity, s, Offset['homonym'], synset)

EndTests = FirstTest+Test_count

#Now replace the bytes for Connection_count, Synthetic_cout, Test_count in Preamble
Preamble[1] = Component_count
Preamble[2] = Synthetic_count
Preamble[3] = Test_count
                          
# set up list of items in record + synthetic items.  Use python object list
T = [None]*(Component_count+Synthetic_count) #was: np.zeros(Component_count
                                             #  +Synthetic_count, dtype=np.uint8)

Te = np.zeros(Test_count, dtype=np.float32)

def main():
    print(f'Setup time: {time.time()-start_time}')
    total_records_processed = 0
    with open(train_input_file) as fi:
        train = json.load(fi)
        with open(train_output_file,'wb') as fo:
            fo.write(bytes(Preamble))
            print (f'Preamble is {len(Preamble)} bytes long')
            for key,example in train.items():

                #print(key, time.time())
                average, stdev, test_slice = evalu(example,key)
                total_records_processed += 1
                
                #print('evalu returned.')

                #write binary file 
                fo.write(np.int32(int(key))) #4 bytes for key
                #fo.write(np.float32(average))  #4 bytes for average and stdev`
                choices = example['choices']
                fo.write(np.int32(compressed_choices(choices))) # for version 4
                fo.write(np.float32(stdev))  

                #print('finished average,stdev.  Starting test output')

                fo.write(test_slice)         #Test_count * 4 bytes for all tests
                #print('did test output')
            print(f'Total records processed: {total_records_processed}'
                  f'\nTotal runtime: {time.time()-start_time}')
            print(f'debugging info: Small_pivot {Small_pivot} Total_pivots {Total_pivots} Synset_None {Synset_None} {BadSynsetList}')

def evalu(example,key):
    # read in components from file.
    for i,name in enumerate(Components):
        if name == 'example':
            T[i] = key
        else:
            T[i] = example[name]
    # synthesize derived components
    for j,s in enumerate(Syntheses):
        i = j+Component_count
        T[i] = s()
    #perform tests
    for i,s in enumerate(Tests):
        Te[i] = s()

    # return slice of T containing tests.
    return T[Offset['average']], T[stdev], Te


def compressed_choices(choise):
    """
    Take a list of 5 ints, each in the range(1,6)
    and return an int 
    """
    choices = list(choise)  # make a copy, rather than changing input data
    for i,c in enumerate(choices):
        if c<1:
            choices[i] = 1
        if c>5:
            choices[i] = 5
    choices.sort()
    lc = len(choices)
    if lc > 5:
        choices = choices[-5:]
    elif lc < 5 and lc > 0:
        choices = ([choices[0]]*(5-lc)) + choices
    if len(choices) != 5:
        complain()
    retval = 0
    for i,c in enumerate(choices):
        retval |= int(c)<<3*i
    return retval

def choice_list_standardize(ls):
    choices = list(ls) # copy the list, instead of modifying original
    for i,c in enumerate(choices):
        if c<1:
            choices[i] = 1
        if c>5:
            choices[i] = 5
    choices.sort()
    lc = len(choices)
    if lc > 5:
        choices = choices[-5:]
    elif lc < 5 and lc > 0:
        choices = ([choices[0]]*(5-lc)) + choices
    if len(choices) != 5:
        complain()
    return choices

def uncompress_choices(num):
    """
    accept an int as produced by compress_choices, and turn it into a list
    """
    retval = []
    for i in range(5):
        retval.append(num&7)
        num = num >> 3
    return retval


def wset(st):
    """
    accept a string, and turn it into a set of words
    """
    ast = st.strip().split()
    wds = set()
    for s in ast:
        s = s.strip('.,!?:;-')
        s = s.lower()
        wds.add(s)
    return wds

def lemma_set(se):
    """
    se is a set of words.
    return set of all the lemmas of all the words
    """
    retval = set()
    for w in se:
        retval.update(lemmat(w))   # was union.  also see lemmas |= below...
        
    if len(retval) == 0:
        pass
    return retval

def lemmat(s):
    """
       return a set of the lemmas of form s
    """
    lemmas = set()
    for pos in 'vnasr':
        lemmas |= set(wn._morphy(s,pos,True))
    return lemmas

with open('../hdefs') as fi:
    HomonymDict = dict()

    homInProcess = None
    partialEntry = None

    for lin in fi:
        ho,sy,df = lin.strip().split('\t')
        if homInProcess is None:
            homInProcess = ho
            partialEntry = [(sy,df)]
        elif homInProcess == ho:
            partialEntry.append((sy,df))
        else:
            HomonymDict[homInProcess] = partialEntry
            homInProcess = ho
            partialEntry = [(sy,df)]

    if homInProcess is not None:
        HomonymDict[homInProcess] = partialEntry
                            

def get_wordnet_synset(homonym,definition):
    """
    The 'judged_meaning' field is a (close approximation of) synset definition
    The HomonymDict provides the synset for homonyms and meanings in
    the train, dev, and test files.

    The previous code, most of which is below, corrected for several
    differences between the data and wordnet.
    """
    syn_def_list = HomonymDict[homonym]
    for sy,df in syn_def_list:
        if df == definition:
            return sy
    print(f'could not get synset of "{homonym}"')
    return None
        
"""
This code moved to count_homynyms.py:
Bogus = [ 
         ('appendicitis','appendix'),
         ('foundational', 'foundation'),
         ('reaction', 'react'),
         ('Mass', 'Massachusetts'),
         ('delivery', 'deliverer'),
         ('score', 'scores'),
         ('stand', 'testify'),
         ('stand', 'stands'),
         ('step', 'steps')
         ]

"""

"""
  # this also 
def get_wordnet_synset(homonym, definition):
    ss = wn.synsets(homonym)
    for s in ss:
        if s.definition() == definition:
            return s._name
        else:
        # edit out left single quotes?
            d = s.definition()
            p = d.find('`')
            while p >=0:
                d = d[:p] + "'" + d[p+1:]
                if d == definition:
                    return s._name
                p = d.find('`')

            # try removing final punctuation from definition
            d = s.definition()
            e = definition # that is, the parameter to this function
            if (e[-1] == ';' or e[-1] == '.' or e[-1] == '!') and  d == e[:-1]:
                return s._name

        # try substitution table
    for w,sw in Bogus:
        if w == homonym:
            for s in wn.synsets(sw):
                if s.definition() == definition:
                    return s._name
    # could try more, but instead print error and continue
    print(f'could not get synset of "{homonym}"')
    return None
"""


def score(x,y):
    """
    """
    lx = len(x)
    ly = len(y)
    if lx == 0 or ly == 0: return 0
    inter = x .intersection ( y)
    li = len(inter)
    return li/(lx+ly-li)

def bias_words(synset, howmany=-1, keep_underscores=True):
    """
    Fetch the bias_words for the indicated synset.  
        howmany indicates how many to fetch:
           -1 means all
    """
    if synset is None: 
        return set() # empty set
    s0 = bs.get_bias_words(synset) # they arrive as a list
    #if keep_underscores: # remove words with underscores (are none in data)
    #    if howmany < 0:
    #        return set(s0)
    #    return set(s0[:es])

    # remove words like 'electrical_potential', which wordnet has many of
    # a different strategy would be to return both words -- I'll think about it
    retval = set()
    if howmany < 0:
        howmany = 50 #this is the number of bias words for each synset
    count = 0
    for w in s0:
        if (not keep_underscores) and w.find('_') >= 0:
            continue
        retval.add(w)
        count += 1
        if count >= howmany:
            return retval
    return retval  # removed a few, but return as many as we found


#/usr/bin/env python3
"""
a series of subrountines to support context compatibility checks based on 
bias_word sets.   Since there are some related functions in train1.py, 
perhaps those should be moved downhill to here, or perhaps these and those
will be part of train2.py.
"""
from nltk.corpus import wordnet as wn

syn2ball = dict()
# lem2syn = dict() # this datastructure not very valuable: wn.synset(lem) exists

Small_pivot = 0
Total_pivots = 0

def find_pivots(context, homonym, homonym_synset):
    """
    pivot words are words which share a synset with the homonym, 
      there is code in train1 to find the synset for the homonym by comparing
      the 'judged_meaning' to the .definition() for each of the synsets of
      the homonym.  It doesn't fail on the train.json file ...
    The parameter *could* be that definition, which would simplify actions
    using this routine.

    In the current scheme, we guarantee an interesting list of synsets
    by using the synsets of the homonym.  Hopefully the rank of the
    the synsets appropriateness to the context will be interesting.
    This a non-parametric aproach, because the context fit of 
    the judged_meaning varies widely.
    """

    synsets_defs = HomonymDict[homonym]


    #retval = {wn.synset(homonym_synset)}  # make sure this set is non-empty
    retval = {homonym_synset}  # make sure this set is non-empty
    for s,d in synsets_defs:
        retval.add(s)
    """
    for lemma in context:
        ss = wn.synsets(lemma)
        for sy in  wn.synsets(lemma):
            if sy.name() == homonym_synset:  
                retval.update(set(ss))
    """

    global Small_pivot, Total_pivots
    lr = len(retval)
    if lr == 1:
        Small_pivot += 1
    if lr > 1:
        Total_pivots += lr
        pass

    return retval


def fits_word(word, psynset):
    """
    return the largest size of the largest intersection of 
    any of the synsets of the word, and the pivot synset.
    """
    retval = None

    pset = syn2ball.get(psynset,None)
    if pset is None:
        pset = set(bs.get_bias_words(psynset)) #was psynset.name()))
        syn2ball[psynset] = pset
    lpset = len(pset)
    for syn in wn.synsets(word):
        synn = syn._name
        ball = syn2ball.get(synn,None)
        if ball is None:
            ball = set(bs.get_bias_words(synn))
            syn2ball[synn] = ball
        lball = len(ball)
        incommon = pset .intersection (ball)
        linc = len(incommon)/(lpset+lball)
        
        if retval is None or linc > retval: 
            retval = linc

    return retval

def fits_context(context, psynset):
    """
    produce a figure-of-merit for the synset in the context.  
    this is done by comparing the ball of bias-words for the synset to
    each of the similar balls of each synset of each word of the context, 
    """

    retval = 0
    word_count = 0
    for word in context:
        word_count += 1
        retval += fits_word(word, psynset)
    
    return retval/word_count   #normalize fit to number of words



def ranked_senses(context, pivots):
    """
    extract the list of all synsets from the set of pivots
    build list of (fits,synset) pairs
    sort list
    return list of triples: (rank, fits, synset)
    """
    pairs = []
    for psyn in pivots:
        pairs.append((fits_context(context, psyn), psyn))
    pairs.sort(key=lambda x:x[0])

    # compute rank, where items with common fits share the same average rank
    triples = []
    i = 0
    lpairs = len(pairs)
    while i<lpairs:
        n = i+1
        while n < lpairs and pairs[n] == pairs[i]:
            n += 1
        sr_in = 0
        for j in range(i,n):
            sr_in += j+1
        for j in range(i,n):
            triples.append((sr_in/(n-i),pairs[i][0],pairs[i][1]))
        i = n
    return triples

Cached_triple = None
Synset_None = 0

# find rank in triples of homonym_synset
def rank5(triples,homonym_synset):
    """
    find the rank in the list of sorted triples of the homonym_synset, and
    normalize it to the interval 1-5
    It turns out that the list is 1 item long 2/3 of the time.
    I haven't figured out a consistent way to be interested for that 1/3.
    So the rank isn't so interesting.
    
    The average rank scheme guarantees that a list of n triples will sum
    to n(n+1)/2.  on a 1-5 interval that would be 15 = 5*6/2,
    so multiplying each triple rank by 30/n/(n+1) gives us a 5-normalized rank,
    such that the sum of all the ranks sums to 15.
    """
    global Cached_triple 


    n = len(triples)
    if n == 0: return 0

    for i,(r,f,s) in enumerate(triples):
        if s == homonym_synset:
            Cache_triple = (r,f,s)
            Cache_rank = 5*(r/n)  # was: 30*r/n/(n+1) want rank to be 5-based.
            return np.float32(Cache_rank)

            return f                  #since n is usually 1, rank isn't useful
    print('rank5 failed')             #BUT the DrumCache action can retrieve it
    return 1

BadSynsetList = []
Drumroll_calls = 0
def drumroll(context,testset,homonym_synset):
    """
    combine the previous batch of routines to estimate human score

    here I presume to find the pivots based only on the precontext *not testset*
    """
    global Drumroll_calls, syn2ball, Synset_None

    homonym = T[Offset['homonym']]  # can't pass this to drumroll, too many args

    if homonym_synset is None:
        Synset_None += 1
        BadSynsetList.append((T[0],homonym))
        return 0

    Drumroll_calls += 1
    
    if len(testset) == 0:
        return 0

    if Drumroll_calls % 180 == 0:
        syn2ball = dict()   # don't let this dict get too big.

    return rank5(ranked_senses(testset, 
                               find_pivots(context,homonym, homonym_synset)),
                 homonym_synset) 


def EuclideanDistance(vset, wset):
    """
    using the Emodel english model, build an average distance in the embedding
    space between the elements of vset and wset
    intended use is for vset to be the bias-word set of the homonym synset,
    and wset a set of words for precontext, sentence, or ending
    bias words are always lemmas, but probably not necessary to lemmatize.
    We have a choice of 10, 20, 50 bias words, wset has a similar range.
    """
    lv = len(vset)
    lw = len(wset)
    if lv<=lw:
        smallset = vset
        bigset = wset
    else:
        smallset = wset
        bigset = vset
    
    dim = Emodel.vectors.shape[1]
    Vs = np.ndarray((len(bigset),dim), dtype=np.float32)
    i = 0
    for v in bigset:
        try:
            Vs[i] = Emodel[v]
        except:
            continue
        i += 1
    Vs = Vs[:i,:] # in case there were some missing words, ignore them
 
    total = items = 0
    for w in smallset:
        try:
            wvec = Emodel[w]
        except:
            continue
        ds = Vs @ wvec # Emodel.cosine_similarities(wvec, Vs)
        if np.isnan(ds.dot(ds)):
            print('?nan')
            pass

        items += ds.shape[0]

        # here is where code to convert cosin similary to euclidean distance
        # would sit.  The formula, which works well for normalized spaces 
        # and acute angles, is
        #     Euclidean distance = sqrt ( 2 - 2 * cosine-distance)
        # but I am momentarily confused about the obtuse angles, which
        # have negative cosines.  These are quite rare in word embeddings,
        # I believe.
        # Those negative cosines would be an excellent reason to use
        # Euclidean distance ... but the nice thing about a mix of 
        # positive cosines is that they emphasize closeness better than
        # Euclidean distance.  Still, pending demonstration of the rarity of
        # negative cosines, I provide the code for the conversion:
        #bds = (ds>1)
        #if any(bds): 
        ds[ds>1] = 1.0 # identical vectors generate many 1.0000001 1.0000002
                        # dot products, due to hardware rounding last bits.
        #eds = np.sqrt(-2*ds + 2) # this is the correct computation
        eds = 1 -ds # this omits the sqrt, which pulls values toward 1, 
                    # and gives the same range, since neg cos adds as much as 1
        total += np.sum(eds)
    if items == 0:
        return 0

    if T[0] == 78:
        pass
    retval = total/items  # the range of Euclidean distance is [0:2]
    return retval  # the range of Euclidean distance is [0:2]

def mvectors(wset, model):
    """
    wset a set of words
    model a word-embedding
    returns an array of word-embedding-vectors
    """
    ls = list()
    for w in wset:
        try:
            ls.append(model[w])
        except:
            continue
    return np.array(ls, dtype= np.float32)

Discardable = 0.91 # a value of cosine distance too distant to care about.
            
def CosineDistance(vset, wset):
    """
    compute the 'normalized' cosine distance between two sets.
    'normalized' means that the distance should be betweem 0 and 1.
    just as the cosine distance is, but it might be:
    an average, a minimum, an average of distances less than some parameter...
    I'm going to code it first, and try them all.
    """
    #maybe this should be special-cased in calling routine
    if len(vset) == 0:
        return 1 # max cosine distance... 
    V = mvectors(vset,Emodel)
    W = mvectors(wset,Emodel) # these are all normalized vectors
    lots = V @ (W.T)  #compare all words in each set --e.g. compute cosines
    #return 1-lots.mean() # simplest idea (except I left the 1 off earlier)

    # next code never reached, but sitting here waiting for me to return to it
    f = lots > 1.0 
    lots[f] = 1.0    # those float32 vectors have more rounding errors...
    mlots = np.mean(lots, axis = 1)  # get maxima along one axis; 1 is max cos
                #mlots is the best score among for each v, among all w
    nlots = 1- mlots # compute cosine distance
    f = nlots < Discardable  # prepare to discard some distant words
    denom = np.sum(f)
    if denom == 0:
        return 1
    else:
        retval = np.sum( nlots[f]) / denom
    return retval

Dist_list_stash = None

Syn3ball = dict() # use separate dict; syn2ball sets are all len 50
def synset_polarity(testSet, homonym, homonym_synset):
#def synset_polarity(testSet, homonym_synset):
    """
    compare the Euclidean distances between the testSet and the synset 
    word-balls of the various synsets of the homonym.  
      if the homonym-synset is the closest, return +1
      if some other synset is closer, return -1
      if there is a tie, return 0.   Might want to set a minimum distance,
      such that if no synset is closer than the mininum distance, we return
      zero.

    """
    global Syn3ball, Dist_list_stash
    
    MinDist = 1e-5 # chosen rather small, to cover rounding error.
                   # might increase it, if doing so helps.
    
    #get the list of synsets
    synsets = find_pivots(None, homonym, homonym_synset) # context arg not used,
    dist_list = []

    for syn in synsets: # 
        ball = Syn3ball.get(syn,None)
        if ball is None:
            if len(Syn3ball) > 200:
                Syn3ball = dict()   # don't let this dict get too big
            siz = 0
            blist = bs.get_bias_words(syn)
            ball = set()
            ind = 0
            while siz < 10 and ind < len(blist):  # copy only this many words
                while ind<len(blist) and blist[ind] not in Emodel:
                    ind += 1
                if ind<len(blist):
                    ball.add(blist[ind])
                    ind += 1
                    siz += 1
            Syn3ball[syn] = ball
        dist_list.append((
            #EuclideanDistance(testSet, ball)
            CosineDistance(testSet, ball)
            , syn))
    Dist_list_stash = dist_list  #used by a caller of this routine for stats
    if len(dist_list) == 1:
        item = dist_list[0]
        if item[0] > MinDist:
            if item[1] == homonym_synset:
                return +1
            else: return -1
        else:
            return 0
    dist_list.sort(key = lambda x: x[0])
    item = dist_list[0]
    item2 = dist_list[1]
    if item2[0] - item[0] < MinDist:
        if item[1] == homonym_synset or item2[1] == homonym_synset:
            return 0
    if item[1] == homonym_synset:
        return 1
    else:
        return -1

                         
            







        
if __name__ == '__main__': main()


