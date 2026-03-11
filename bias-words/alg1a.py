"""
This is algorithm 1 from the third page (p 1682) of Pilehar and Collier 2016,
De-conflating semantic representations, EMNLP 

The purpose of this code, is to provide a list of 'sense biasing words' for 
Wordnet word senses.

on 19.2.2026 I began to modify this file to make use of the WordNet 3.0
"gloss relation" as provided by IX_stuff/wnet30g_rels.txt
"""
from nltk.corpus import wordnet as wn
import numpy as np
import numpy.linalg as nl
import relationsA as relations
import scipy.sparse
import sys
import time

#Some debugging flags
PRINT_BIAS_WORDS = False


M = relations.loadM()  # a matrix describing synset connnections
SNumbers = relations.loadS() # SNumbers[ss] -> number used for synset ss in M
Snames =  relations.loadN() # Synsets[Snames[n]] = n

max_PPR_iterations = 10  # now these two are parameters, modified in main()
Number_of_bias_words = 50

def alg1(ss):
    """
    ss is a wordnet synset name

    M is a sparse matrix describing the semantic relationship graph
    between wordnet synsets.  Each row of M corrsponds to the the synset
    numbered i, and each element M[i,j] is 1/degree(i) if there is a
    wordnet relation from i to j, and zero otherwise.
    The wordnet relations I used can be seen in relations.py



    """
    t = SNumbers[ss]
    Ss = wn.synset(ss)

    biases = Ss.lemma_names()  # lines 1,2,3
    biaset = set(biases)
    m = len(SNumbers)  # number of synsets in graph M
    p = np.ndarray(m, dtype= np.float32) # probably 16 bits would be plenty
    # I think that this loop is a fiction; better computed by finding P once,
    # and the explanation for P&H algorithm does so.
    #for i in range(m):      # line 4
    #    if i == t: continue
    #    p[i] = PPR(i,t) # we don't pass M, its global #line 5 # p is m-long vector
    P = PPR(t) # lines 4,5. M is global, so we don't pass G. we do all i at once
    I = np.argsort(P) # this list of indices includes t, which we should ignore
    #I'm not yet sure how many bias words we ought to look for, but I think
    # that 200 is more than enough
    # P&H says they settled on 25.
    while len(biases) < Number_of_bias_words:
        for h in I[::-1]: # I[0] is lowest rank. so start h from highest...
            if h == t: continue
            if len(biases)>=Number_of_bias_words:
                break
            yh = Snames[h]
            SS = wn.synset(yh)
            wl = SS.lemma_names()
            for w in wl:
                if w in biaset: continue
                if len(biases)>=Number_of_bias_words: break
                biases.append(w)
                biaset.add(w)
    
    return biases
    
    
def main():
    # check command line for iteration count:
    global max_PPR_iterations, Number_of_bias_words 

    if len(sys.argv)> 1: max_PPR_iterations = int(sys.argv[1])
    if len(sys.argv)> 2:
        syns = sys.argv[2] # Filename of synsets to process
        with open(syns,'r') as fi:
            ToProcess = []
            for lin in fi:
                ToProcess.append(lin.strip())
    else: 
        ToProcess = Snames
    if len(sys.argv) > 3:
        Number_of_bias_words = int(sys.argv[3])


    start_time = time.time()
    test_interval = 1000

    B = dict()
    # save biases as single python string:

    for n,ss in enumerate(ToProcess): 
    #for n,ss in enumerate('digit.n.01,digit.n.03'.split(',')): 
        biases = alg1(ss) # this is the chart, table, tabular_array sense
        B[ss] = ' '.join(biases)
        if n > test_interval: 
            print(n, ss, time.time()-start_time)
            test_interval += 1000


    print (max_iterations, 'max iterations in PPR')
    # now write these out to bias file:
    with open('biases.txt', 'w') as fo:
        for ss,biases in  B.items():
            print(f'{ss}: {biases}', file=fo)
            if PRINT_BIAS_WORDS:
                print(f'{ss}: {biases}')

total_iterations = max_iterations = 0
def PPR(t):  #The argument 'i' from P&H removed; we do all i at once.
    """
    This implements the Personalized PageRank algorithm, as described in
    Pilehar and Collier 2016.

    it uses M, the m by m description of the connections from synset i to 
    synset j, in which, if Sname[i] has d outgoing relations, then
    if Sname[j] is one of those connections M[i,j] == 1/d; otherwise 0

    The description in P&C suggests m**2 calls to find the biases for
    all i,j.

    Here I plan to do only O(m) calls, but will do calculations for
    all p[i] in a single call.  
    I think that the point of the personalized pagerank algorithm is 
    that by not intermingling other starting points, we get only the
    page rank wrt i.

    The function builds the vector P by 
    starting with P0 = indicator(t), that is vector A : A[x] = 1 if x==t else 0
    then iterates with:
        (sigma = 0.85)
        P_{n+1} = (1-sigma)*P[0] + sigma * M @ P_n
    until P_{n+1} == P_n

    """
    global max_iterations, total_iterations

    m = M.shape[0]
    #P = np.zeros((3,m), dtype=np.float32)
    P = np.zeros((6,m), dtype=np.float32)
    P[0,t] = 1
    latest = 0
    coming = 1
    P[1] = 0.15*P[0] + 0.85 * (P[0] @ M)

    # after I am satisfied with this perhaps I can code it inline...
    #def not_same(): # return false if P[coming] same as P[latest]
    #    if any(P[coming] != P[latest]):
    #        return True
    #    else: return False

    iterations = 0
    while iterations < max_PPR_iterations: #not_same(): // use fixed # instead of steady state
        latest = coming
        #coming = latest ^ 3  # coming,latest alternate between 1,2 and 2,1
        coming = 1+((latest+1)%5)  # keep a couple of vectors so I can debug
        P[coming] = 0.15*P[0] + 0.85 * (P[latest] @ M)
        iterations += 1

    total_iterations += iterations
    if  iterations > max_iterations:
        max_iterations = iterations

    return P[latest,:]


if __name__ == '__main__': main()


