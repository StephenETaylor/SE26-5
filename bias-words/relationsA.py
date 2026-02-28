"""
statistics from one run:   (wordnet figures probably static)
 time python relations.py
there are  117659 synsets
w/ relations: 109096 totalDegree: 267357

real	0m17,258s
user	0m16,689s
sys	0m0,547s

# this code, now commented, used in creating initial file
synset_relations =    'hypernyms,instance_hypernyms,hyponyms,instance_hyponyms,member_holonyms,substance_holonyms,part_holonyms,member_meronyms,substance_meronyms,part_meronyms,attributes,entailments,causes,also_sees,verb_groups,similar_tos'.split(',')
for r in synset_relations:
#    print (f'    retval.update({"{"}x for l in ss.{r}() for x in l{"}"})')
"""

import DefRel as dr
from nltk.corpus import wordnet as wn
import numpy as np
import pickle
import scipy.sparse as scsp
import sys

# some debugging flags; although they could go into a UI, it makes better
# sense to edit them by hand, since once the questions they are designed to
# answer are resolved, there will be no further need for them.
USE_IXA_RELATIONS = True # gloss relation from University of Basque Country
PRUNE_DANGLES = True
if USE_IXA_RELATIONS:
    PRUNE_DANGLES = False # these Dangly bits will have gloss connections
PRUNE_ORPHANS = False
USE_LEMMA_RELATIONS = False
USE_DEFREL = False

SNumbers = Snames = M = None
def main0():
    global SNumbers , Snames , M 
    try:
        # this will of course fail, if file doesn't exist
        with open('smA.bin','rb') as fib:
            (SNumbers,Snames) = pickle.load(fib)
            M = scsp.load_npz(fib)
            pass
    except:
        main1()

def loadM():
    if M is None:
        main0()
    return M

def loadS():
    if SNumbers is None:
        main0()
    return SNumbers

def loadN():
    if Snames is None:
        main0()
    return Snames

def main():

    global Snumbers, Snames, M

    if len(sys.argv)>1 and sys.argv[1].lower == '--recompute':
        main1()

        #check if the work already done
    elif SNumbers is not None and len(SNumbers) == 117659:
        print('We have already computed Snumbers, and probably all else!')
        print('If you really want to recompute,  run with --recompute')
    else:
        main1()

def main1():
    global SNumbers, Snames, M
    if USE_IXA_RELATIONS:
        """
        import pickle
        with open('IXA_stuff/3dicts.bin','rb') as fib:
            xx = pickle.load(fib)
            wn2ix = xx['wn2ix']
            ix2wn = xx['ix2wn']
            """
        def ix2wn(ix):
            # ix is a string, 
            # return a Synset object
            offset = ix[:-2]
            pos = ix[-1] # one of 'a', 'n', 'r', 'v'
            #retval = wn._synset_offset_cache[pos][int(offset)]
            retval = wn.synset_from_pos_and_offset(pos,int(offset))
            return retval

        # replaces wn2ix:
        def wn2ixS(Syn):
            # Syn a synset object
            # return a string
            offset = Syn._offset
            pos = Syn._pos
            if pos == 's':
                pos = 'a'
            soffset = '%08d' % offset
            retval = soffset + ':' + pos
            return retval
        
        

# build list of synsets:
    SNumbers = dict()
    Snames = []
    nextnum = 0
    if USE_IXA_RELATIONS:
        IxNumbers = dict()
        with open('IXA_stuff/wnet30g_rels.txt') as fi:
            vset = set()
            for lin in fi:
                ixu = lin[2:12]
                ixv = lin[15:25]
                IxNumbers[ixu] = nextnum
                if ixv not in IxNumbers:
                    vset.add(ixv)
                ss = ix2wn(ixu) #[ixu]
                ssn = ss.name()
                if ssn in SNumbers:
                    continue
                SNumbers[ssn] = nextnum
                nextnum += 1
                Snames.append(ssn)

            # check to see if there are any synsets we missed in gloss relation
            # yes, about 1200 synsets occurred in v position, but not u:
            # perhaps this should be a bunch of new synsets, which happen
            # not to be in WN3.0, or maybe just not in the nltk version.

            # (a slightly more clever version of this code would keep vset as
            #  a priority queue and take out the least elements as the
            #  u values passed them, in order to have completely ordered
            #  insertions down below, except that given that these synsets
            #  didn't point to gloss words, we could continue from that point
            #  after all gloss entries handled...)

            gloss_entries = nextnum # This is the number of synsets with glosses

            for ixv in vset:  
                if ixv in IxNumbers:
                    continue
                ss = ix2wn(ixv)
                ssn = ss.name()
                if ssn in SNumbers:
                    continue
                SNumbers[ssn] = nextnum
                nextnum+= 1
                Snames.append(ssn)
            IxNumbers = None   # free up storage
            vset = None

            # but given this experience, and the fact that 'matter.n.03' 
            # still not in SNumbers, let's do the next dance...
            gloss_rhs_only = nextnum    # I don't actually reference this..

            for w in wn.words():
                for ss in wn.synsets(w):
                    ssn = ss.name()
                    if ssn in SNumbers: continue
                    SNumbers[ssn] = nextnum
                    nextnum += 1
                    Snames.append(ssn)

    else:
      for w in wn.words():
        for ss in wn.synsets(w):
            ssn = ss.name()
            if ssn in SNumbers: continue
            if PRUNE_DANGLES and len(get_rel(ss)) == 0:
                # I called these orphans, but they are old maids; no successors
                continue
            SNumbers[ssn] = nextnum
            nextnum += 1
            Snames.append(ssn)

    ssize = len(SNumbers)
    print ('there are ' , ssize, 'synsets')
    #lil_array can grow more efficiently than a CSR_array
    sparseM = scsp.lil_array((ssize,ssize),dtype=np.float32)
    ssZero = ssNotZero = totalDegree = 0
    argSsZero = None
    maxDegree = 0

    def tidbit(s, i,rel):
        # store the rel items in the sparseM atrix in row i

        nonlocal ssZero, ssNotZero, maxDegree, totalDegree # and sparseM of course...
        nonlocal argSsZero
        degree = len(rel)
        if degree == 0:
            ssZero += 1
            argSsZero = s
            return
        ssNotZero += 1
        totalDegree += degree
        if degree > maxDegree: maxDegree = degree
        for s2 in rel:
            j = SNumbers[s2] #.name()]
            sparseM[i,j] = 1/degree  #split up prob mass entering s

    # add gloss relation first
    if USE_IXA_RELATIONS:
        with open('IXA_stuff/wnet30g_rels.txt') as fi:
            prev_u = None
            for lin in fi:
                a,b,_,_ = lin.strip().split(' ')
                _,u = a.split(':')
                _,v = b.split(':')
                if prev_u is None:
                    prev_u = u
                    vlist = {ix2wn(v)._name} # note that vlist is actually a set..
                    continue
                elif prev_u == u:
                    vlist.add(ix2wn(v)._name)
                    continue
                elif prev_u != u:
                    # add in code below.
                    ss = ix2wn(prev_u)
                    ssn = ss.name()
                    prev_u = u
                    i = SNumbers[ssn]
                    rel = get_rel(ss)
                    rel .update(vlist)
                    vlist = {ix2wn(v)._name}
                    
                    # this tidbit was there before...
                    tidbit(ssn, i, rel)

        # now insert all those synsets which didnt seem to have a gloss
        # or maybe their glosses were all stopwords
        for i in range(gloss_entries,len(SNumbers)):
            ssn = Snames[i]
            rel = get_rel(wn.synset(ssn))
            tidbit(ssn, i,rel)

    else:  # not using IX gloss relation
      for s,i in SNumbers.items():
        rel = get_rel(wn.synset(s))
        degree = len(rel)
        if degree == 0: 
            ssZero += 1
            argSsZero = s
            continue
        else:
            # following code removes references to Old Maids, but may create
            # new infertile nodes
            toRemove = set()
            for s2 in rel:
                if s2 not in SNumbers:
                    toRemove.add(s2)
            if len(toRemove) == 0:
                pass
            elif len(toRemove) == len(rel):
                ssZero += 1
                argSsZero = s
                continue
            else:  # there will be something left in the rel set, so 
                rel.difference_update(toRemove)
                   # fall through to add what is left to sparseM

            ssNotZero += 1
            totalDegree += degree
            if degree > maxDegree: maxDegree = degree
            for s2 in rel:
                j = SNumbers[s2]
                sparseM[i,j] = 1/degree
    
    # arrive here after one of those two loops initialized sparseM
    # csr_array said to be more efficient at matrix multiplication.
    M = scsp.csr_array(sparseM) # this is the last reference to sparseM

    true_orphans = 0
    ones = np.ones((1,M.shape[0]), dtype = np.float32)
    colsum = ones @ M
    for i in range(M.shape[1]): # for each column
        if colsum[0,i] == 0:
            true_orphans += 1
            argcolzero = Snames[i]


    if true_orphans > 0:
        print('orphaned synsets:', true_orphans, 'including:',argcolzero)
    if ssZero != 0:
        print ('infertile synsets', ssZero, 'including:', argSsZero)
    if lemma_added != 0:
        print('connections added by lemmas:', lemma_added)
    print('w/ relations:', ssNotZero, 
          'totalDegree:',totalDegree, 
          'maxDegree:', maxDegree)

    with open('smA.bin','wb') as fob:
        pickle.dump((SNumbers,Snames),fob)
        scsp.save_npz(fob,M)

lemma_added = 0    # count of new stuff
def get_rel(ss):
    """
    return a set of synsets  for which ss has a relation
    """
    global lemma_added
    retval = set()
    # these are the synset relations
    retval.update({x.name() for x in ss.hypernyms()})
    retval.update({x.name() for x in ss.instance_hypernyms()})
    retval.update({x.name() for x in ss.hyponyms()})
    retval.update({x.name() for x in ss.instance_hyponyms()})
    retval.update({x.name() for x in ss.member_holonyms()})
    retval.update({x.name() for x in ss.substance_holonyms()})
    retval.update({x.name() for x in ss.part_holonyms()})
    retval.update({x.name() for x in ss.member_meronyms()})
    retval.update({x.name() for x in ss.substance_meronyms()})
    retval.update({x.name() for x in ss.part_meronyms()})
    retval.update({x.name() for x in ss.attributes()})
    retval.update({x.name() for x in ss.entailments()})
    retval.update({x.name() for x in ss.causes()})
    retval.update({x.name() for x in ss.also_sees()})
    retval.update({x.name() for x in ss.verb_groups()})
    retval.update({x.name() for x in ss.similar_tos()})

    if USE_DEFREL:
        # I built the following function because it appears that something
        # similar was used in the P&H code
        retval.update(dr.definition_extract(ss))
        # now add to retval synsets related through their lemmas

    if USE_LEMMA_RELATIONS:
        sofar = len(retval)
        lset = set()
        for L in ss.lemmas():
            lset.update(relatives(L))
        retval.update( {Ls.synset().name() for Ls in lset})
        lemma_added += len(retval)-sofar

    return retval

"""
#These methods all return lists of Lemmas:

lemma_relations ='antonyms hypernyms instance_hypernyms hyponyms instance_hyponyms member_holonyms substance_holonyms part_holonyms member_meronyms substance_meronyms part_meronyms topic_domains region_domains usage_domains attributes derivationally_related_forms entailments causes also_sees verb_groups similar_tos pertainyms'.split(' ')

"""
def main7():
    table_synsets = wn.synsets('table')
    for ss in table_synsets:
        sset = get_rel(ss)
        lset = set()
        for Lemma in ss.lemmas():
            lset.update(relatives(Lemma))
        slset = {L.synset().name() for L in lset}
        sdif = sset ^ slset
        if len(sdif) > 0:
            print (ss, sdif)
            


def relatives(lemma):
    retval = set()

####for rel in lemma_relations:
####### retval.update(lemma.rel())

    retval.update(lemma.antonyms())
    retval.update(lemma.hypernyms())
    retval.update(lemma.instance_hypernyms())
    retval.update(lemma.hyponyms())
    retval.update(lemma.instance_hyponyms())
    retval.update(lemma.member_holonyms())
    retval.update(lemma.substance_holonyms())
    retval.update(lemma.part_holonyms())
    retval.update(lemma.member_meronyms())
    retval.update(lemma.substance_meronyms())
    retval.update(lemma.part_meronyms())
    retval.update(lemma.topic_domains())
    retval.update(lemma.region_domains())
    retval.update(lemma.usage_domains())
    retval.update(lemma.attributes())
    retval.update(lemma.derivationally_related_forms())
    retval.update(lemma.entailments())
    retval.update(lemma.causes())
    retval.update(lemma.also_sees())
    retval.update(lemma.verb_groups())
    retval.update(lemma.similar_tos())
    retval.update(lemma.pertainyms())

    return retval

if __name__ == '__main__':
     main()
