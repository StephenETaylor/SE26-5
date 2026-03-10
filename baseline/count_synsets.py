import json
from nltk.corpus import wordnet as wn

homonyms = set()
defs = dict()
d_per_h = dict()  #synsets / homonym
top = 0  # max number of synsets for any homonym

#for file in ['train' ]:
#for file in ['dev']:
#for file in [ 'test']:
def main():
    global top 
    for file in ['train','dev', 'test']:
        with open ('../'+file+'.json') as fi:
            data = json.load(fi)

            # now loop through the data
            for key,example in data.items():
                h = example['homonym']
                d = example['judged_meaning']
                homonyms.add(h)
                ds = defs.get(h,None)
                if ds is None:
                    defs[h] = ds = set()
                ds.add(d)
                ds2 = d_per_h.get(h,None)
                if ds2 is None:
                    d_per_h[h] = ds2 = set()
                ds2.add(d)
                if len(ds2) > top:
                    top = len(ds2)

    maxd = None
    for h in homonyms:
        scount = len(wn.synsets(h))
        if maxd is None or scount > maxd:
            maxd = scount
            maxh = h
    print('maximum synsets for any homonym:',maxh,':', maxd)
    # now count defs/homynym
    print ('total homonyms:',  len(homonyms))

    print('any unusual ones: ')
    un_k = tot_un_d = 0
    for k,v in defs.items():
        if len(v) != 2:
            un_k += 1
            print(' ',k,len(v),';  ',end = '')
            tot_un_d += len(v)
    print()
    print(f'unusual homynyms: {un_k}, mean_defs: {tot_un_d/un_k}')

    histo = [0]*top
    for h,se in d_per_h.items():
        histo[len(se)-1] += 1
    print('histogram of synsets per homonym')
    for i in range(top):
        print('%2d'%(i+1),'%3d'%histo[i],'*'*histo[i])

    #now save defs to file
    weirdos = 0
    with open('../hdefs','w') as fo:
        
        missingSynsets = 0
        for k in sorted(defs.keys()):
            v = defs[k]
            """
            syns = wn.synsets(k)
            sdefs = []
            for s in syns:
                sdefs.append(s.definition())
            for d in v:
                printed = False
                for i,ds in enumerate(sdefs):
                    while ds[-1] == ' ':
                        ds = ds[:-1]
                    if ds[-1] == ';':
                        ds = ds[:-1]
            """
            for d in v:
                s = get_wordnet_synset(k,d)
                if s is not None:
                        print(f'{k}\t{s}\t{d}',file=fo)

                else:
                        print(f'{k}\tNONE\t{d}',file=fo)
                        print(f'missing a synset of {k}: {d}')
                        missingSynsets += 1

    print(f'missing synsets: {missingSynsets}')



# this code lifted from train4 on 22 Jan; that code will be replaced
# by placing the output from this program in a dict.
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
    print(f'could not get synset of "{homonym}": {definition}')
    return None

if __name__ == '__main__': main()
