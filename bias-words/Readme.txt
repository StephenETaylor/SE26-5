This directory is some work with re-implementing the code described in 
  De-Conflated Semantic Representations,
  by Pilehvar and Collier  EMNLP  2016

  which is a scheme for retrofitting wordnet synset vectors into an existing
  word model.

Files in this directory
    alg1.py
	an implementation of algorithm 1 from the paper.
	goal is to find a list of related 'biasing words' for each synset,
	ranked according to relatedness.
	Takes about an hour to run

	Does not produce exactly the same output as the paper; they
	clearly used an 'extended Wordnet' which contained at least
	one extra relation based on word definitions.  So table 1 shows
	'system_of_numeration' and 'element' as biasing words for digit.n.01,
	both of which occur in the definition of the synset.

    alg2.py
	an implementation of equation (2) in the paper, which uses the
	biasing words to compute a weighted average vector for the
	synset.
	Takes about 40 seconds to run; currently skips 40% of synsets.

    biases.txt
	a rather large file.  I stored 50 bias words per synset,
	for 109096 synsets.

    DefRel.py
	An attempt to recreate the extra definition relation in wordnet.
	My attempt was way too wordy, and the resulting synsets seem to overlap.

    nuff.bin
	this is the output file of alg2.py

    __pycache__
	the directory which hold compiled files used for imports.  
	contains:
	    DefRel.cpython-312.pyc
	    DefRel.cpython-38.pyc
	    relations.cpython-312.pyc
	    relations.cpython-38.pyc

    relations.py
	this file reads through the wordnet relations and builds the M
	file used in the PPR (Personalized Page Rank) algorithm mentioned
	in the paper.

    sm.bin
	contains a cached version of the data computed by relations.py

    test.py
	a program for executing the ../senses/testing/test_scws.py program
	in the context of the files in this directory.
	See testSCWS, below.

    testSCWS.py
	a hard link to ../senses/testing/test_scws.py used by test.py
	since the filenames are module globals in that file,
	test.py imports testSCWS and sets the filenames for the 
	program to run correctly.
	test_scws.py requires a recent version of scipy Spearman correlation;
	I use python 3.12. 3.8 is too un-recent.



