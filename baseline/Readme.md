bias_search.py
    this module contains a routine to return 'related words' 
        for a wordnet synset name

count_synsets.py
    this standalone program writes a ../hdefs file which provides homonym, 
        definition, synset lines, that is used now by train4.py
cstats.py

data_polarity.py
    this standalone program estimates the synset polarity of the
        pre-context and ending sentences in ../train.json, and
        writes two files, trainPol.dat and pickled.pol, containing the info.
        I wrote it to use in polarity_scaffold.py

dev3E.bin devCE.bin predictions2.json predictions3E.json
predictions.json t2.bin t3.bin t.bin trainCE.bin
    these are data files used or written by 'fossil' python code

fit1.py fit2A.py fit2B.py fit2C.py fit2.py fit3D.py fit3E.py fit3.py fit.py
train1.py train2.py train3.py train.py
    these are 'fossil' python code

fit4.py
    this file reads a t4.bin file written by train4.py and builds
    classification models, then writes a prediction file.

gnuplot.dat  I never got around to using this file.

jsonconcat.py
    a program to concatenate to json files, used to combine
        ../train.json and ../dev.json into ../train_dev.json

Makefile
    make dev_pred builds a predictions4.json file for the ../dev.json file
    make test_pred builds a ~/Desktop/submission.zip file containing
        predictions based on the current state of the data files and code
        for the ../test.json file, using ../train_dev.json

pickled.pol
    an output file from data_polarity.py, loadable with pickle,
        which provides a dictionary which gives the estimated 
        synset_polarity of phrases from ../train.json, pol[(synset,phrase)]
        intended to be used in improving the train4.synset_polarity code
        toward the estimated version.

pivot_contexts.py

polarity_scaffold.py
        intended to be used in improving the train4.synset_polarity code
        toward the estimated version.

power_sequence.py

predictions4.json   the default output of fit4.py

predictions.jsonl 
    a file created because the codabench.org/competitions/10877
    requires a zip file containing a file by this name.
    (Makefile has a line to create it by copying)

__pycache__
    A directory created by python to hold the compiled versions of modules

score_dev_pred.py
    a program to read a predictions.json file and score it for accuracy.
        it compares the file against the cases in the ../dev.json file,
        so it would be used in estimating how good some change in the code was

stupid.pred
    a test file for score_dev_pred, which should have 100% accuracy

stupid.py
    the python program which wrote stupid.pred

t4.bin
      by convention, the output of train4.py after reading ../train.json

td4.bin
      by convention, the output of train4.py after reading ../train.json

test_examiner.py test_reader.py test_tester.py

train4.py
    train4.py reads a json file, and performs a number of tests for each
    story/example

trainPol.dat
    an output file written by data_polarity.py.
    inspecting it shows that the human raters we are trying to rate were
    not completely sensible in considering how a phrase and its synset 
    are related.  A few phrases which seem strongly related to the synset
    are rated 0 or -1, and glancing at the 'float_pol' or non-integer polarity,
    sometimes even differ in sign from the expected polarity.
    none-the-less, the polarity ratings are quite reasonable overall.
