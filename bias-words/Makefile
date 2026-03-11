
syns.dat:	../hdefs
	 gawk -- '    {print $$2}' ../hdefs > syns.dat

biases.txt:	Makefile alg1a.py  smA.bin
	python alg1a.py 30 syns.dat 100

smA.bin:	relationsA.py
	python relationsA.py

