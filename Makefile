ALL:	biases.sort2 train_dev.json hdefs

baises.sort2:	bias-words/biases.txt
	LC_ALL=C sort bias-words/biases.txt > biases.sort2

hdefs:	train.json dev.json test.json baseline/count_synsets.py
	cd baseline; python count_synsets.py

train_dev.json:	train.json dev.json test.json baseline/jsonconcat.py
	baseline/jsonconcat.py

