ALL = {biases.sort2}

baises.sort2:	biases.txt
	LC_ALL=C sort biases.txt > biases.sort2

