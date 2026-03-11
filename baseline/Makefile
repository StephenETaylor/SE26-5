
results:	predictions.jsonl
	zip ~/Desktop/submissions.zip predictions.jsonl

t4.bin:	train4.py ../train.json ../hdefs ../biases.sort2
	python train4.py

td4.bin:	train4.py ../train_dev.json ../hdefs ../biases.sort2
	python train4.py ../train_dev.json td4.bin

predictions4.json:	t4.bin fit4.py ../dev.json
	python fit4.py

dev_pred:	t4.bin fit4.py ../dev.json
	python fit4.py ../dev.json
	python score_dev_pred.py 

test_pred:	td4.bin fit4.py ../test.json ../hdefs ../biases.sort2
	python fit4.py ../test.json td4.bin
	cp predictions4.json predictions.jsonl
	zip ~/Desktop/submissions.zip predictions.jsonl

predictions.jsonl:	predictions4.json
	cp predictions4.json predictions.jsonl

pickled.pol:	data_polarity.py	
	python data_polarity.py

sp_score:	sp_score.py
	python sp_score.py
		



