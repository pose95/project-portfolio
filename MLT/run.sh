#!/bin/bash
##imdb_all
#python main.py /home/user/project/JointModel/data/imdb/vocab_5000/ --train_data_file /home/user/project/data/imdb_train.txt.ss --test_data_file /home/user/project/data/imdb_test.txt.ss --num_label 10 --cuda 0 --maxmin 3 --batchsize 1 --temp_batch_num 150000 --dt 300 --vocab 5000 --clf_weight 20 --tm_weight 1 --max_epochs 20 --min_epochs 10 &

###全部imdb数据集 (tutti i set di dati imdb)
#nohup python main.py /data/lengjia/topic_model/imdb/all --train_data_file /data/lengjia/topic_model/imdb/all/train.ss.txt --test_data_file /data/lengjia/topic_model/imdb/all/test.ss.txt --num_label 10 --cuda 0 --maxmin 1 --batchsize 1 --temp_batch_num 20000 --dt 50 --vocab 2000 --clf_weight 9 --tm_weight 1 --max_epochs 50 --min_epochs 20&

#不同的训练测试数据比例 (diverse porzioni di dati di trainind e di test set)
#nohup python main.py /data/lengjia/topic_model/imdb/75+25/ --train_data_file /data/lengjia/topic_model/imdb/75+25/imdb_train.txt.ss --test_data_file /data/lengjia/topic_model/imdb/75+25/imdb_test.txt.ss --num_label 10 --cuda 0 --maxmin 1 --batchsize 1 --temp_batch_num 5000 --dt 50 --vocab 2000 --clf_weight 9 --tm_weight 1 --max_epochs 20 --min_epochs 10&

###不同的α β参数 (diversi parametri a,b)
#nohup python main.py /data/lengjia/topic_model/imdb/tf100/vocab_2000 --train_data_file /data/lengjia/topic_model/imdb/imdb_train.txt.ss --test_data_file /data/lengjia/topic_model/imdb/imdb_test.txt.ss --num_label 10 --cuda 2 --maxmin 3 --batchsize 1 --temp_batch_num 5000 --dt 50 --vocab 2000 --clf_weight 5 --tm_weight 1 --max_epochs 20 --min_epochs 10 &
#nohup python main.py /data/lengjia/topic_model/imdb/tf100/vocab_2000 --train_data_file /data/lengjia/topic_model/imdb/imdb_train.txt.ss --test_data_file /data/lengjia/topic_model/imdb/imdb_test.txt.ss --num_label 10 --cuda 3 --maxmin 3 --batchsize 1 --temp_batch_num 5000 --dt 50 --vocab 2000 --clf_weight 1 --tm_weight 10 --max_epochs 20 --min_epochs 10 &

##yelp2013
nohup python main.py /home/user/project/JointModel/data/yelp2013/vocab_5000/ --train_data_file /home/user/project/data/yelp-2013-train.txt.ss --test_data_file /home/user/project/data/yelp-2013-test.txt.ss --num_label 10 --cuda 0 --maxmin 3 --batchsize 1 --temp_batch_num 100000 --dt 300 --vocab 5000 --clf_weight 20 --tm_weight 1 --max_epochs 20 --min_epochs 10 &

#mio run
python main.py "\Users\matteo posenato\Documents\tesi\codice\MLT" --train_data_file yelp-train.txt.ss --test_data_file yelp-test.txt.ss --num_label 10 --cuda 0 --maxmin 3 --batchsize 1 --temp_batch_num 100000 --dt 300 --vocab 5000 --clf_weight 20 --tm_weight 1 --max_epochs 20 --min_epochs 10

#con train di 20 righe e test di 5
python main.py "\Users\matteo posenato\Documents\tesi\codice\MLT" --train_data_file yelp40-train.txt.ss --test_data_file yelp40-test.txt.ss --num_label 10 --cuda 0 --maxmin 3 --batchsize 1 --temp_batch_num 100000 --dt 5 --vocab 783 --clf_weight 20 --tm_weight 1 --max_epochs 20 --min_epochs 10

python main.py "\Users\matteo posenato\Documents\tesi\codice\MLT" --train_data_file yelp200M-train.txt.ss --test_data_file yelp200M-test.txt.ss --num_label 10 --cuda 0 --maxmin 3 --batchsize 10 --temp_batch_num 100000 --dt 5 --vocab 3000 --clf_weight 20 --tm_weight 1 --max_epochs 5 --min_epochs 10      

python main.py "\Users\matteo posenato\Documents\tesi\codice\MLT" --train_data_file yelp-train.txt.ss --test_data_file yelp-test.txt.ss --num_label 10 --cuda 0 --maxmin 1 --batchsize 50 --temp_batch_num 100000 --dt 5 --vocab 3000 --clf_weight 20 --tm_weight 1 --max_epochs 3 --min_epochs 1       

