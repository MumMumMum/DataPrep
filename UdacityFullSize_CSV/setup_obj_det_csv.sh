# start instance with SG setting rule anywhere and 8080-9000 for TCP R, HTTPS

#!/bin/bash
pip install tensorflow-gpu
sudo apt-get update
sudo apt-get dist-upgrade
sudo apt-get install protobuf-compiler python-pil python-lxml
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib

git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
#export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim
#export PYTHONPATH=/home/ubuntu/models/research:/home/ubuntu/models/research/slim
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
echo $PYTHONPATH
chmod u+x setup.py 
python setup.py build
python setup.py install
chmod u+x slim/setup.py 
python slim/setup.py build
python slim/setup.py install
python object_detection/builders/model_builder_test.py
====================================================================================================
#get models
cd  object_detection/models
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
tar -xvzf ssd_inception_v2_coco_11_06_2017.tar.gz

====================================================================================================================
#get Data from git images and xml
cd /home/ubuntu/
git clone https://github.com/MumMumMum/DataPrep.git
cd DataPrep/UdcaityFullSize_CSV
cp -f UTL_label_map.pbtxt /home/ubuntu/models/research/object_detection/data
cp -f ssd_inception_v2_coco_Udacity.config /home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017


========================================================================================================================
#genertae and move TFrecord in data


python generate_tfrecord_udacity.py --csv_input=train_labels.csv  --output_path=train.record
python generate_tfrecord_udacity.py    --csv_input=test_labels.csv  --output_path=test.record

cp -i train.record  /home/ubuntu/models/research/object_detection/data/
cp -i test.record   /home/ubuntu/models/research/object_detection/data/

=========================================================================================================================================

+====================================================================================================================================================
#train cropped iamge on SSD inception net
python train.py --logtostderr --pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco_Udacity.config 
--train_dir=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/train

#saving cropped image on SSD  inception for inference
python export_inference_graph.py 
--pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco_Udacity.config 
--trained_checkpoint_prefix=/home/ubuntu/models/reserach/object_detection/models/ssd_inception_v2_coco_11_06_2017/train/model....
--output_directory=models/fz_models/ssd_inception_v2_coco_11_06_2017/

tensorboard --logdir=models/ssd_inception_v2_coco_11_06_2017/ --host=0.0.0.0 --port=8080 
python eval.py \
    --logtostderr \
    --pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco_Udacity.config \
    --checkpoint_dir=models/ssd_inception_v2_coco_11_06_2017/train\
    --eval_dir=models/ssd_inception_v2_coco_11_06_2017/eval
	
	
+=========================================================================================================================================
http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz

check your working model.
check how to work with board view
understand the graphs
make simulator images
make git for image xml annotation
check how to download and unzip model
Bosch Data set model.
SIM images.

579 , 580, 581