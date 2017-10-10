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
export PYTHONPATH=$PYTHONPATH:'pwd':'pwd'/slim
export PYTHONPATH=/home/ubuntu/models/research:/home/ubuntu/models/research/slim
echo $PYTHONPATH
chmod u+x setup.py 
python setup.py build
python setup.py install
chmod u+x slim/setup.py 
python slim/setup.py build
python slim/setup.py install
python object_detection/builders/model_builder_test.py


cd object_detection/models
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
tar -xvzf community_images.tar.gz

wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz
tar -xvzf community_images.tar.gz

cd /ubuntu/
mkdir Data
cd Data
git clone https://github.com/MumMumMum/DataPrep.git
cp -i create_Udacity_tf_record.py   /home/ubuntu/models/research/object_detection
cp -i UTL_label_map.pbtxt /home/ubuntu/models/research/object_detection/data

python create_Udacity_tf_record.py --data_dir=/home/ubuntu/Data/DataPrep/Udacity_cropped --output_dir=/home/ubuntu/Data/DataPrep/Udacity_cropped
cd /home/ubuntu/Data/DataPrep/Udacity_cropped
rename UTL_Train.record  UTL_Train_cropped.record
rename UTL_Eval.record   UTL_Eval_cropped.record
cp -i UTL_Train_cropped.record  /home/ubuntu/models/research/object_detection/data/
cp -i UTL_Eval_cropped.record   /home/ubuntu/models/research/object_detection/data/

python create_Udacity_tf_record.py --data_dir=/home/ubuntu/Data/DataPrep/Udacity_fullSize --output_dir=/home/ubuntu/Data/DataPrep/Udacity_fullSize
cd /home/ubuntu/Data/DataPrep/Udacity_fullSize
rename UTL_Train.record UTL_Train_full.record
rename UTL_Eval.record  UT_Eval_full.record
cp -i UTL_Train_full.record  /home/ubuntu/models/research/object_detection/data/
cp -i UTL_Eval_full.record   /home/ubuntu/models/research/object_detection/data/

cd /home/ubuntu/models/research/object_detection

#train cropped iamge on SSD mobile net
python train.py --logtostderr --pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco_Udacity.config 
--train_dir=/home/ubuntu/models/reserach/object_detection/data

#saving cropped image on SSD mobile net for inference
python export_inference_graph.py 
--pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco_Udacity.config 
--trained_checkpoint_prefix=models/ssd_mobilenet_v1_coco_11_06_2017/model.ckpt-XXXX 
--output_directory=models/ssd_mobilenet_v1_coco_11_06_2017/

#evaluation for image on SSD mobile net
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco_Udacity.config \
    --checkpoint_dir=models/ssd_mobilenet_v1_coco_11_06_2017/ \
    --eval_dir=models/ssd_mobilenet_v1_coco_11_06_2017/eval
tensorboard --logdir=/training/SSD_mobile_net --host=0.0.0.0 --port=8080 
http://yourURL:8080

#train cropped iamge on SSD inception net
python train.py --logtostderr --pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco_Udacity.config 
--train_dir=/home/ubuntu/models/reserach/object_detection/data

#saving cropped image on SSD  inception for inference
python export_inference_graph.py 
--pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/ssd_inception_v2_coco_Udacity.config 
--trained_checkpoint_prefix=models/ssd_inception_v2_coco_11_06_2017/model.ckpt-XXXX 
--output_directory=models/ssd_inception_v2_coco_11_06_2017/

tensorboard --logdir=/training/SSD_inception --host=0.0.0.0 --port=8080 
python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=/home/ubuntu/models/research/object_detection/models/ssd_inception_v2_coco_11_06_2017/ssd_mobilenet_v2_coco_Udacity.config \
    --checkpoint_dir=models/ssd_inception_v2_coco_11_06_2017/ \
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

