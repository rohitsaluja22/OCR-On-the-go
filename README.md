## OCR On-the-go
## Code for ICDAR 2019 Paper: "OCR On-the-Go: Robust End-to-end Systems for Reading License Plates and Street Signs", Rohit Saluja, Ayush Maheshwari, Ganesh Ramakrishnan, Parag Chaudhuri, and Mark Carman, International Conference on Document Analysis and Recognition (ICDAR) 2019, Sydney, Australia.	

## Contacts

Authors:
Rohit Saluja <rohitsaluja22@gmail.com>
Ayush Maheshwari <ayush.hakmn@gmail.com>
## Requirements
1. Copy the attention_ocr code from https://github.com/tensorflow/models/tree/master/research/attention_ocr to your PC

2. Install tensorflow 1.1.0 with python 2.7
```
virtualenv --system-site-packages ~/.multi_head
(-p /usr/bin/python2.7)
source ~/.tensorflow/bin/activate
pip2.7 install --upgrade pip
pip2.7 install --upgrade tensorflow-gpu==1.1.0
```

3. Copy files for multi-head attention and dropout from this repo to appropriate locations:
```
cp ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seqCopy.py
cp OCR-on-the-go/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/
cp attention_ocr/seq2seq.py attention_ocr/seq2seqCopy.py
cd ..
cp OCR-On-the-go/seq2seq.py attention_ocr/
```

4. For using inception-resenet-v2 instead of inception-v3:
```
cp attention_ocr/model.py attention_ocr/modelCopy.py
cp OCR-on-the-go/model.py attention_ocr/
cp ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/inception.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/inceptionCopy.py
cp OCR-on-the-go/inception.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/
cp OCR-on-the-go/inception_resnet_v2.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/
```
You can now train and test the models in attention_ocr/ with inception-resenet-v2 and multi-head attention with procedures described in https://github.com/tensorflow/models/tree/master/research/attention_ocr:


5. For License Plate experiments and make changes in attention_ocr/datasets/fsns.py:
```
Change size of image to 480 X 260
Change charset_size=134.txt to charset_sizeNPR.txt/charset_sizeDev.txt
Change size of training, validation and test data appropriately
Create folder attention_ocr/datasets/data/fsns/
Copy Charset charset_sizeNPR.txt/charset_sizeDev.txt from OCR-on-the-go/ to attention_ocr/datasets/data/fsns/
```

6. For fsns dataset refer https://github.com/tensorflow/models/tree/master/research/attention_ocr
