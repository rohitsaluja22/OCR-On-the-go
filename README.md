## OCR-On-the-go
## For ICDAR 2019 Paper on End-to-end License Plate and Scene Text Recognition with multi-head attention models

## Contacts

Authors:
Rohit Saluja <rohitsaluja22@gmail.com>

## Requirements
1. Copy the attention_ocr code from https://github.com/tensorflow/models/tree/master/research/attention_ocr to your PC

2. Install tensorflow 1.4.1 with pip
```
virtualenv --system-site-packages ~/.multi_head
source ~/.tensorflow/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu==1.4.1
```

3. Copy files for multi-head attention and dropout from this repo to appropriate locations:
```
cp ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seqCopy.py
cp OCR-on-the-go/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/
cp attention_ocr/eq2seq.py attention_ocr/eq2seqCopy.py
cd ..
cp OCR-On-the-go/seq2seq.py attention_ocr/
```

4. For License Plate experiments and make changes in attention_ocr/datasets/fsns.py:
```
Change size of image to 480 X 260
Change charset_size=134.txt to charset_sizeNPR.txt/charset_sizeDev.txt
Change size of training, validation and test data appropriately
Create folder attention_ocr/datasets/data/fsns/
Copy Charset charset_sizeNPR.txt/charset_sizeDev.txt from OCR-on-the-go/ to attention_ocr/datasets/data/fsns/
```

5. For fsns dataset refer https://github.com/tensorflow/models/tree/master/research/attention_ocr
