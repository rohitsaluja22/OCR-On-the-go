## OCR On-the-go
## Code for ICDAR 2019 Paper: "OCR On-the-Go: Robust End-to-end Systems for Reading License Plates and Street Signs".
## Contacts

Authors:
Rohit Saluja <rohitsaluja22@gmail.com>,
Ayush Maheshwari <ayush.hakmn@gmail.com>,
 Ganesh Ramakrishnan, Parag Chaudhuri, and Mark Carman
## Requirements
1. Copy the attention_ocr code from https://github.com/tensorflow/models/tree/master/research/attention_ocr to your PC

2. Install tensorflow 1.1.0 with python 2.7
```
virtualenv --system-site-packages ~/.multi_head
(try -p /usr/bin/python2.7 instead of --system-site-packages if default is python3.x)
source ~/.multi_head/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow-gpu==1.1.0
```

3. Copy files for multi-head attention and dropout from this repo to appropriate locations:
```
cp ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seqCopy.py
cp OCR-on-the-go/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/
cp attention_ocr/seq2seq.py attention_ocr/seq2seqCopy.py
cd ..
cp OCR-On-the-go/seq2seq.py attention_ocr/
```
If using inception-v3 (default encoder in attention_ocr code), change attention_feature_vector_size = 288 in line 9 seq2seq.py.
For addition/changes made in all the python scripts w.r.t. attention_ocr code, search for comments with phrase "OCR-on-the-go" in individual files.
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
6. To generate Synthetic dataset with pyton3.x related to License Plates:-
```
generate dset.h5 from 8000 Synthtext backgrounds as described in https://github.com/ankush-me/SynthText
Download code for SynthText in Arabic (as available in python3, later used for adding Devanagari as well, so used this for License Plates as well):-
git clone https://github.com/adavoudi/SynthText.git
Save dset.h5 in SynthText/dset_path/
download fonts from https://www.fontspace.com/category/license%20plate and save to SynthText/data/fonts/NPR/ (create directory NPR in SynthText/data/fonts/)
cp NPRSynthText/fontlist.txt SynthText/data/fonts/fontlistCopy.txt
cp NPRSynthText/fontlist.txt SynthText/data/fonts/
cp NPRSynthText/ExtractSinglePlates.py SynthText/
cp SynthText/gen.py SynthText/genCopy.py
cp SynthText/text_utils.py SynthText/text_utilsCopy.py
cp NPRSynthText/text_utils.py SynthText/
cp NPRSynthText/gen.py SynthText/
cd SynthText/data/newsgroup/
cp NPRSynthText/newsgroup.txt SynthText/data/newsgroup/newsgroup.txt
cp newsgroup.txt newsgroupCopy.txt
wget https://drive.google.com/file/d/1iFrJDSaAxy0ZBhgSu1TCOGF0ekUWd9H-/view?usp=sharing
unzip newsgroup.txt.zip
cd ../../../SynthText/
python3 prep_scripts/update_freq.py #for updating counts for newsgroup.txt related to NPR, if it does not work move update_freq.py to SynthText/ and try running it from there
python3 invert_font_size.py
python3 gen.py # this will generate results/SynthNPR.h5, with each scene having multiple-license plate numbers
mkdir SinglePlates
python3 ExtractSinglePlates.py > SinglePlateAnnotations.txt # to generate scenes of size 480 X 260 each with single license plate, and annotations will be stored in SinglePlateAnnotations.txt
```
7. To generate Synthetic dataset with pyton3.x related to Multi-lingual Indic Paragraphs:-
```
Install pycharm from source after making changes using https://bitbucket.org/pygame/pygame/pull-requests/52/add-complex-text-layout-to-pygamefreetype/diff
Download all unicode fonts from http://indiatyping.com/index.php/download/top-50-hindi-unicode-fonts-free
```
8. For fsns dataset refer https://github.com/tensorflow/models/tree/master/research/attention_ocr
