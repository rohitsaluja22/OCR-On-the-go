## OCR On-the-go
## Code for ICDAR 2019 Paper: "OCR On-the-Go: Robust End-to-end Systems for Reading License Plates and Street Signs".
You can read the paper here: https://www.cse.iitb.ac.in/~rohitsaluja/PID6011503.pdf
## Contacts

Authors:
Rohit Saluja <rohitsaluja22@gmail.com>,
Ayush Maheshwari <ayush.hakmn@gmail.com>,
Ganesh Ramakrishnan, Parag Chaudhuri, and Mark Carman

## Annotation Framework: StreetOCRCorrect

Our annotation framework for license plates can be found here: https://github.com/rohitsaluja22/StreetOCRCorrect
Feel free to update it for street signs and share

## Requirements
1. Copy the attention_ocr code from https://github.com/tensorflow/models/tree/master/research/attention_ocr to your PC

2. Install tensorflow 1.1.0 with python 2.7
```
$ virtualenv --system-site-packages ~/.multi_head
(try -p /usr/bin/python2.7 instead of --system-site-packages if default is python3.x)
$ source ~/.multi_head/bin/activate
$ pip install --upgrade pip
$ pip install --upgrade tensorflow-gpu==1.1.0
```

3. Copy files for multi-head attention and dropout from this repo to appropriate locations:
```
$ cp ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/seq2seqCopy.py
$ cp OCR-On-the-go-master/seq2seq.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/legacy_seq2seq/python/ops/
$ cp attention_ocr/sequence_layers.py attention_ocr/sequence_layersCopy.py
$ cd ..
$ cp OCR-On-the-go-master/sequence_layers.py attention_ocr/
```
If using inception-v3 (default encoder in attention_ocr code), change attention_feature_vector_size = 288 in line 9 seq2seq.py.
For addition/changes made in all the python scripts w.r.t. attention_ocr code, search for comments with phrase "OCR-on-the-go" in individual files.
4. For using inception-resenet-v2 instead of inception-v3:
```
Change 'final_endpoint' to 'Mixed_6a' in common_flags.py or pass it via arguments
$ cp attention_ocr/model.py attention_ocr/modelCopy.py
$ cp OCR-On-the-go-master/model.py attention_ocr/
$ cp ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/inception.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/inceptionCopy.py
$ cp OCR-On-the-go-master/inception.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/
$ cp OCR-On-the-go-master/inception_resnet_v2.py ~/.multi_head/lib/python2.7/site-packages/tensorflow/contrib/slim/python/slim/nets/
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
use this https://github.com/JarveeLee/SynthText_Chinese_version/blob/master/add_more_data.py #add try and except if some background images are missing in the source
Download code for SynthText in Arabic (as available in python3, later used for adding Devanagari as well, so used this for License Plates as well):-
$ git clone https://github.com/adavoudi/SynthText.git
Save dset.h5 in SynthText/dset_path/
download fonts from https://www.fontspace.com/category/license%20plate and save to SynthText/data/fonts/NPR/ (create directory NPR in SynthText/data/fonts/)
$ cp NPRSynthText/fontlist.txt SynthText/data/fonts/fontlistCopy.txt
$ cp NPRSynthText/fontlist.txt SynthText/data/fonts/
$ cp NPRSynthText/ExtractSinglePlates.py SynthText/
$ cp SynthText/gen.py SynthText/genCopy.py
$ cp SynthText/text_utils.py SynthText/text_utilsCopy.py
$ cp NPRSynthText/text_utils.py SynthText/
$ cp NPRSynthText/gen.py SynthText/
$ cd SynthText/data/newsgroup/
$ cp NPRSynthText/newsgroup.txt SynthText/data/newsgroup/newsgroup.txt
$ cp newsgroup.txt newsgroupCopy.txt
$ wget https://drive.google.com/file/d/1iFrJDSaAxy0ZBhgSu1TCOGF0ekUWd9H-/view?usp=sharing
$ unzip newsgroup.txt.zip
$ cd ../../../SynthText/
$ python3 prep_scripts/update_freq.py #for updating counts for newsgroup.txt related to NPR, if it does not work move update_freq.py to SynthText/ and try running it from there
$ python3 invert_font_size.py
$ python3 gen.py # this will generate results/SynthNPR.h5, with each scene having multiple-license plate numbers
$ mkdir SinglePlates
$ python3 ExtractSinglePlates.py > SinglePlateAnnotations.txt # to generate scenes of size 480 X 260 each with single license plate, and annotations will be stored in SinglePlateAnnotations.txt
```
7. To generate Synthetic dataset with pyton3.x related to Multi-lingual Indic Paragraphs:-
```
0. Install https://github.com/HOST-Oman/libraqm (we first installed harfbuzz-2.4.0, before that sudo apt-get install ragel might be needed, and then freetype-2.10.0 from source code)
$ sudo apt-get install libfreetype6-dev libharfbuzz-dev libfribidi-dev gtk-doc-tools
$ git clone https://github.com/HOST-Oman/libraqm.git
$ cd libraqm
$ ./autogen.sh
$ ./configure
$ make
$ make install #might need sudo for this
1. Install pycharm from source after making changes using https://bitbucket.org/pygame/pygame/pull-requests/52/add-complex-text-layout-to-pygamefreetype/diff
Or alternatively, works on Ubuntu 16.04, in this repo go to pygame-1.9.3:-
(refer these prerequisites https://askubuntu.com/questions/401342/how-to-download-pygame-in-python3-3)
$ cd pygame-1.9.3
$ sudo rm -fr /usr/local/lib/python3.5/dist-packages/pygame* #OR remove it from site-packages of python3.5 virtual env you are working on
$ python3.5 setup.py #might need sudo for this if not in venv
if you get sdl related error on import pycharm in python3.5:-
$ sudo apt-get install libsdl-ttf2.0-0

2. Download all unicode fonts from http://indiatyping.com/index.php/download/top-50-hindi-unicode-fonts-free
3. Use IndicBackgrounds.h5 file which has frames from around 200 real background videos. Both are shared at https://www.cse.iitb.ac.in/~rohitsaluja/project
4. Go to SyntheDevEn and follow instructions similar to SynthNPR
```
8. For fsns dataset refer https://github.com/tensorflow/models/tree/master/research/attention_ocr
