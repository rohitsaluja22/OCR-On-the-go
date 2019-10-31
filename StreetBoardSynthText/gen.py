# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 4# no. of times to use the same image
SECS_PER_IMG = 5 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
DB_FNAME = '/mnt/data/1SceneTextBangla/data/synth_il.h5'#osp.join(DATA_PATH,'synth_il.h5')
#DB_FNAME = osp.join("/media/rohit/Seagate Backup Plus Drive/",'dset_8000.h5')
# url of the data (google-drive public file):
# DATA_URL = 'https://www.dropbox.com/s/gnzbtvdemy06xyq/data.tar.gz?dl=1'
#OUT_FILE = 'results/SynthText.h5'
#OUT_FILE = '/media/rohit/Seagate Backup Plus Drive/SynthText1NikoshBAN.h5'
#OUT_FILE = 'results/1AdorshoLipi_20-07-2007.h5'
#OUT_FILE = 'results/3NikoshBAN.h5'# 3) 15844 #2kalpurush.h5' 
#OUT_FILE = 'results/4sagarnormal.h5'
#OUT_FILE = 'results/5akaashnormal.h5' 14825
#OUT_FILE = 'results/6NikoshGrameen.h5' 16154
#OUT_FILE = 'results/7Siyamrupali.h5'
#OUT_FILE = 'results/8AponaLohit.h5'
#OUT_FILE = 'results/9mitra.h5' unknown file format
#OUT_FILE = 'results/10Nikosh.h5' #unknown file format
#OUT_FILE = 'results/11Bangla.h5' 15637
#OUT_FILE = 'results/12Mukti_1.99_PR.h5' 16...
#OUT_FILE = 'results/13BenSenHandwriting.h5'
#OUT_FILE = 'results/14Lohit_14-04-2007.h5' 16103
#OUT_FILE = 'results/15SolaimanLipi_20-04-07.h5' # stopped two times inbetween
#OUT_FILE = 'results/16NikoshLightBan.h5' 15721
#OUT_FILE = 'results/17NikoshLight.h5' 15918
#OUT_FILE = 'results/18BenSen.h5'
#OUT_FILE = 'results/19muktinarrow.h5'
OUT_FILE = 'results/dset_il2.h5'


def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in range(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    db['data'][dname].attrs['lineBB'] = res[i]['lineBB']               
    #db['data'][dname].attrs['txt'] = res[i]['txt']
    L = res[i]['txt']
    print("1:",L)
    L = [n.encode("utf-8", "ignore") for n in L]#"ascii"
    #print("2:",L)
    db['data'][dname].attrs['txt'] = L


def main(viz=False):
  # open databases:
  print (colorize(Color.BLUE,'getting data..',bold=True))
  db = h5py.File(DB_FNAME, 'r')
  print (colorize(Color.BLUE,'\t-> done',bold=True))

  # open the output h5 file:
  out_db = h5py.File(OUT_FILE,'w')#results/SynthText.h5
  out_db.create_group('/data')
  print (colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True))

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)
  # import random as random
  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  for i in range(start_idx,end_idx):
    # i = random.randint(start_idx, end_idx)
    imname = imnames[i]
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = db['depth'][imname][:].T
      # depth = depth[:,:,0]
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      print("here: ", seg.shape, img.size , depth.shape)# = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))# resize img to size of depth map
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))#resize seg to same size as of depth map
      # print("here1: ", seg.shape, img.size , depth.shape, sz)# = db['seg'][imname][:].astype('float32')

      print (colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True))#0 of 4 , 1 of 4.....
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(imname,res,out_db)
      # visualize the output:
      if viz:
        if 'q' in input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print (colorize(Color.GREEN,'>>>> CONTINUING....', bold=True))
      pass
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)
