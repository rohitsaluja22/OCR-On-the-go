# Author: Ankush Gupta
# Date: 2015

"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt 
import h5py 
from common import *

########################################################
#FUNCTIONS FOR WRITING TENSORS AND RESIZE STARTS
import re
import sys
import pickle
#import matplotlib
#%matplotlib inline
import cv2
from os import walk
# Important: We are using PIL to read .png files later.
# This was done on purpose to read indexed png files
# in a special way -- only indexes and not map the indexes
# to actual rgb values. This is specific to PASCAL VOC
# dataset data. If you don't want thit type of behaviour
# consider using skimage.io.imread()
from PIL import Image
import skimage.io as io
import tensorflow as tf

def read_charset(filename, null_character=u'\u2591'):
  """Reads a charset definition from a tab separated text file.

  charset file has to have format compatible with the FSNS dataset.

  Args:
    filename: a path to the charset file.
    null_character: a unicode character used to replace '<null>' character. the
      default value is a light shade block .

  Returns:
    a dictionary with keys equal to character codes and values - unicode
    characters.
  """
  pattern = re.compile(r'(\d+)\t(.+)')
  charset = {}
  with tf.gfile.GFile(filename) as f:
    for i, line in enumerate(f):
      m = pattern.match(line)
      if m is None:
        #logging.warning('incorrect charset file. line #%d: %s', i, line)
        print("incorrect charset file at", i , line)
        continue
      code = int(m.group(1))
      print(m.group(2))
      char = m.group(2)#.decode('utf-8')
      if char == '<nul>':
        char = null_character
      charset[char] = code #charset[code] = char 
  return charset

#charset = read_charset('prep_scripts/charset_size.txt')#/home/rohit/src2/1SceneTextBangla/writeTensors/charset_size.txt')
#print(charset)
def squarify(M):
    val = 0
    c = 600
    (a,b)=M.shape
    #print(a,b)
    Flag = 0;
    if c>b:
        padding=((0,0),(0,c-b));
        Flag = 1;
    if c>a:
        padding=((0,c-a),(0,0))
        Flag = 1;
    if(Flag):#FIRST ADD MASK
        newM = np.pad(M,padding,mode='constant',constant_values=val)
        #newM[20:40,20:300] = 0;
        #newM[270:320,400:600] = 0;
        return newM
    else:
        return M

#cntResize = 0

def squarifyRgb(img, cntResize):# img = np.array(rgb)
    M  = img[:,:,0]
    (a,b)=M.shape
    if ((a > 600) or (b >600)):
        cntResize += 1; #print("cntResize", cntResize, a1, b1)
        img0 = Image.fromarray(img)
        print(a,b)
        if a > b:
            print("a>b")
            img0 = img0.resize((b*600//a, 600), Image.ANTIALIAS)
        else:
            print("b>a")
            img0 = img0.resize((600, a*600//b), Image.ANTIALIAS)
        img0 = np.array(img0)
        M  = img0[:,:,0]
        (a,b)=M.shape
    else: img0 = img
    #print(a,b)
    rgbArray = np.zeros((max(a,b),max(a,b),3), 'uint8')
    rgbArray[:,:,0] = squarify(img0[:,:,0])
    rgbArray[:,:,1] = squarify(img0[:,:,1])
    rgbArray[:,:,2] = squarify(img0[:,:,2])
    img1 = Image.fromarray(rgbArray)#.resize((480, 480), Image.ANTIALIAS)
    #M1  = img1[:,:,0]
    (a1,b1,c1)=np.array(img1).shape
    '''if((not(a1 == 600)) or (not(b1 == 600))):
        img1 = img1.resize((600, 600), Image.ANTIALIAS)
        cntResize += 1; print("cntResize", cntResize, a1, b1)'''
    return np.array(img1), cntResize

#FUNCTIONS FOR WRITING TENSORS AND RESIZE ENDS
########################################################

def viz_textbb(text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    #plt.hold(True)
    H,W = text_im.shape[:2]

    # plot the character-BB:
    for i in range(len(charBB_list)):
        bbs = charBB_list[i]
        ni = bbs.shape[-1]
        for j in range(ni):
            bb = bbs[:,:,j]
            bb = np.c_[bb,bb[:,0]]
            plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)

    # plot the word-BB:
    for i in range(wordBB.shape[-1]):
        bb = wordBB[:,:,i]
        bb = np.c_[bb,bb[:,0]]
        plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
        # visualize the indiv vertices:
        vcol = ['r','g','b','k']
        for j in range(4):
            plt.scatter(bb[0,j],bb[1,j],color=vcol[j])        

    plt.gca().set_xlim([0,W-1])
    plt.gca().set_ylim([H-1,0])
    plt.show(block=False)


def isoverlap(grp1,grp2):
    start1 = grp1[1]; end1 = grp1[3]; start2 = grp2[1]; end2 = grp2[3];
    """Does the range (start1, end1) overlap with (start2, end2)?"""
    return end1 >= start2 and end2 >= start1# https://nedbatchelder.com/blog/201310/range_overlap_in_two_compares.html

overlapstr = "\t".encode("utf-8") # " $ "
newClusterstr = " \n ".encode("utf-8") # " $$ "
def combineAndRefereshOverlaps(txt, minX, minY, maxX, maxY, i, j):
    txtN=[]; minXN=[]; minYN=[]; maxXN=[]; maxYN=[];
    if (minX[i] <= minX[j]):
        for l in range(0,len(txt)):
            if (l == i):
                txtN.append(txt [i] + overlapstr + txt [j]);
                minXN.append(min(minX[i],minX[j]));
                minYN.append(min(minY[i],minY[j]));
                maxXN.append(max(maxX[i],maxX[j]));
                maxYN.append(max(maxY[i],maxY[j]));
            elif (not(l == j)):
                txtN.append(txt[l]);
                minXN.append(minX[l]);
                minYN.append(minY[l]);
                maxXN.append(maxX[l]);
                maxYN.append(maxY[l]);
    else:
        #print("hereInCombineElse")
        for l in range(0,len(txt)):
            #print("hereInCombineElseFor", l)
            if (l == j):
                #print("hereInCombineElseForIf1", l,j,i)
                txtN.append(txt [j] + overlapstr + txt [i]);
                minXN.append(min(minX[i],minX[j]));
                minYN.append(min(minY[i],minY[j]));
                maxXN.append(max(maxX[i],maxX[j]));
                maxYN.append(max(maxY[i],maxY[j]));
            elif (not(l == i)):
                #print("hereInCombineElseForIf2", l,j,i)
                txtN.append(txt[l]);
                minXN.append(minX[l]);
                minYN.append(minY[l]);
                maxXN.append(maxX[l]);
                maxYN.append(maxY[l]);
    #print("hereInCombine",len(txtN), len(minXN),i,j,minX[i],minX[j])
    return txtN, minXN, minYN, maxXN, maxYN



def formClusters(txt,minX,minY,maxX,maxY):
	#print("hereStartClustering")-
	TextBoxesCovered = []
	Clusters = []
	Cli = -1
	i =0; groupsCnt = len(txt);
	while (i < groupsCnt):
	    # find first cluster
	    iFlag = 1;
	    j=0;
	    ClstrMinY = minY[i]; PrvClstrMinY = ClstrMinY; ClstrMaxY = maxY[i]; PrvClstrMaxY = ClstrMaxY;
	    while (j < groupsCnt):
	        #print(i,j,minY[i],maxY[i],minY[j],maxY[j])
	        if((not(i==j)) and (not(i in TextBoxesCovered)) and (not(j in TextBoxesCovered))):
	            #if(isoverlap([minX[i],minY[i],maxX[i],maxY[i]], [minX[j],minY[j],maxX[j],maxY[j]])):# if there is overlap between group i and j
	            if(isoverlap([minX[i],ClstrMinY,maxX[i],ClstrMaxY], [minX[j],minY[j],maxX[j],maxY[j]])):# if there is overlap between group i and j
	            # make the cluster instance
	                if(iFlag):
	                    Clusters.append([[txt[i],minX[i],minY[i],maxX[i],maxY[i]],[txt[j],minX[j],minY[j],maxX[j],maxY[j]]]);
	                    Cli += 1;
	                    ClstrMinY = min([ClstrMinY, minY[i], minY[j]])
	                    ClstrMaxY = max([ClstrMaxY, maxY[i], maxY[j]])
	                    TextBoxesCovered.append(j)
	                    iFlag = 0;
	                else:
	                    #print(Cli,len(Clusters), j, len(txt))
	                    Clusters[Cli].append([txt[j],minX[j],minY[j],maxX[j],maxY[j]])
	                    ClstrMinY = min([ClstrMinY, minY[j]])
	                    ClstrMaxY = max([ClstrMaxY, maxY[j]])
	                    TextBoxesCovered.append(j)
	                #[txt, minX, minY, maxX, maxY] = combineAndRefereshOverlaps(txt, minX, minY, maxX, maxY, i, j)
	                #print("hereAfterCombine",i,j,len(txt), len(minX),len(txt), len(minX),txt,minX)
	                #j -= 1;
	                #groupsCnt = len(txt);#print("groupsCnt", groupsCnt);
	        j += 1;
	        jnew = j;
	        #print("j", j, Clusters, ClstrMinY, ClstrMaxY, PrvClstrMinY, PrvClstrMaxY)
	        # Code to take care if i's overlapers overlap new boxes now which ith box didn't:
	        if (j == groupsCnt) and (iFlag == 0) and (not(ClstrMinY == PrvClstrMinY)) and (not(ClstrMaxY == PrvClstrMaxY)):
	            j = 0;
	            #print("in overlaper's overlap", i, len(Clusters))
	        if (jnew == groupsCnt):
	            PrvClstrMinY = ClstrMinY;
	            PrvClstrMaxY = ClstrMaxY;
	    if((iFlag == 1) and (not(i in TextBoxesCovered))):
	        Clusters.append([[txt[i],minX[i],minY[i],maxX[i],maxY[i]]])
	        Cli += 1;
	    TextBoxesCovered.append(i)
	    i += 1;
	return Clusters

def JoinIntraClusters(Clusters):
	Clusters1 = [];
	for Clstr in Clusters:
		ClstrMinX = [Clstr[i][1] for i in range(0,len(Clstr))]
		#print(ClstrMinX)
		indices = np.argsort(np.array(ClstrMinX))
		Clstr1 = [Clstr[i] for i in indices]
		Clstrtxt = ''.encode("utf-8")
		ClstrTxt = [Clstr1[i][0] for i in range(0,len(Clstr1))]
		#print(ClstrTxt)
		for txt in ClstrTxt:
			Clstrtxt += txt + overlapstr;
		ClstrminX = [Clstr1[i][1] for i in range(0,len(Clstr1))]
		CminX = min(ClstrminX);
		ClstrminY = [Clstr1[i][2] for i in range(0,len(Clstr1))]
		CminY = min(ClstrminY);
		ClstrmaxX = [Clstr1[i][3] for i in range(0,len(Clstr1))]
		CmaxX = max(ClstrmaxX);
		ClstrmaxY = [Clstr1[i][4] for i in range(0,len(Clstr1))]
		CmaxY = max(ClstrmaxY);			
		Clusters1.append([Clstrtxt[:-1], CminX, CminY, CmaxX, CmaxY])		
	return Clusters1

def JoinInterClusters(Clusters):
	Clusters1 = [];
	ClstrMinY = [Clusters[i][2] for i in range(0,len(Clusters))]
	indices = np.argsort(np.array(ClstrMinY))
	Clstr1 = [Clusters[i] for i in indices]
	Clstrtxt = ''.encode("utf-8")
	ClstrTxt = [Clstr1[i][0] for i in range(0,len(Clstr1))]
	for txt in ClstrTxt:
		Clstrtxt += txt + newClusterstr;
	ClstrminX = [Clstr1[i][1] for i in range(0,len(Clstr1))]
	CminX = min(ClstrminX);
	ClstrminY = [Clstr1[i][2] for i in range(0,len(Clstr1))]
	CminY = min(ClstrminY);
	ClstrmaxX = [Clstr1[i][3] for i in range(0,len(Clstr1))]
	CmaxX = max(ClstrmaxX);
	ClstrmaxY = [Clstr1[i][4] for i in range(0,len(Clstr1))]
	CmaxY = max(ClstrmaxY);			
	Clusters1.append([Clstrtxt[:-3], CminX, CminY, CmaxX, CmaxY])		
	return Clusters1


def keepithWord(rgb, wordBB, txt, i):

    #Group bbs w.r.t txt #Find minX minY maxX maxY of each group
    minX = []; minY = []; maxX = []; maxY =[];
    indxcovered = 0;
    for groupi in range(0, len(txt)):
        indxcoveredOffset = len(txt[groupi].split())
        #print("groupi", groupi, indxcovered, indxcovered+indxcoveredOffset, wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset])
        minX.append(np.amin(wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset]))
        maxX.append(np.amax(wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset]))
        minY.append(np.amin(wordBB[1,:,indxcovered:indxcovered+indxcoveredOffset]))
        maxY.append(np.amax(wordBB[1,:,indxcovered:indxcovered+indxcoveredOffset]))
        #print(groupi,minX,minY,maxX,maxY)
        indxcovered += indxcoveredOffset

    #Blacklist other groups than i
    for groupi in range(0, len(txt)):
        if(not(groupi==i)):
            print(int(minX[groupi]),int(maxX[groupi]),int(minY[groupi]),int(maxY[groupi]))
            rgb[int(minX[groupi]):int(maxX[groupi]),int(minY[groupi]):int(maxY[groupi]),:] = 0;
   
    return rgb

from PIL import Image, ImageDraw #https://stackoverflow.com/questions/50204604/how-to-draw-a-filled-rotated-rectangle-with-center-coordinates-width-height-an
import math
import numpy.matlib as npm

def reducebbxgpi(rgb, wordBB, groupi):
    rgb = Image.fromarray(rgb)
    polygoni = [ wordBB[0,0,groupi], wordBB[1,0,groupi],  wordBB[0,1,groupi], wordBB[1,1,groupi],wordBB[0,2,groupi], wordBB[1,2,groupi],wordBB[0,3,groupi], wordBB[1,3,groupi]]
    ImageDraw.Draw(rgb).polygon(polygoni, outline=0, fill=0)
    return np.array(rgb)
         
def findNoOfPlates(txt):
    platesCnt = 0
    for txti in txt:#txt contain  text of each paragraph 
        #print(txti, txti.split(b"\n")) # 
        for txtj in txti.split(b"\n"):
            if len(str(txtj).replace(" ","").replace("-","").replace(".","").replace("\n","")) - 3 > 6:
                platesCnt += 1.0;
            else: platesCnt += 0.5;
    return platesCnt

import random
def takeCrop(rgb, b1, a1, ilist, wordBB,txt):
    #Group bbs w.r.t txt #Find minX minY maxX maxY of each group
    minX = []; minY = []; maxX = []; maxY =[];
    indxcovered = 0;
    for groupi in range(0, max(ilist)+1):
        #indxcoveredOffset = len(txt[groupi].split())
        #print("groupi", groupi, indxcovered, indxcovered+indxcoveredOffset, wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset])
        minX.append(np.amin(wordBB[0,:,groupi]))
        maxX.append(np.amax(wordBB[0,:,groupi]))
        minY.append(np.amin(wordBB[1,:,groupi]))
        maxY.append(np.amax(wordBB[1,:,groupi]))
    M  = rgb[:,:,0]
    (a,b)=M.shape#a is height x, b is width w
    minXi = 100000; minYi = 100000;maxXi = 0; maxYi = 0;
    #print(ilist, txt, len(txt))
    for i in ilist:
        minXi = min(minXi,int(minX[i]))
        minYi = min(minYi,int(minY[i]))
        maxXi = max(maxXi,int(maxX[i]))
        maxYi = max(maxYi,int(maxY[i]))
    #a1,b1 h,w 260, 480
    if(minXi -0 < b-maxXi):# crop randomly from left with limits of 0, minXi
        cropXMin = (minXi) - random.randint(0,0 + (minXi)); cropXMin = max(cropXMin,0);
        cropXMax = cropXMin + b1; cropXMax = min(b,cropXMax)
        #print(cropXMin, b1,cropXMax)
    else:# crop randomly from right with limits of 0, b - (maxXi)
        cropXMax = (maxXi) + random.randint(0, b - (maxXi)); cropXMax = min(b,cropXMax);
        cropXMin = cropXMax - b1; cropXMin = max(cropXMin,0)
    if(minYi - 0 < a - maxYi):
        cropYMin = (minYi) - random.randint(0, 0 + (minYi)); cropYMin = max(cropYMin,0);
        cropYMax = cropYMin + a1; cropYMax = min(a,cropYMax)
        #print(cropYMin, a1,cropYMax)
    else:
        cropYMax = (maxYi) + random.randint(0,a - (maxYi)); cropYMax = min(a,cropYMax);
        cropYMin = cropYMax - a1; cropYMin = max(cropYMin,0);
        #print(cropYMin, a1,cropYMax)
    #print("a,b,a1,b1,minXi,maxXi,minYi,maxYi,b1 - (maxXi- minXi),a1 - (maxYi - minYi),cropYMin,cropYMax, cropXMin,cropXMax")
    #print(a,b,a1,b1,minXi,maxXi,minYi,maxYi,b1 - (maxXi- minXi),a1 - (maxYi - minYi),cropYMin,cropYMax, cropXMin,cropXMax)

    rgbCrop = rgb[cropYMin:cropYMax, cropXMin:cropXMax,:]
    M  = rgbCrop[:,:,0]
    (a,b)=M.shape
    cropLRUDRand = random.randint(0,3)
    if(cropLRUDRand == 0): padding=((0,max(a1-a,0)),(0,max(b1-b,0)))
    if(cropLRUDRand == 1): padding=((max(a1-a,0),0),(0,max(b1-b,0)))
    if(cropLRUDRand == 2): padding=((0,max(a1-a,0)),(max(b1-b,0),0))
    if(cropLRUDRand == 3): padding=((max(a1-a,0),0),(max(b1-b,0),0))
    rgbArrayNew = np.zeros((a1,b1,3), 'uint8')
    for ti in range(0,3):
        M = np.pad(rgbCrop[:,:,ti],padding,mode='constant',constant_values=0)
        rgbArrayNew[:,:,ti] = M
    return rgbArrayNew

'''    val = 0
    c = 600
    M  = img[:,:,0]
    (a,b)=M.shape
    #print(a,b)
    Flag = 0;
    if c>b:
        padding=((0,0),(0,c-b));
        Flag = 1;
    if c>a:
        padding=((0,c-a),(0,0))
        Flag = 1;
    if(Flag):#FIRST ADD MASK
        newM = np.pad(M,padding,mode='constant',constant_values=val)
        #newM[20:40,20:300] = 0;
        #newM[270:320,400:600] = 0;
        return newM'''

def getilist(rgb, wordBB, txt, i):
    #print("No of plates", findNoOfPlates(txt))
    #Group bbs w.r.t txt #Find minX minY maxX maxY of each group
    indxcovered = 0;
    subplates = []; cntrSubplates = []
    cntTxti = 0
    platesCnt = 0.0; subplatesCnt=0;wordsCnt = 0;
    # add \n separated plates/subplates to txt1
    #print("here5", i, txt)
    for txti in txt:#txt contain  text of each paragraph 
        #print(txti, txti.split(b"\n")) # 
        for txtj in txti.split(b"\n"):# \n separated words not plates finding index for each i in txt1 as txt1i
            #print(txtj, platesCnt, subplatesCnt)
            subplates.append(txtj)
            cntrSubplates.append(subplatesCnt)
            #print(len(str(txtj).replace(" ","").replace("-","").replace(".","").replace("\n","")), platesCnt)
            #print(platesCnt, i)
            if(platesCnt == float(i)): 
                cntTxti1 = wordsCnt; cntTxti2 = cntTxti1 + len(txtj.split()); cntTxti = subplatesCnt;
            subplatesCnt += 1;
            if len(str(txtj).replace(" ","").replace("-","").replace(".","").replace("\n","")) - 3 > 6:
                platesCnt += 1.0;
            else: platesCnt += 0.5;

            wordsCnt += len(txtj.split())
    #print("platesCnt i", platesCnt, i)
    #print("cntTxti1", cntTxti1)
    #print(txt1, txt1[0], str(txt1[0]), len(str(txt1[0])), len(str(txt1[0]).replace(" ","").replace("-","").replace(".","").replace("\n","")))
    #print(cntTxti)
    # take ilist of ith plates/2subplates from txt1
        #cntTxti1 = wordsCnt
    #print(cntTxti1, cntTxti)
    if(len(str(subplates[cntTxti]).replace(" ","").replace("-","").replace(".","").replace("\n","")) - 3 > 6):
        ilist = [t for t in range(cntTxti1,cntTxti1+len(subplates[cntrSubplates[cntTxti]].split()))]
        plateno = subplates[cntrSubplates[cntTxti]]
    else:
        ilist = [t for t in range(cntTxti1,cntTxti1+len(subplates[cntrSubplates[cntTxti]].split()) + len(subplates[cntrSubplates[cntTxti]+1].split()))]
        plateno = subplates[cntrSubplates[cntTxti]] + subplates[cntrSubplates[cntTxti]+1]
    #print(ilist, cntTxti1, cntTxti, (subplates[cntrSubplates[cntTxti]].split()), (subplates[cntrSubplates[cntTxti]+1].split()))
    #txt = txt1
    #print(txt[0].split(),ilist)
    return ilist, plateno

def keep0thWord(rgb, wordBB, txt, indexlist):# indexlist the index of plates to be kept
    ilist = []; xlist =[];#xlist for sorting left to right
    #print(txt)
    plateno = [];
    #print("here3", indexlist)
    for i in indexlist:
        #print("here4", i)
        try: ilisti, platei = getilist(rgb, wordBB, txt, i)
        except: break;
        plateistr = str(platei).replace(".","").replace("-","").replace("/","").replace(" ","")[2:-1]
        if(len(plateistr)<=10):
            #print("here7", ilisti)
            ilist = ilist + ilisti
            xlisti = []
            for ili in ilisti:
                xlisti.append(wordBB[0,0,ili]); xlisti.append(wordBB[0,1,ili]); xlisti.append(wordBB[0,2,ili]); xlisti.append(wordBB[0,3,ili]);
            xlist.append(min(xlisti))
            plateno.append(plateistr)
            #print("here7",plateno, plateistr)
    #print("here8", ilist, plateno)
    #Blacklist other groups than i
    #print("ilist", ilist)
    if (len(ilist) == 0): return [], ""
    for groupi in range(0, wordBB.shape[-1]):
        if(not(groupi in ilist)):
            #print(int(minX[groupi]),int(maxX[groupi]),int(minY[groupi]),int(maxY[groupi]))
            rgb = reducebbxgpi(rgb, wordBB, groupi)#, int(minY[groupi]):int(maxY[groupi]),int(minX[groupi]):int(maxX[groupi])
    #plateno = str(plateno).replace(" ","").replace("-","").replace(".","").replace("\n","")
    #print(plateno)
    finalplates = ""
    sortedxlistindex = sorted(range(len(xlist)), key=lambda k: xlist[k])
    #print("xlist", xlist, plateno, sortedxlistindex)
    for platen in sortedxlistindex:
        finalplates = finalplates + " " + plateno[platen]

    return takeCrop(rgb, 480, 260, ilist, wordBB,txt), finalplates

def correctReadingOrder(wordBB, txt):
    #print(wordBB)# 2X4Xn
    #print(txt, len(txt))

    #Group bbs w.r.t txt #Find minX minY maxX maxY of each group
    minX = []; minY = []; maxX = []; maxY =[];
    indxcovered = 0;
    for groupi in range(0, len(txt)):
        indxcoveredOffset = len(txt[groupi].split())
        #print("groupi", groupi, indxcovered, indxcovered+indxcoveredOffset, wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset])
        minX.append(np.amin(wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset]))
        maxX.append(np.amax(wordBB[0,:,indxcovered:indxcovered+indxcoveredOffset]))
        minY.append(np.amin(wordBB[1,:,indxcovered:indxcovered+indxcoveredOffset]))
        maxY.append(np.amax(wordBB[1,:,indxcovered:indxcovered+indxcoveredOffset]))
        #print(groupi,minX,minY,maxX,maxY)
        indxcovered += indxcoveredOffset
    #Find overlap of each group
    overlap = [];
    for groupi in range(0, len(txt)):
        overlapCount = 0
        for groupj in range(0, len(txt)):
            if(not(groupi==groupj)):
                if(isoverlap([minX[groupi],minY[groupi],maxX[groupi],maxY[groupi]], [minX[groupj],minY[groupj],maxX[groupj],maxY[groupj]])):
                    overlapCount += 1
        overlap.append(overlapCount)
    #print("overlap", overlap)

    if (1):#not(max(overlap) == 0):
	    #Sort in decreasing order of overlaps of each group
	    indices = np.argsort(-np.array(overlap))
	    overlap = [overlap[i] for i in indices]
	    txt = [txt[i] for i in indices]
	    minY = [minY[i] for i in indices]
	    minX = [minX[i] for i in indices]
	    maxY = [maxY[i] for i in indices]
	    maxX = [maxX[i] for i in indices]

	    #Form The Clusters
	    Clusters = formClusters(txt,minX,minY,maxX,maxY);
	    #print("Clusters", len(Clusters), Clusters)

	    #Join clusters in order left to right i.e sorted order of miX, and find minY of each cluster
	    Clusters1 = JoinIntraClusters(Clusters)
	    #print("Clusters1", len(Clusters1), Clusters1)

	    #Join clusters in order top to bottom
	    Clusters2 = JoinInterClusters(Clusters1)
	    #print("Clusters2", len(Clusters2), Clusters2)

	    '''#Read the "group with highest no of overlaps" and "all its overlaps" in order left to right i.e sorted order of miX, and find minY of each cluster
	    #Clusters = []
	    i =0; j=0; groupsCnt = len(txt);
	    while (i < groupsCnt):
	        # find first cluster		
	        while (j < groupsCnt):
	            #print(i,j,minY[i],maxY[i],minY[j],maxY[j])
	            if(not(i==j)):
	                if(isoverlap([minX[i],minY[i],maxX[i],maxY[i]], [minX[j],minY[j],maxX[j],maxY[j]])):# if there is overlap between group i and j
	                    [txtN, minXN, minYN, maxXN, maxYN] = combineAndRefereshOverlaps(txt, minX, minY, maxX, maxY, i, j)
	                    minX = []; minY = []; maxX = []; maxY =[]; txt = [];
	                    #print("hereAfterCombine",i,j,len(txtN), len(minXN),len(txt), len(minX),txtN,minXN)
	                    j -= 1;
	                    minX = minXN; minY = minYN; maxX = maxXN; maxY =maxYN; txt = txtN;
	                    groupsCnt = len(txt);#print("groupsCnt", groupsCnt);
	            j += 1;
	        i += 1;
	    #print(len(txt), len(minX))'''
	    '''for words in txt:
	        print("here1: ", words.decode("utf-8"))'''
	    '''# join text of clusters read above in order of minY sorted from top to bottom
	    indices = np.argsort(np.array(minY)); print(indices)
	    txtInReadingOrder = ''.encode("utf-8")
	    for i in indices:
	        txtInReadingOrder += txt[i] + newClusterstr'''
	    txtInReadingOrder = Clusters2[0][0]	
    else:
	    indices = np.argsort(np.array(minY))
	    txtInReadingOrder = ''.encode("utf-8")
	    for i in indices:
	        txtInReadingOrder += txt[i] + newClusterstr
    return txtInReadingOrder

import glob, os
import scipy.misc
os.chdir(".")

def main(db_fname):
    #print(db_fname)
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    #print(db.keys(), dsets)
    print ("total number of images : ", colorize(Color.RED, len(dsets), highlight=True))
    N=5
    #Generate logs 1:
    imageh =[]; imagew =[]; texth =[]; textw = []; ImagesUsed = [];maxMinMaxX =[]
    cnt = 0; maxStrLen = 0;maxStr ='';
    #Writing tensors 1
    cntResize = 0; cntGr10 = 0;
    for k in dsets:
        #k = 'hiking_125.jpg_0'
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']
        '''txtInReadingOrder = (correctReadingOrder(wordBB, txt).decode("utf-8"))
        
        #Generate Logs 2
        cnt += 1; print(cnt);
        imageh.append(rgb.shape[0])
        imagew.append(rgb.shape[1])
        minX = np.amin(wordBB[0,:,:]); maxX = np.amax(wordBB[0,:,:]);
        minY = np.amin(wordBB[1,:,:]); maxY = np.amax(wordBB[1,:,:]);
        if (maxX>600): # certain bounding boxes exceed X dimention
            maxX = 600;
        texth.append(maxY - minY)
        textw.append(maxX - minX)
        maxMinMaxX.append([minX,maxX])
        txtInReadingOrderNew = txtInReadingOrder.replace(" \n ", "%").replace("\n", "$").replace("\t", "#")
        print(txtInReadingOrderNew)#, len(txtInReadingOrder.replace(" \n ", "%").replace("\n", "$").replace("\t", "#")))
        prevmaxStrLen = maxStrLen; maxStrLen = max(len(txtInReadingOrderNew), maxStrLen);
        if (maxStrLen > prevmaxStrLen): maxStr = txtInReadingOrderNew;
        ImagesUsed.append(k.split(".jpg_")[0]+".jpg")'''

        #Writing tensors 1
        #rgb, cntResize = squarifyRgb(rgb, cntResize)

        #saveImage
        #scipy.misc.imsave("{0:0=3d}".format(cnt)+".jpeg", rgb)
        # keep 1st word make others black
        Nplates = int(findNoOfPlates(txt));
        #print("here0", Nplates)
        for t in range(0,math.ceil(Nplates/N)):
            try:
                rgb1,plateno = keep0thWord(rgb, wordBB, txt, list(range(5*t,min(5*t+5,Nplates))))#t
                #print("here1", Nplates)
                if(not(plateno == "")):
                    scipy.misc.imsave("SinglePlates/{0:0=3d}".format(cnt)+"_"+str(t)+".jpeg", rgb1)
                    print("SinglePlates/{0:0=3d}".format(cnt)+"_"+str(t)+".jpeg", plateno[1:])
                #else: print("invalid txt at 5%", txt)
            except:
                pass
        cnt += 1;
        #break;

        #visualize each:
        '''viz_textbb(rgb, [charBB], wordBB)
        print ("image name        : ", colorize(Color.RED, k, bold=True))
        #print ("  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1]))
        print ("  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1]))
        #for i in txt:
        print ("  ** text         : ", colorize(Color.GREEN, plateno))#txtInReadingOrder.replace(" \n ", "\n")))
        #print(txtInReadingOrder.replace(" \n ", "\n")

        if 'q' in input("next? ('q' to exit) : "):
            break'''
    db.close()
    #print(cntGr10)
    #Generate Logs 3
    '''print("cntResize",cntResize)
    print("UniqImagesUsed", len(set(ImagesUsed)))
    print("maxStrLen", maxStrLen ,"maxStr", maxStr)
    print("max(imageh), max(imagew), max(texth), max(textw):",max(imageh), max(imagew), max(texth), max(textw), maxMinMaxX[np.argmax(textw)])
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(imageh)
    axs[0, 1].hist(imagew)
    axs[1, 0].hist(texth)
    axs[1, 1].hist(textw)
    #plt.subplot_tool()
    plt.show()'''

if __name__=='__main__':
    main('results/SynthNPR.h5')
