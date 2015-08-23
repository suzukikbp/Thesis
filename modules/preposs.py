# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        preposs
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     31/01/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


# Copyright 2011 Guillaume Roy-Fontaine and David Brouillard. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY Guillaume Roy-Fontaine and David Brouillard ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Guillaume Roy-Fontaine and David Brouillard.

"""
Module ``datasets.mnist_basic`` gives access to the MNIST basic dataset.

| **Reference:**
| An Empirical Evaluation of Deep Architectures on Problems with Many Factors of Variation
| Larochelle, Erhan, Courville, Bergstra and Bengio
| http://www.dmi.usherb.ca/~larocheh/publications/deep-nets-icml-07.pdf

"""


import numpy as np
import os,time,sys,cv2,zipfile,tarfile,shutil
from sklearn import svm, datasets
import mlpython.misc.io as mlio
import pylab
import matplotlib.pyplot as plt


#Download MNIST dataset
def obtain(dir_path,kind):
    import urllib
    tt = time.time()
    print 'Download: '+kind

    # Download the zip/tar file
    dir_path = os.path.expanduser(dir_path)
    if(kind=='mnist_basic'):
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip',os.path.join(dir_path,kind+'.zip'))
    elif 'noise' in kind:
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/'+kind+'.tar.gz',os.path.join(dir_path,kind+'.tar.gz'))
    else:
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/'+kind+".zip",os.path.join(dir_path,kind+'.zip'))
    print "Finish downloading : %.1f sec" %((time.time()-tt))

    # Unpack and make validation data
    if 'noise' in kind:
        untar(dir_path,kind)
        makeValid_noise(dir_path,kind)
    else:
        unzip(dir_path,kind)
        makeValid(dir_path,kind)

# Extract a zip file
def unzip(dir_path,kind):
    print 'Extract zip: ', kind
    fh = open(os.path.join(dir_path,kind+'.zip'), 'rb')
    z = zipfile.ZipFile(fh)
    for name in z.namelist():
        z.extract(name,dir_path)
    fh.close()

# Extract a tar file
def untar(dir_path,kind):
    print 'Extract tar.gz: ', kind
    tar = tarfile.open(os.path.join(dir_path,kind+'.tar.gz'), 'r')
    for item in tar:
        tar.extract(item, dir_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])

# Add the lines of the file into a list
def getLines(fp,name,show=True):
    lineList = []
    for line in fp:
        lineList.append(line)
    fp.close()
    if show:print "Total line " + name+" "+str(len(lineList))
    return lineList

def checkLines(dir_path,kind):
    train_file_path = os.path.join(dir_path,kind+'_train.amat')
    valid_file_path = os.path.join(dir_path,kind+'_valid.amat')
    test_file_path = os.path.join(dir_path,kind+'_test.amat')

    getLines(open(train_file_path),'train')
    getLines(open(valid_file_path),'valid')
    getLines(open(test_file_path),'test')

# Split data in valid file and train file
def makeValid(dir_path,kind):
    train_file_path = os.path.join(dir_path,kind+'_train.amat')
    valid_file_path = os.path.join(dir_path,kind+'_valid.amat')
    test_file_path = os.path.join(dir_path,kind+'_test.amat')
    lineList=getLines(open(train_file_path),'train',show=False)

    # Create valid file and train file
    valid_file = open(valid_file_path, "w")
    train_file = open(train_file_path, "w")

    # Write lines into valid file and train file
    c=0
    for i, line in enumerate(lineList):#line = 28*28
        if ((i + 1) > 10000):
            valid_file.write(line)
        else:
            c=c+1
            train_file.write(line)

    print "Total line " +"train"+" "+str(c)
    print "Total line " +"valid"+" "+str(len(lineList)-c)
    getLines(open(test_file_path),'test',show=True)
    valid_file.close()
    train_file.close()
    ## Delete Temp file
    #os.remove(os.path.join(dir_path,'mnist_basic.zip'))

# Split data in valid file and train file
def makeValid_noise(dir_path,kind):
    for i in range(1,7):
        train_file_path =os.path.join(dir_path,kind+'s_all_'+str(i)+'_train.amat')
        shutil.copy2(os.path.join(dir_path,kind+'s_all_'+str(i)+'.amat'), train_file_path)
        valid_file_path = os.path.join(dir_path,kind+'s_all_'+str(i)+'_valid.amat')
        test_file_path = os.path.join(dir_path,kind+'s_all_'+str(i)+'_test.amat')

        lineList=getLines(open(train_file_path),'train',show=False)

        # Create valid file and train file
        valid_file = open(valid_file_path, "w")
        train_file = open(train_file_path, "w")
        test_file = open(test_file_path, "w")

        # Write lines into valid file and train file
        c=cc=0
        for j, line in enumerate(lineList):#line = 28*28
            if ((j + 1) > 6000):
                test_file.write(line)
            elif ((j + 1) > 5000):
                cc+=1
                valid_file.write(line)
            else:
                c=c+1
                train_file.write(line)

        print "noiselevel: "+str(i)
        print "  Total line " +"train"+" "+str(c)
        print "  Total line " +"valid"+" "+str(cc)
        print "  Total line " +"test"+" "+str(len(lineList)-c-cc)
        valid_file.close()
        train_file.close()
        test_file.close()


#Loads the MNIST basic dataset
def load(dir_path,kind,pixel,trlen=100,vlen=100,telen=1000):
    print "Loading"
    tt = time.time()

    input_size = pixel**2
    dir_path = os.path.expanduser(dir_path)
    targets = set(range(10))

    def convert_target(target):
        return target_mapping[target]

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:-1]]), int(float(tokens[-1])))

    def setToVariable(datas):
        # training data
        images_tr = (datas["train"][0].mem_data[0]).astype(np.float32)
        labels_tr = (datas["train"][0].mem_data[1]).astype(np.int32)
        # test data
        images_te = (datas["test"][0].mem_data[0]).astype(np.float32)
        labels_te = (datas["test"][0].mem_data[1]).astype(np.int32)
        # validation data
        images_val = (datas["valid"][0].mem_data[0]).astype(np.float32)
        labels_val = (datas["valid"][0].mem_data[1]).astype(np.int32)
        return [images_tr,labels_tr,images_val,labels_val,images_te,labels_te]

    train_file,valid_file,test_file = [os.path.join(dir_path, kind+'_' + ds + '.amat') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    lengths = [min(trlen,10000), min(vlen,2000), min(50000,telen)]
    #train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]
    train=mlio.MemoryDataset(train,[(input_size,),(1,)],[np.float64,int],lengths[0])
    valid=mlio.MemoryDataset(valid,[(input_size,),(1,)],[np.float64,int],lengths[1])
    test=mlio.MemoryDataset(test,[(input_size,),(1,)],[np.float64,int],lengths[2])

    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,'targets':targets} for l in lengths]

    print "Loading", int(time.time()-tt),"sec"
    return setToVariable({'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)})


def selectNum(labels,nums):
    lbllist_binary=[]
    for i in range(0,len(labels)):
        if(labels[i] in nums[0]):
            lbllist_binary.append(0)
        elif(labels[i] in nums[1]):
            lbllist_binary.append(1)
    return np.array(lbllist_binary)


def drawimg(images,labels,dir_path,kind,pixel,tnam='',ncol=20,nrow=5,skip=True):
    tt = time.time()
    if(len(pylab.get_fignums())>0):pylab.close()
    count = 0

    for index, (image, label) in enumerate(zip(images, labels)[:(nrow*ncol*10-10)]):
        if skip:
            if(index%4!=0):continue
        if len(image.shape)==1:
            image = image.reshape(pixel,pixel)
        if not 'basic' in kind:image = image.T

        pylab.subplot(nrow, ncol, count + 1)
        pylab.axis('off')
        pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
        pylab.title('%s' % str(label))
        count =count+1

    pylab.suptitle(tnam)
    pylab.savefig(os.path.join(dir_path,kind+".png"), dpi=100)
    pylab.close()
    #print "  Drawing", int(time.time()-tt),"sec"

def drawimgContour(images,labels,contours,dir_path,kind,pixel):
    ncol = 20
    nrow = 3

    for index, (image, label,cont) in enumerate(zip(images, labels,contours)[:nrow*ncol]):
        image = image.reshape(pixel,pixel)/100.0
        if kind.find('basic') == -1:image = image.T
        #if not 'basic' in kind:image = image.T
        pylab.subplot(nrow, ncol, index + 1)
        pylab.axis('off')
        cv2.drawContours(image, cont[0], -1, (128,255,128), -1 )
        if(len(cont)>1):
            cv2.drawContours(image, cont[1], -1, (128,128,128), -1 )
        pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
        pylab.title('%i' % label)

    pylab.savefig(os.path.join(dir_path,"images_"+kind+".png"), dpi=100)









