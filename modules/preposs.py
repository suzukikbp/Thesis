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
import os,time,sys,cv2
from sklearn import svm, datasets
import mlpython.misc.io as mlio
import pylab
import matplotlib.pyplot as plt

#Downloads the MNIST dataset
def obtain(dir_path,kind):
    import urllib
    tt = time.time()
    print 'Downloading the dataset'

    ## Download the main zip file
    dir_path = os.path.expanduser(dir_path)
    #urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip',os.path.join(dir_path,'mnist_basic.zip'))
    if(kind=='mnist_basic'):
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip',os.path.join(dir_path,kind+'.zip'))
    else:
        urllib.urlretrieve('http://www.iro.umontreal.ca/~lisa/icml2007data/'+kind+".zip",os.path.join(dir_path,kind+'.zip'))

    # Extract the zip file
    """
    print 'Extracting the dataset'
    import zipfile
    fh = open(os.path.join(dir_path,'mnist_basic.zip'), 'rb')
    z = zipfile.ZipFile(fh)
    for name in z.namelist():
        outfile = open(os.path.join(dir_path, name), 'wb')
        outfile.write(z.read(name))
        outfile.close()
    fh.close()
    """
    print "  Finish downloading : %.1f sec" %((time.time()-tt))
    print "  You need to unzip downloaded data by hand"
    sys.exit()

#Loads the MNIST basic dataset
def load(dir_path,load_to_memory,kind,pixel):

    tt = time.time()
    print "  Loading"

    input_size = pixel**2
    dir_path = os.path.expanduser(dir_path)
    targets = set(range(10))

    def convert_target(target):
        return target_mapping[target]

    def load_line(line):
        tokens = line.split()
        return (np.array([float(i) for i in tokens[:-1]]), int(float(tokens[-1])))

    train_file,valid_file,test_file = [os.path.join(dir_path, 'mnist_basic_' + ds + '.amat') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [10000, 2000, 50000]
    if load_to_memory:
        train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]

    # Get metadata
    #train_meta,valid_meta,test_meta = [{'input_size':input_size, 'length':l, 'targets':targets} for l in lengths]
    train_meta,valid_meta,test_meta = [{'input_size':input_size,'targets':targets} for l in lengths]

    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

    """
    ##train_file,valid_file,test_file = [os.path.join(dir_path, kind + ds + '.amat') for ds in ['_train','_valid','_test']]
    train_file,test_file = [os.path.join(dir_path, kind + ds + '.amat') for ds in ['_train','_test']]

    # Get data
    ##train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    train,test = [mlio.load_from_file(f,load_line) for f in [train_file,test_file]]

    # Load data to memory
    #lengths = [10000, 2000, 50000]
    #l2 = [1000, 200, 500]
    lengths = [10000,  50000]
    l2 = [1000, 500]
    if load_to_memory:
        ##train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]
        train,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,test],lengths)]

    # Get metadata
    ##train_meta,valid_meta,test_meta = [{'input_size':input_size, 'length':l, 'targets':targets} for l in lengths]
    train_meta,test_meta = [{'input_size':input_size, 'length':l, 'targets':targets} for l in lengths]

    print "  Loading : %.1f sec" %((time.time()-tt))
    #return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}
    return {'train':(train,train_meta),'test':(test,test_meta)}
    """



def makeVarid(dir_path,kind):

    train_file_path = os.path.join(dir_path,kind+'_train.amat')
    valid_file_path = os.path.join(dir_path,kind+'_valid.amat')
    test_file_path = os.path.join(dir_path,kind+'_test.amat')

    # Rename train and test files
    #os.rename(os.path.join(dir_path,'mnist_train.amat'), train_file_path)
    #os.rename(os.path.join(dir_path,'mnist_test.amat'), test_file_path)

    # Split data in valid file and train file
    fp = open(train_file_path)

    # Add the lines of the file into a list
    lineList = []
    for line in fp:
        lineList.append(line)
    fp.close()

    # Create valid file and train file
    valid_file = open(valid_file_path, "w")
    train_file = open(train_file_path, "w")

    # Write lines into valid file and train file
    for i, line in enumerate(lineList):
        if ((i + 1) > 10000):
            valid_file.write(line)
        else:
            train_file.write(line)

    valid_file.close()
    train_file.close()

    ## Delete Temp file
    #os.remove(os.path.join(dir_path,'mnist_basic.zip'))

    print 'Done'

def selectNum(images,labels,nums):
    imglist =[]
    lbllist =[]
    lbllist_binary =[]
    for i in range(0,len(labels)-1):
        if(labels[i] in nums[0]):
            lbllist.append(labels[i])
            imglist.append(images[i])
            lbllist_binary.append(0)
        elif(labels[i] in nums[1]):
            lbllist.append(labels[i])
            imglist.append(images[i])
            lbllist_binary.append(1)
    imglist = np.array(imglist)
    lbllist = np.array(lbllist)
    labels_binary = np.array(lbllist_binary)

    return imglist,lbllist,labels_binary


def drawimg(images,labels,dir_path,kind,pixel):

    # draw the first 10 samples
    # digits.images[i] : image
    # digits.target[i] : the lael of image
    #ncol = 10
    #nrow = 2
    ncol = 40
    nrow = 5
    count = 0

    for index, (image, label) in enumerate(zip(images, labels)[:(nrow*ncol*10-10)]):
        if(index%10!=0):continue
        #if(count==160):
        #    testttt = 0
        #print str(index)+","+str(count)
        image = image.reshape(pixel,pixel)
        if kind.find('mnist_basic') == -1:image = image.T
        pylab.subplot(nrow, ncol, count + 1)
        pylab.axis('off')
        pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
        pylab.title('%i' % label)
        count =count+1

    pylab.savefig(os.path.join(dir_path,"images_"+kind+".png"), dpi=100)
    #print "Showing now"
    #pylab.show()


def drawimgContour(images,labels,contours,dir_path,kind,pixel):

    # draw the first 10 samples
    # digits.images[i] : image
    # digits.target[i] : the lael of image
    ncol = 20
    nrow = 3

    for index, (image, label,cont) in enumerate(zip(images, labels,contours)[:nrow*ncol]):
        image = image.reshape(pixel,pixel)/100.0
        if kind.find('mnist_basic') == -1:image = image.T
        pylab.subplot(nrow, ncol, index + 1)
        pylab.axis('off')
        cv2.drawContours(image, cont[0], -1, (128,255,128), -1 )
        if(len(cont)>1):
            cv2.drawContours(image, cont[1], -1, (128,128,128), -1 )
        pylab.imshow(image, cmap=pylab.cm.gray_r, interpolation='nearest')
        pylab.title('%i' % label)





    pylab.savefig(os.path.join(dir_path,"images_"+kind+".png"), dpi=100)


