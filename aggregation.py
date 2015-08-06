#-------------------------------------------------------------------------------
# Name:        aggregation
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     16/07/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import csv,os,sys
import numpy as np
from operator import itemgetter, attrgetter

import matplotlib.pyplot as plt
import pylab


dat_b={}

def makeTable(nrow,labels=[],constraints=[]):
    outs=[]
    objects=[]
    labs=[]
    for i in range(0,nrow):
        strs=''
        tmp=[]
        flg=True
        for j in range(0,len(labels)):
            val = dat_b[labels[j]][i]
            if constraints[j] == 'N':
                tmp.append(val)
                strs += str(dat_b[labels[j]][i])+'&'
                if i ==0:labs.append(labels[j])
            elif val != constraints[j]:
                flg=False
            """
            if (constraints[j] != 'N') and (val != constraints[j]):
                flg=False
            tmp.append(val)
            strs += str(dat_b[labels[j]][i])+'&'
            """
        strs=strs[:-1]
        strs+= '\\\\\n'

        if(flg==True):
            outs.append(strs)
            objects.append(tuple(tmp))
    return outs,objects,labs

def makeTable2(objs):
    strs=''
    for i in range(0,len(objs)):
        raw=objs[i]
        for j in range(0,len(raw)):
            val = str(raw[j])
            strs += val+'&'
        strs=strs[:-1]
        strs+= '\\\\\n'
    return strs



if __name__ == '__main__':
    dir_input = "..\data\input"
    dir_output = "..\data\output"
    SHOW=False

    #######################
    # load data
    #######################
    reader = csv.reader(open(os.path.join(dir_output,'basicResults.csv'), 'rb'), delimiter=',')
    #header
    fieldnames = next(reader)
    ncol = len(fieldnames)
    arr = [[0 for i in range(0)] for j in range(ncol)]
    # load csv
    for row in reader:
        for i in range(0,ncol):
            if ncol == 1:arr[i].append(int(row[i]))
            try:
                arr[i].append(float(row[i]))
            except ValueError:
                arr[i].append(str(row[i]))

    #######################
    # select datas
    #######################

    # ----- select fileds
    strs = ''
    for i in range(0,ncol):
        st = 'No. %d: %s'%(i,str(fieldnames[i]))
        print st
        dat_b[str(fieldnames[i])]=np.array(arr[i])
        strs+=st+'\n'
    try:
        dat_b['ID']=dat_b['ID'].astype(np.int32)
        dat_b['Target']=dat_b['Target'].astype(np.int32)
    except KeyError:
        print '\n\n\nField name doesnt exist\n  Kill program\n\n '
        sys.exit()


    nrow=len(arr[0])
    if SHOW:
        fields=raw_input('Field number\n\n %s'%strs)
    else:
        fields = '1,2,4,10,11,12,13,14,15'
    print fields

    # ----- set constraints
    labels=[]
    strs2=''
    strs3=''
    i=-1
    for fno in fields.split(','):
        i +=1
        fno = int(fno)
        labels.append(fieldnames[fno])
        strs2 += 'N,'
        if type(arr[fno][0])==str:
            strs3 +='No. %d (%d): %s       %s\n'%(i,fno,str(fieldnames[fno]),str(list(set(arr[fno]))))
        else:
            strs3 +='No. %d (%d): %s       %s\n'%(i,fno,str(fieldnames[fno]),str([min(arr[fno]),max(arr[fno])]))
    if SHOW:
        constraints=raw_input('Constraints %d \n\n %s '%(len(fields.split(',')),strs3))
    else:
        constraints='1436842405,Gabor,N,N,N,N,N,N,N'
    print constraints
    constraints=constraints.split(',')
    i=-1
    for con in constraints:
        i +=1
        try:
            constraints[i]=int(con)
        except ValueError:
            constraints[i]=con
    outputs,objs,labels = makeTable(nrow,labels=labels, constraints=constraints)
    print strs3

    #######################
    # sort data
    #######################
    i=-1
    for obj in objs[0]:
        i +=1
        strs3 +='No. %d %s\n'%(i,str(labels[i]))
        print 'No. %d %s'%(i,str(labels[i]))

    if SHOW:
        sorts=raw_input('Sort number\n%s'%strs3[:-1])
    else:
        sorts='0,1,5'#'4,14,10'

    for sortnb in sorts.split(','):
        #objs = sorted(objs, key=lambda x: x[3])#, reverse=True)
        objs = sorted(objs, key=lambda x: x[int(sortnb)])#, reverse=True)
    print sorts

    str_all=makeTable2(objs)
    #######################
    # output
    #######################
    f = open(os.path.join(dir_output,'test.txt'), 'wb')
    f.write(str_all)
    #for row in outputs:
    #    f.write(row)
    f.close()

#2,11,12,13













