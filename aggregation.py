#-------------------------------------------------------------------------------
# Name:        aggregation
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     16/07/2015_
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import csv,os,sys,copy
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

    date='00_resume_0824'
    #filename='results_mnist_basic.csv'
    filename='results_mnist_background_images.csv'
    dir_input = r'..\data\input'
    dir_output = r'..\data\output\%s'%date
    SHOW=False

    # output  data
    f = open(os.path.join(dir_output,filename.split('.')[0]+'_optSVM.csv'), 'wb')
    writer = csv.writer(f)
    f2 = open(os.path.join(dir_output,filename.split('.')[0]+'_optClsf.csv'), 'wb')
    writer2 = csv.writer(f2)

    #######################
    # load data
    #######################
    reader = csv.reader(open(os.path.join(dir_output,filename), 'rb'), delimiter=',')
    #header
    fnams = next(reader)
    ncol = len(fnams)
    arr = [[0 for i in range(0)] for j in range(ncol)]



    def getVal(row,fieldname):
        return row[fnams.index(fieldname)]

    def setDic(dictionary,row,fieldname):
        dictionary.setdefault(getVal(row,fieldname),'')
        return dictionary[getVal(row,fieldname)]

    # load csv
    for r in reader:
        for i in range(0,ncol):
            if ncol == 1:arr[i].append(int(r[i]))
            try:
                arr[i].append(float(r[i]))
            except ValueError:
                arr[i].append(str(r[i]))
    arr=np.array(arr)
    data=copy.copy(arr)

    #######################
    # select datas
    #######################

    def getData(dat,field_name,value):
        dat = dat[:,dat[fnams.index(field_name)]==value]
        return dat

    exportFields=['Dataset','CLASW','Feature','FeatureParams','optimization','target1','C','gamma','Weight','time','ROC_opt','ROC']
    writer.writerow(exportFields)
    for clsw in np.unique(data[fnams.index('CLASW')]):
        data1 = getData(data,'CLASW',clsw)
        #for feat in np.unique(data[fnams.index('Feature')]):
        for feat in ['PCA','ChainBlur','Gabor_set','Gabor_opencv','Bilateral','Canny','HOG','All_f11']:
            #writer.writerow([])
            #writer.writerow([])
            data2 =  getData(data1,'Feature',feat)
            for feat2 in sorted(np.unique(data[fnams.index('FeatureParams')]),reverse=True):
                data3 =  getData(data2,'FeatureParams',feat2)
                if 0 < data3.shape[1]:
                    writer.writerow([])
                    writer.writerow([])
                    #for opt in np.unique(data[fnams.index('optimization')]):
                    for opt in ['Default','Gsearch','Optunity']:
                        data4 =  getData(data3,'optimization',opt)
                        for dgt in np.unique(data[fnams.index('target1')]):
                            data5 =  getData(data4,'target1',dgt)
                            data5_te =  getData(data5,'Prediction','test')
                            data5_tr =  getData(data5,'Prediction','train')

                            if data5.shape[1]!=0:
                                ex=[]
                                for field in exportFields:
                                    ex.append(data5_tr[fnams.index(field)][0])
                                ex.append(data5_te[fnams.index('ROC')][0])
                                writer.writerow(ex)
    f.close()



    exportFields1=['Dataset','CLASW','Feature','FeatureParams','optimization','target1','Classifier','C','gamma','degree','coef0','Weight','n_estimators','max_features','ROC_opt','ROC']
    exportFields2=['ROC','time']
    writer2.writerow(exportFields1+exportFields2)
    for clsw in np.unique(data[fnams.index('CLASW')]):
        data1 = getData(data,'CLASW',clsw)
        for feat in ['PCA','ChainBlur','Gabor_set','Gabor_opencv','Bilateral','Canny','HOG','All_f11']:
            #writer2.writerow([])
            #writer2.writerow([])
            #writer2.writerow([])
            data2 =  getData(data1,'Feature',feat)
            for feat2 in np.unique(data[fnams.index('FeatureParams')]):
                data3 =  getData(data2,'FeatureParams',feat2)
                if 0 < data3.shape[1]:
                    writer2.writerow([])
                    writer2.writerow([])
                    writer2.writerow([])
                    for opt in ['Optunity_mcl']:
                        data4 =  getData(data3,'optimization',opt)
                        for dgt in np.unique(data[fnams.index('target1')]):
                            data5 =  getData(data4,'target1',dgt)
                            data5_te =  getData(data5,'Prediction','test')
                            data5_tr =  getData(data5,'Prediction','train')

                            if data5.shape[1]!=0:
                                ex=[]
                                for field in exportFields1:
                                    ex.append(data5_tr[fnams.index(field)][0])
                                for field in exportFields2:
                                    ex.append(data5_te[fnams.index(field)][0])
                                writer2.writerow(ex)

    f2.close()


    print 'finish'

    """

    # ----- select fileds
    strs = ''
    for i in range(0,ncol):
        st = 'No. %d: %s'%(i,str(fnams[i]))
        print st
        dat_b[str(fnams[i])]=np.array(arr[i])
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
        labels.append(fnams[fno])
        strs2 += 'N,'
        if type(arr[fno][0])==str:
            strs3 +='No. %d (%d): %s       %s\n'%(i,fno,str(fnams[fno]),str(list(set(arr[fno]))))
        else:
            strs3 +='No. %d (%d): %s       %s\n'%(i,fno,str(fnams[fno]),str([min(arr[fno]),max(arr[fno])]))
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
    """








