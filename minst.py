# encoding: utf-8

#-------------------------------------------------------------------------------
# Name:        mnist
# Purpose:
#
# Author:      KSUZUKI
#
# Created:     30/01/2015
# Copyright:   (c) KSUZUKI 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import os,cv2,time,csv,copy,pylab
import numpy as np
import optunity,optunity.metrics
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

from modules.preposs import *
from modules.pca import *
from modules.ccode import *
from modules.evaluation import *
from modules.gabor import *
from modules.gabor_opt import *

SHOW=1
DEBUG=0

dataSets = ['mnist_basic','mnist_background_images','mnist_background_random','mnist_rotation','mnist_rotation_back_image','mnist_rotation_new']
#featureExtrs = ['PCA','ChainBlur','Gabor','Gabor_opt']
featureExtrs = ['Gabor_opt']
opts = ['Default','CV','Gsearch','Optunity']
solvers = ['particle swarm','grid search','random search','cma-es','nelder-mead']
img_tr_all=lab_tr_all=img_te_all=lab_te_all=img_vl_all=lab_vl_all=[]
eigenvalues=eigenvectors=mu=0.0
basicResults=[]
fparams=''  # feature extraction's params
bname = ''  # basic filename

if DEBUG==1:
    featureExtrs = ['Gabor_opt']
    featureExtrs = ['Gabor']
    opts = ['Default','Gsearch']


# normalization
# note that negative value is possible
def norm(imgs):
    imglists=[]
    for img in imgs:
        imglists.append(np.array(img.max()*((img - img.mean()) / img.std()),dtype=img.dtype))
    return np.array(imglists)

def writeout(datas,dir_out):
    labels=["img_tr_all","lab_tr_all","img_val_all","lab_val_all","img_te_all","lab_te_all"]
    for i in range(0,len(datas)):
        data = datas[i]
        f = open(os.path.join(dir_output,labels[i]+'.csv'), 'wb')
        csvw = csv.writer(f)
        if i%2 == 0:
            for j in range(0,(len(data)-1)):
                csvw.writerow(data[j])
        else:
            for j in range(0,(len(data)-1)):
                csvw.writerow([data[j]])
        del csvw
        f.close()

def appendBList(lists):
    for i in lists:
        if type(i)!=str:i=str(i)
        basicResults.append(i)

def appendRList(orglist,lists):
    for i in lists:
        if type(i)!=str:i=str(i)
        orglist.append(i)
    return orglist




def main(targetNumber,pixel,dataName,featureExtr,dir_output):

    #################################
    # 1. Extract the specific number based on "target_names"
    target_names =([targetNumber],np.delete(np.linspace(0,9,10,np.int16),targetNumber))

    # class(0:target, 1:rest of all)
    images_tr,labels_tr,labels_trbi= selectNum(img_tr_all,lab_tr_all,target_names)
    images_vl,labels_vl,labels_vlbi= selectNum(img_vl_all,lab_vl_all,target_names)
    images_te,labels_te,labels_tebi= selectNum(img_te_all,lab_te_all,target_names)
    # set binary data to Y
    ytrain =labels_trbi

    if SHOW:
        if featureExtr == 'PCA':
            for dat,lab,nam in zip([xtrain,xtrain,xtest,xtest],[labels_tr,labels_trbi,labels_te,labels_trbi],['train','train','test','test']):
                show_scores(dat,lab,dir_output,name=bname+nam+'_')
        name = bname.split('_')[0]+'_'+bname.split('_')[2]
        drawimg(img_tr_all,lab_tr_all,dir_output,(name+"_org_char"),pixel,basic=True)
        drawimg(images_tr,labels_trbi,dir_output,name+"_char",pixel,basic=True)

    appendBList([target_names[0][0],-target_names[0][0],len(labels_trbi[labels_trbi==0]),len(labels_trbi[labels_trbi==1]),len(labels_tebi[labels_tebi==0]),len(labels_tebi[labels_tebi==1])])
    print "  #train:%d (%d: %d, %d: %d)"%(len(labels_tr),target_names[0][0],len(labels_trbi[labels_trbi==0]),-target_names[0][0],len(labels_trbi[labels_trbi==1]))
    print "  #test: %d (%d: %d, %d: %d)"%(len(labels_te),target_names[0][0],len(labels_tebi[labels_tebi==0]),-target_names[0][0],len(labels_tebi[labels_tebi==1]))

    #################################
    # 2. Optimization
    ROCs = {}
    for param in opts:
        print '\n   '+param
        t = time.time()

        results = copy.copy(basicResults)
        results.append(param)

        crange = np.logspace(-4,1.5,num=15,base=10.0)
        grange =  np.logspace(-4,1.5,num=15,base=10.0)

        if param == 'Default':
            model =svm.SVC(kernel='rbf',probability=True)
            results=appendRList(results,[model.C,model.gamma,0.0])
            tt=0

        elif param=='CV':
            av=0
            for c in crange:
                for g in grange:
                    clf =svm.SVC(kernel='rbf',C=c,gamma=g,probability=True)
                    scores = cross_validation.cross_val_score(clf,images_vl,labels_vlbi,cv=10, scoring='roc_auc')
                    if(av<np.mean(scores)):
                        av=np.mean(scores)
                        model=svm.SVC(kernel='rbf',C=c,gamma=g,probability=True)
                        tmp=[c,g]
            tt =time.time()-t
            tmp.append(tt)
            results=appendRList(results,tmp)

        elif param == 'Gsearch':
            args = {'kernel': ['rbf'], 'C':crange, 'gamma':grange}
            for score_name, score_func in [('roc_auc', roc_auc_score)]:
                clf = GridSearchCV(svm.SVC(), args, scoring=score_name)
                clf.fit(xtrain,ytrain)
                clf.best_params_['probability']=True
                model=svm.SVC(**clf.best_params_)
                tt =time.time()-t
                results=appendRList(results,[clf.best_params_["C"],clf.best_params_["gamma"],tt])

        elif param == 'Optunity':
            # set score function: twice iterated 10-fold cross-validated accuracy
            @optunity.cross_validated(x=xtrain, y=ytrain, num_folds=10, num_iter=2)
            def svm_acc(x_train, y_train, x_test, y_test, C, gamma):
                model = svm.SVC(kernel='rbf',C=C, gamma=gamma,probability=True).fit(x_train, y_train)
                y_pred = model.predict(x_test)
                return optunity.metrics.accuracy(y_test, y_pred)

            args = {'num_evals':100, 'C':[min(crange),max(crange)], 'gamma':[min(grange),max(grange)]}
            #optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc,pmap=optunity.pmap, **args)
            optimal_pars, optimal_pars2, _ = optunity.maximize(svm_acc, **args)
            optimal_pars['probability']=True
            optimal_pars['kernel']='rbf'

            model =svm.SVC(**optimal_pars)
            tt = optimal_pars2.stats["time"]
            results=appendRList(results,[optimal_pars["C"],optimal_pars["gamma"],tt])
        print '   %s: %0.2f sec'%(param,tt)
        print '    C:%0.2f, g:%0.2f'%(float(results[-3]),float(results[-2]))

        #################################
        # 3. Train model
        for data in ['test','train']:
            print '     '+data
            results.insert(14,data)
            if data == 'test':
                images_true = images_te
                labels_true =labels_tebi
                xpredict = xtest
            elif data=='train':
                images_true = images_tr
                labels_true =labels_trbi
                xpredict =xtrain

            # fit to the test data
            optimal_model = model.fit(xtrain, ytrain)
            ypredict = optimal_model.predict(xpredict)
            #ypredict_prob = optimal_model.predict_proba(xpredict)
            ypredict_score = optimal_model.decision_function(xpredict)

            name = bname+data+'_'+param
            if(SHOW == 1):drawimg(images_true,ypredict,dir_output,name+'_predict',pixel,basic=True)

            #################################
            # 4. Evaluation
            e=Evaluation(labels_true,ypredict,ypredict_score,dir_output,name,results,target_names)
            results,roc=e.evMain()
            if not ROCs.has_key(data): ROCs[data]={param:roc}
            else:ROCs[data].update({param:roc})

            # export
            global fparams
            results.append(fparams)
            csvWriter.writerow(results)
            results = copy.copy(results)[0:15]
            results.remove(data)

    return ROCs

if __name__ == '__main__':

    # set Parameters
    pixel = 28 # pixel size (only for a square)
    dir_input = "..\data\input"
    dir_output = "..\data\output"
    dataSet =dataSets[0]
    global numEx
    numEx = int(time.time())
    trainSize=valSize=200
    teSize=200          #50000
    if DEBUG==1:trainSize=valSize=teSize=10
    normalization = False # MNIST is already normalized

    # file setting
    csvname=os.path.join(dir_output,'basicResults.csv')
    dir_output =os.path.join(dir_output,str(numEx))
    os.mkdir(dir_output)
    dataName = dataSet.split('_')[1]

    print 'No. %d starts'%numEx
    #################################
    # 1. downloade data
    #obtain(dir_input,dataSet)

    #################################
    # 2. make Varidation data
    #makeVarid(dir_input,dataSet)
    #checkLines(dir_input,dataSet)

    #################################
    # 3. data preprocessing
    img_tr_all,lab_tr_all,img_vl_all,lab_vl_all,img_te_all,lab_te_all = load(dir_input,dataSet,pixel,trlen=trainSize,telen=teSize,vlen=valSize)
    #datas = load(dir_input,dataSet,pixel,trlen=trainSize,vlen=valSize,telen=teSize)
    #writeout(datas,dir_output)
    if normalization: img_tr_all,img_te_all,img_te_all = [norm(indat) for indat in [img_tr_all,img_te_all,img_te_all]]

    #################################
    # 4. Extract features
    for featureExtr in featureExtrs:
        t = time.time()
        print '\n'+featureExtr
        bname = dataName+'_'+featureExtr+'_'

        # 4.1 PCA
        if(featureExtr=='PCA'):
            # eigenvectors and corresponding eigenvalues
            [eigenvalues, eigenvectors, mu] = pca(img_tr_all)
            # select top-k largest eigen values
            cp=show_eigenvalues(eigenvalues,dir_output,name=bname,ratio=0.90,cp_show=100)
            show_eigenvalues(eigenvalues,dir_output,name=bname,ratio=0.90,cp_show=eigenvectors.shape[0])
            eigenvalues = eigenvalues[:cp]
            eigenvectors = eigenvectors[:,:cp]
            # projection
            xtrain = pca_project(eigenvectors,img_tr_all,mu)#(#data, cp)
            xtest = pca_project(eigenvectors,img_te_all,mu)
            fparams=cp

        # 4.2. Chaincode/Bluring
        # directional decomposition of image into directional sub-images
        elif(featureExtr=='ChainBlur'):
            xtrain,_ = chaincode(img_tr_all.copy(),pixel,dir_output)
            xtest,_= chaincode(img_te_all.copy(),pixel,dir_output)
            fparams=[]

        # 4.3.1 Gabor
        elif(featureExtr=='Gabor'):
            gab= Gabor(pixel,img_tr_all,img_te_all,dir_input,dir_output,dataName,3.,opt=False)
            #gab= Gabor(pixel,img_tr_all,img_te_all,dir_input,dir_output,dataName,3.,opt=True)
            xtrain,xtest,fparams=gab.gaborMain()

        # 4.3.2 Gabor
        elif(featureExtr=='Gabor_opt'):
            gab= Gabor_opt(pixel,img_tr_all,img_te_all,dir_input,dir_output,dataName,3.)
            xtrain,xtest,fparams=gab.gaborMain()

        tt =time.time()-t
        print "%s: %0.2f sec" %(featureExtr,tt)

        #################################
        # 5. Classification
        for i in range(1,10):
            f = open(csvname, 'ab+')
            global csvWriter
            csvWriter = csv.writer(f)

            bname = dataName+'_'+featureExtr+'_'+str(i)+'_'
            basicResults=[]
            appendBList([dataName,numEx,featureExtr,tt])

            dir_output_ =os.path.join(dir_output,str(i))
            if not os.path.exists(dir_output_):os.mkdir(dir_output_)

            print "\n%s, digit %d "%(dataName,i)
            rocs = main(i,pixel,dataName,featureExtr,dir_output)
            multipleROC(rocs,dir_output_,bname)
            del csvWriter
            f.close()
    print "finish"

