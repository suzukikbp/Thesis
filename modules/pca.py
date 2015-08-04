# encoding: utf-8

import os
import numpy as np
import copy
import matplotlib.pyplot as plt


class PCA():

    # PCA calc
    #   @datas: train,test,val
    #   @return: projections of train,test,val,CP
    def main(self,datas,dir_output,bname):
        # 1. eigenvectors and corresponding eigenvalues
        [eigenvalues, eigenvectors, mu] = self.pca(datas[0])

        # 2. select #CP
        cp=self.show_eigenvalues(eigenvalues,dir_output,name=bname,ratio=0.90,cp_show=100)
        self.show_eigenvalues(eigenvalues,dir_output,name=bname,ratio=0.90,cp_show=eigenvectors.shape[0])
        #eigenvalues,eigenvectors=self.select_eigenvalues(eigenvalues,eigenvectors,num_components=cp)
        eigenvalues,eigenvectors = eigenvalues[:cp],eigenvectors[:,:cp]

        # 3. projection
        xtrain,xtest,xval = [self.pca_project(eigenvectors,dat,mu)for dat in datas]
        return xtrain,xtest,xval,cp

    def pca_project(self,W, X, mu=None):
        if mu is None:
            return np.dot(X,W)
        return np.dot(X - mu, W)

    def pca_reconstruct(self,W, Y, mu=None):
        if mu is None:
            return np.dot(Y,W.T)
        return np.dot(Y, W.T) + mu

    def show_scores(self,data,lab,dir_output,name='',cp=5):
        col=["b",'orange',"g","r","c","m","y","k","w","#77ff77"]
        for i in range(1,cp):
            plt.clf()
            for j in range(0,len(set(lab))):
                idx=np.where(lab==j)
                plt.scatter(data[idx,0],data[idx,i],marker='o',label=str(j),c=col[j])
            plt.xlabel("Principal component 1")
            plt.ylabel("Principal component "+str(i+1))
            plt.legend()
            #plt.savefig(os.path.join(dir_output,'PC'+str(i+1)+'_'+dataname+'_'+str(len(set(lab)))+'.png'))
            plt.savefig(os.path.join(dir_output,name+'PC'+str(i+1)+'_'+str(len(set(lab)))+'.png'))

    def show_eigenvalues(self,eigenvalues,dir_output,name='',cp_show=100,ratio=0.90):
        cp=len(eigenvalues)
        culm=[]
        val=0
        val_all=sum(eigenvalues)

        for i in range(0,len(eigenvalues)-1):
            val=eigenvalues[i]+val
            culm.append(val*1.0/val_all)
            if((val*1.0/val_all>ratio)and(i<cp)):cp=i
        plt.clf()
        plt.plot(culm[0:cp_show])
        plt.xlabel("Index")
        plt.ylabel("Cumulated Eigenvalues (%)")
        plt.savefig(os.path.join(dir_output,name+'eigenvalues_'+str(cp_show)+'.png'))
        return cp

    # select only num_components
    def select_eigenvalues(self,eigenvalues,eigenvectors,num_components=0):
        eigenvalues = eigenvalues[0:num_components].copy()
        eigenvectors = eigenvectors[:,0:num_components].copy()
        return eigenvalues, eigenvectors

    def pca(self,X, num_components=0):
        [n,d] = X.shape
        if (num_components <= 0) or (num_components>n):
            num_components = n
        mu = X.mean(axis=0),
        X = X - mu
        if n>d:
            C = np.dot(X.T,X)
            [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        else:
            C = np.dot(X,X.T)
            [eigenvalues,eigenvectors] = np.linalg.eigh(C)
            eigenvectors = np.dot(X.T,eigenvectors)
            for i in xrange(n):
                eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
        # or simply perform an economy size decomposition
        # eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
        # sort eigenvectors descending by their eigenvalue
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        return [eigenvalues, eigenvectors, mu]

