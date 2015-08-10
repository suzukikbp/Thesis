# encoding: utf-8
# gabor.py
# 2012-3-8
# Eiichiro Momma
#__author__ = 'momma'

import numpy as np
import cv2 as cv
import os

src_f = 0
pos_ks =21
pos_sigma = 184
pos_lm = 200#50
pos_th = 0
pos_psi = 90

def mkKernel(ks,sig,th,lm,ps):
    # original
    #if not ks%2:exit(1)
    if not ks%2:ks += 1

    phi = th * np.pi/180.
    psi = ps * np.pi/180.
    lmbd = np.float(lm)/100.
    #lmbd = np.float(lm)/10
    sigma = np.float(sig)/100.
    #sigma = np.float(sig)/ks

    # interval
    xs=ys=np.linspace(-1.,1.,ks)
    x,y = np.meshgrid(xs,ys)

    R1 = x*np.cos(phi)+y*np.sin(phi)
    R2 = -x*np.sin(phi)+y*np.cos(phi)

    exy = -0.5/(sigma**2)
    cscale = (2.*np.pi)/lmbd

    scale=1./(2.*np.pi*(sigma**2))
    #b =np.exp(exy*(R1**2+R2**2))
    #c =np.exp(1j*(cscale*R1))
    b =np.exp(-0.5*((R1**2+R2**2)/(sigma**2)))
    c =np.exp(1j*(cscale*R1))

    # paper version
    ker = np.real(np.array(scale*b*c,dtype=np.complex128))

    # this code version
    ##ker=np.array(np.exp(-0.5*(R1**2+R2**2)/sigma**2)*np.cos(2.*np.pi*R1/lmbd + psi),dtype=np.float32)
    ##ker=np.array(b*c,dtype=np.float32)
    #ker=np.array(b*np.cos(cscale*R1+psi),dtype=np.float32)

    print 'sig:%.4f,phi:%.4f,lamda:%.4f,ps:%.4f\n'%(sigma,phi,lmbd,psi)

    return ker


def Process():
    sig = pos_sigma
    lm = pos_lm
    th = pos_th
    ps = pos_psi
    ks = pos_ks

    kernel = mkKernel(ks,sig,th,lm,ps)

    kernelimg = kernel/2.+0.5
    global src_f
    dest = cv.filter2D(src_f, cv.CV_32F,kernel)

    cv.namedWindow('Process window',cv.cv.CV_WINDOW_NORMAL)
    cv.namedWindow('Kernel',cv.cv.CV_WINDOW_NORMAL)
    cv.namedWindow('Mag',cv.cv.CV_WINDOW_NORMAL)
    cv.imshow('Process window', dest)
    cv.imshow('Kernel', kernel)
    cv.imshow('Kerneling', cv.resize(kernelimg, (pos_ks*20,pos_ks*20)))
    #cv.imshow('Kernel', kernelimg)
    cv.imshow('Mag', np.power(dest,2))

def cb_sigma(pos):
    global pos_sigma
    pos_sigma = pos
    """
    if pos > 0:
        pos_sigma = pos
    else:
        pos_sigma = 1
    """
    Process()

def cb_ks(pos):
    global pos_ks
    pos_ks = pos
    Process()

def cb_lm(pos):
    global pos_lm
    pos_lm = pos
    Process()

# Phase: theta = phai
def cb_th(pos):
    global pos_th
    pos_th = pos
    Process()

# Psi:
def cb_psi(pos):
    global pos_psi
    pos_psi = pos
    Process()



if __name__ == '__main__':
    dir_input = "..\data\input"
    image = cv.imread(os.path.join(dir_input,'example4_.png'),1)
    image = cv.resize(image, (500, 500))
    cv.imshow('Src',image)
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #global src_f
    src_f = np.array(src, dtype=np.float32)
    src_f /= 255.
    if not pos_ks%2:
        pos_ks += 1

    cv.namedWindow('Process window',1)
    #cv.createTrackbar('Sigma','Process window',pos_sigma,pos_ks,cb_sigma)
    cv.createTrackbar('Sigma','Process window',pos_sigma,1000,cb_sigma)
    cv.createTrackbar('Lambda', 'Process window', pos_lm, 1000, cb_lm)
    cv.createTrackbar('Theta(phi)', 'Process window', pos_th, 180, cb_th)
    cv.createTrackbar('Psi', 'Process window', pos_psi, 360, cb_psi)
    cv.createTrackbar('KernelSize','Process window',pos_ks,200,cb_ks)
    Process()
    cv.waitKey(0)
    cv.destroyAllWindows()
