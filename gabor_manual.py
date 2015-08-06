# encoding: utf-8
# gabor.py
# 2012-3-8
# Eiichiro Momma
#__author__ = 'momma'

import numpy as np
import cv2 as cv
import os

def mkKernel(ks, sig, th , lm, ps):
    # original
    if not ks%2:
        exit(1)
    """
    hks = ks/2
    theta = th * np.pi/180.
    psi = ps * np.pi/180.
    xs=np.linspace(-1.,1.,ks)
    ys=np.linspace(-1.,1.,ks)
    lmbd = np.float(lm)
    x,y = np.meshgrid(xs,ys)
    #sigma = np.float(sig)
    sigma = np.float(sig)/ks
    x_theta = x*np.cos(theta)+y*np.sin(theta)
    y_theta = -x*np.sin(theta)+y*np.cos(theta)
    ker=np.array(np.exp(-0.5*(x_theta**2+y_theta**2)/sigma**2)*np.cos(2.*np.pi*x_theta/lmbd + psi),dtype=np.float32)
    print 'sig:%.2f,theta:%.2f,lamda:%.1f,ps:%.2f\n'%(sigma,theta,lmbd,psi)
    """

    phi=th * np.pi/180.
    sigma = np.float(sig)/100.
    lmbd = np.float(lm)

    # interval
    #xs=ys=np.linspace(-ks/2.,ks/2.,ks)
    xs=ys=np.linspace(-1.,1.,ks)
    x,y = np.meshgrid(xs,ys)

    R1 = x*np.cos(phi)+y*np.sin(phi)
    R2 =-x*np.sin(phi)+y*np.cos(phi)

    exy = -0.5/(sigma**2)
    cscale = 2.*np.pi/lmbd

    scale=1./(2*np.pi*(sigma**2))
    b =np.exp(exy*(R1**2+R2**2))
    c =np.exp(1j*(cscale*R1))
    ker = np.real(np.array(scale*b*c,dtype=np.complex128))

    print 'sig:%.2f,phi:%.2f,lamda:%.1f\n'%(sigma,phi,lmbd)

    return ker


src_f = 0

kernel_size =7
pos_sigma = 5
pos_lm = 1#50
pos_th = 0
pos_psi = 90

def Process():
    sig = pos_sigma
    #lm = 0.5+pos_lm/100.
    lm = pos_lm/10.
    lm = pos_lm
    th = pos_th
    ps = pos_psi
    kernel = mkKernel(kernel_size, sig, th, lm, ps )
    kernelimg = kernel/2.+0.5
    global src_f
    dest = cv.filter2D(src_f, cv.CV_32F,kernel)
    cv.namedWindow('Process window',cv.cv.CV_WINDOW_NORMAL)
    cv.namedWindow('Kernel',cv.cv.CV_WINDOW_NORMAL)
    cv.namedWindow('Mag',cv.cv.CV_WINDOW_NORMAL)
    cv.imshow('Process window', dest)
    cv.imshow('Kernel', cv.resize(kernelimg, (kernel_size*20,kernel_size*20)))
    #cv.imshow('Kernel', kernelimg)
    cv.imshow('Mag', np.power(dest,2))

def cb_sigma(pos):
    global pos_sigma
    if pos > 0:
        pos_sigma = pos
    else:
        pos_sigma = 1
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

    image = cv.imread(os.path.join(dir_input,'example3.png'),1)
    image = cv.resize(image, (500, 500))
    cv.imshow('Src',image)
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #global src_f
    src_f = np.array(src, dtype=np.float32)
    src_f /= 255.
    if not kernel_size%2:
        kernel_size += 1

    cv.namedWindow('Process window',1)
    #cv.createTrackbar('Sigma','Process window',pos_sigma,kernel_size,cb_sigma)
    cv.createTrackbar('Sigma','Process window',pos_sigma,1000,cb_sigma)
    cv.createTrackbar('Lambda', 'Process window', pos_lm, 100, cb_lm)
    cv.createTrackbar('Theta(phi)', 'Process window', pos_th, 180, cb_th)
    cv.createTrackbar('Psi', 'Process window', pos_psi, 360, cb_psi)
    Process()
    cv.waitKey(0)
    cv.destroyAllWindows()
