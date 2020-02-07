#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python

# author : Prakash [प्रकाश]
# date   : 2019-09-28 23:47

from abc import abstractmethod
import copy

import scipy as sp
import scipy.optimize
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from . import plot as mplt
from . import utilities as utl

class Hist (object):
    def __init__(self):
        self.H = None

    @abstractmethod
    def  plot(self,ax,**kw):
        raise NotImplementedError('`plot` method is not implemented.')

    @abstractmethod
    def __make_hist(self,**kw):
        raise NotImplementedError('`make_hist` method is not implemented.')

class Hist1D(Hist):
    def __init__(self,x,bins='auto',range=None,weights=None):
        self.x = x
        self.bins = bins
        self.range = range
        self.weights = weights
        self.centers = None
        self.norm_par = None
        self.binorm_par = None
        self.norm_erf_par = None
        self.H = None

        self.fxmin = None
        self.fxmax = None

        self.__make_hist()

    def __make_hist(self):
        self.H, self.be = np.histogram(self.x,self.bins,self.range)
        self.centers = (self.be[1:]+self.be[:-1])/2
        self.error = np.sqrt(self.H)

    def normalize(self):
        area = np.sum(self.H*np.diff(self.be))
        #area = np.sum(self.H*np.diff(self.be)**2*len(self.x))
        self.error = np.sqrt(self.H)/area
        self.H = self.H/area
        return self

    def  __truediv__(self,factor):
        if isinstance(factor,float) or isinstance(factor,int):
            return self.__mul__(1/factor)
    
    def  __mul__(self,factor):
        cp = copy.deepcopy(self)
        if isinstance(factor,float) or isinstance(factor,int):
            cp.be *= factor
            cp.centers *= factor
            if cp.fxmin:
                cp.fxmin *= factor
                cp.fxmax *= factor
            if cp.norm_par:
                mu,sigma,A = cp.norm_par[0]
                emu,esigma,eA = cp.norm_par[1]
                nparam = np.array([mu*factor,sigma*factor,A])
                ncov = np.array([emu*factor,esigma*factor,eA])
                cp.norm_par = (nparam,ncov)

            if cp.norm_erf_par:
                mu,sigma,A,Ae = cp.norm_erf_par[0]
                emu,esigma,eA,eAe = cp.norm_erf_par[1]
                nparam = np.array([mu*factor,sigma*factor,A,Ae])
                ncov = np.array([emu*factor,esigma*factor,eA,eAe])
                cp.norm_erf_par = (nparam,ncov)

            if cp.binorm_par:
                cp.binorm_par = (factor*cp.binorm_par[0], factor*cp.binorm_par[1] )

            return cp


    def fit_normef(self,p0=None):
        params,cov = (None,None)
        if p0 is None:
            #print(f'Length of cneters an hist are , {len(self.centers)} and {len(self.H)}')
            x,y,mu,sigma,A = utl.get_subset_near_peak(self.H,self.centers,devs=4)
            self.fxmin = min(x); self.fxmax = max(x)
            params,cov = sp.optimize.curve_fit(utl.norm_erf_func,x,y,p0=(mu,.4*sigma,A,A/10))
        else:
            params,cov = sp.optimize.curve_fit(utl.norm_erf_func,self.centers,self.H,p0=p0)

        self.norm_erf_par = (params,cov)
        return self

    def fit_normal(self,p0=None):
        params,cov = None,None
        if p0 is None:
            x,y,mu,sigma,A = utl.get_subset_near_peak(self.H,self.centers,devs=4)
            self.fxmin = min(x); self.fxmax = max(x)
            params,cov = sp.optimize.curve_fit(utl.norm_func,x,y,p0=(mu,0.8*sigma,A))
        else:
            params,cov = sp.optimize.curve_fit(utl.norm_func,self.centers,self.H,p0=p0)

        self.norm_par = (params,cov)

        return self

    def fit_binormal(self,p0=None):
        params,cov = sp.optimize.curve_fit(utl.binorm_func,self.centers,self.H,p0=p0)
        self.binorm_par = (params,cov)
        return self


    def plot(self,ax=None,ebar=True,steps=False,**kw):
        mplt.plot1d(self,ax,ebar,steps,**kw)
        return self

class Hist2D(Hist):
    def __init__(self,XY,bins=20,weights=None):

        self.x = XY[0]
        self.y = XY[1] 
    
        self.bins = bins
        self.weights = weights
        self.biv_params = None

        self.__make_hist()

    def __add__(self,h2):
        h1 = copy.deepcopy(self)
        if h1.H.shape  == h2.H.shape:
            h1.H = h1.H +  h2.H
        else:
            raise ValueError("The histograms have different shapes")
        return h1

    def  __sub__(self,h2):
        h1 = copy.deepcopy(self)
        if h1.H.shape  == h2.H.shape:
            h1.H = h1.H - h2.H

        else:
            raise ValueError("The histograms have different shapes")
        return h1

    def __truediv__(self,factor):
        if isinstance(factor,float) or isinstance(factor,int):
            return self.__mul__(1/factor)
        return self

    def __mul__(self,factor):
        cp = copy.deepcopy(self)

        if isinstance(factor,float) or isinstance(factor,int):
            cp.H *=  factor
        else:
            cp.H /= factor.H
            
        return cp



    def __make_hist(self):
        Hc,self.xe,self.ye = np.histogram2d(self.x,self.y,self.bins)
        #Hc = np.nan_to_num(Hc,nan=1)
        if self.weights is None:
            self.H = Hc
        else:
            H,self.xe,self.ye = np.histogram2d(self.x,self.y,self.bins,weights=self.weights)
            self.H = H/Hc

        
    def plot(self,ax=None,cmin=None,figsize=None,cbar=False,**kw):
        mplt.plot2d(self,ax,figsize,cmin=cmin,cbar=cbar,**kw)
        return self

    def fit_binorm(self,p0=None):
        xmin = np.min(self.x); xmax = np.max(self.x)
        ymin = np.min(self.y); ymax = np.max(self.y)

        xbin = self.bins; ybin = self.bins


        xgrid = np.linspace(xmin,xmax,xbin)
        ygrid = np.linspace(ymin,ymax,xbin)

        XX,YY = np.meshgrid(xgrid,ygrid)

        if isinstance(self.bins,tuple):
            xbin = self.bins[0]; ybin = self.bins[1]

        params, cov = sp.optimize.curve_fit(utl.biv_norm,(XX,YY),self.H.ravel(),p0=p0)
        self.data_fitted = utl.biv_norm((XX,YY),*params).reshape(xbin,ybin)
        self.biv_params = (params,cov)
        self.XX,self.YY = XX,YY

        return self




def hist2d_from_func(func,params,extra_args=None,bins=100):
    a,b = params
    c = extra_args
    tpc_points = np.c_[a,b,c]
    fv = func(tpc_points)

    return Hist2D((a,b),bins=bins,weights=fv)

