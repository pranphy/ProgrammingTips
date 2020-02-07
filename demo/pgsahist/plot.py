#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python

# author : Prakash [प्रकाश]
# date   : 2019-10-03 11:11

import numpy as np

import matplotlib.pyplot as plt

#from . import hist as mh
from . import utilities as utl

def plot1d(hist,ax=None,ebar=True,steps=False,filled=False,title=None,xlabel=None,ylabel=None,**kw):
    #if not isinstance(hist,mh.Hist1D):
    #    raise ValueError('Unexpected object type '.format(type(hist)) )
    #else:
    if True:
        if ax is None:
            fig,ax = plt.subplots()

        if ebar:
            __ = ax.errorbar(hist.centers,hist.H,yerr=hist.error,fmt='. ',**kw)
        if steps or filled:
            X = np.array([hist.be[:-1],hist.be[1:]]).T.flatten()
            #X = np.array([hist.be,hist.be]).T.flatten()
            #Y = np.hstack([0,np.array([hist.H,hist.H]).T.flatten(),0])
            Y = np.array([hist.H,hist.H]).T.flatten()
            ax.plot(X,Y)   #ax.bar(hist.be[:-1],hist.H,width=np.diff(hist.be),align='edge',fill=False,**kw)
            #hp = np.hstack([hist.H[0],hist.H])
            zeros = np.zeros(len(Y))
            p = ax.plot(X,Y,**kw)
            if filled:
                #ax.fill(X,Y,alpha=0.2)
                ax.fill_between(X,zeros,Y,color=p[0].get_color(),alpha=0.5)
            #ax.fill(hist.be,hp,ds='steps')

        if hist.norm_par is not None:
            fit_func = utl.norm_func

            params  = hist.norm_par[0]
            mu,sigma,A = params[0], params[1],params[2]

            xmin = min(hist.x); xmax = max(hist.x)
            if hist.range is not None:
                xmin = hist.range[0]; xmax = hist.range[1]
            xmin,xmax = hist.fxmin or xmin, hist.fxmax or xmax

            xvals = np.linspace(xmin,xmax,200)
            yvals = fit_func(xvals,mu,sigma,A)
            ax.plot(xvals,yvals,label='$\mu={:4.2f}$  $\sigma={:4.2f}$ $\sigma/\mu={:3.2f}$%'.format(mu,sigma,sigma/mu*100))
            ax.legend()
        
        elif hist.norm_erf_par is not None:
            fit_func = utl.norm_erf_func
            params = hist.norm_erf_par[0]
            mu,sigma,A,Ae = params[0], params[1],params[2],params[3]
            #print(f'mu = {mu}, sigma={sigma}, A = {A}, Ae={Ae}')

            xmin = min(hist.x); xmax = max(hist.x)
            if hist.range is not None:
                xmin = hist.range[0]; xmax = hist.range[1]
            xmin,xmax = (hist.fxmin or xmin), (hist.fxmax or xmax)

            xvals = np.linspace(xmin,xmax,200)

            ynorm = utl.norm_func(xvals,mu,sigma,A)
            erf = utl.cerf_func(xvals,mu,sigma,Ae)

            details = None

            if details:
                
                ax.plot(xvals,ynorm,'--',label='norm')
                ax.plot(xvals,erf,'-.',label='cerf')
                ax.plot(xvals,erf+ynorm,label='sum')
            else:
                labtext = '$\mu = {:.2f}$ $\sigma={:.3f}$ $\sigma/\mu = {:.3f}$%'.format(mu,sigma,sigma/mu*100)
                ax.plot(xvals,ynorm+erf,'--',label=labtext,alpha=.5)
                ax.legend()


        if hist.binorm_par:
            params  = hist.binorm_par[0]
            mu1,sigma1,A1,mu2,sigma2,A2 = params[0], params[1],params[2],params[3],params[4],params[5]

            xmin = min(hist.x); xmax = max(hist.x)
            if hist.range is not None:
                xmin = hist.range[0]; xmax = hist.range[1]
            xvals = np.linspace(xmin,xmax,200)
            yvals = utl.binorm_func(xvals,mu1,sigma1,A2,mu2,sigma2,A2)
            #labtext  = '$\mu_1={mu1:.2f}$, $\sigma_1={sigma1:.2f}$, $\sigma_1/\mu_1={sigma1/mu1*100:.2f}$%\n $\mu_2={mu2:.2f}$, $\sigma_2={sigma2:.2f}$, $\sigma_2/\mu_2={sigma2/mu2*100:.2f}$%'
            labtext  = ('$\mu_1={:.2f}$,' 
                        '$\sigma_1={:.2f}$,' 
                        '$\sigma_1/\mu_1={:.2f}$%\n '
                        '$\mu_2={:.2f}$, '
                        '$\sigma_2={:.2f}$, '
                        '$\sigma_2/\mu_2={:.2f}$%'
                ).format(mu1,sigma1,sigma1/mu1*100,mu2,sigma2,sigma2/mu2*100) 
                    
            ax.plot(xvals,yvals,label=labtext)
            ax.legend()
        if title is not None:  ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='both',style='sci',scilimits=(-3,3))

def plot2d(hist,ax=None,figsize=None,cmin=1,title=None,xlabel=None,ylabel=None,cbar=False,cbarlabel=None,aspect=None,cmap=None,vmin=None,vmax=None,**kw):
    #if not isinstance(hist,mh.Hist2D):
    #    raise ValueError('Unrecognized paramter of type '.format(type(hist)))
    #else:
    if True:
        if ax is None:
            fig,ax = plt.subplots(figsize=figsize)

        if cmin is not None:
            hist.H[hist.H < cmin] = np.nan

        ext = [min(hist.xe),max(hist.xe),min(hist.ye),max(hist.ye)]
        if aspect is None:
            aspect = ((max(hist.ye)-min(hist.ye))/(max(hist.xe)-min(hist.xe)))

        im = ax.imshow(hist.H.T,origin='bottom',extent=ext,aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax)

        if hist.biv_params is not None:
            ax.contour(hist.XX,hist.YY,hist.data_fitted.T,levels=2,colors='r',alpha=0.8)


        if cbar or cbarlabel is not None:
            label = cbarlabel if cbarlabel is not  None else ""
            ax.figure.colorbar(im,ax=ax,label=label)

    
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis='both',style='sci',scilimits=(-3,3))

    return im

