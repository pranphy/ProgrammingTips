#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim: ai ts=4 sts=4 et sw=4 ft=python

# author : Prakash [प्रकाश]
# date   : 2019-10-03 11:20

import numpy as np

import scipy as sp

import matplotlib.pyplot as plt

def cerf_func(x,mu,sigma,A):
    z = (x-mu)/(np.sqrt(2)*sigma)
    cerf = A*(1-(sp.special.erf(z)))
    #cerf = Ae*(1-(sp.special.erf(z)))
    return cerf

def norm_erf_func(x,mu,sigma,A,Ae):
    norval = norm_func(x,mu,sigma,A)
    cerf = cerf_func(x,mu,sigma,Ae)

    return norval + cerf

def norm_func(x,mu,sigma,A):
    #func = sigma*np.sqrt(2*np.pi)*A*sp.stats.norm.pdf(x,mu,sigma)
    func =  A*np.exp(-0.5*((x-mu)/sigma)**2)
    return  func

def binorm_func(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return norm_func(x,mu1,sigma1,A1) + norm_func(x,mu2,sigma2,A2)

def get_subset_near_peak(H,centers,devs=2):
    vmax = np.max(H)
    max_idx = np.where(H == vmax)[0]

    guess_mu = centers[max_idx][0]
    half_max = vmax/2

    rev_v = H[::-1]
    half_idx = len(rev_v) - np.where(rev_v > half_max)[0][0] - 1 

    half_loc = centers[half_idx]
    half_wid = np.abs(guess_mu-half_loc)
    #rprint(f'max = {vmax}, half_max = {half_max}, mu={guess_mu} half_idx={half_idx} lenv = {len(H)} half_loc={half_loc}, half_wid={half_wid}')
    sig = half_wid # I think this gives optimal better
    midmask = (centers > guess_mu - devs*sig)*(centers < guess_mu + devs*sig)

    masked_centers = centers[midmask]
    v_masked = H[midmask]

    return masked_centers,v_masked,guess_mu,sig,vmax

def biv_norm(XY, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x,y = XY
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
    return g.ravel()

