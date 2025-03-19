# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 11:08:33 2025

@author: exx
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:05:59 2025

@author: sbjoh
"""


import os
import sys
import math
import pyvisa
srcpath = os.path.realpath('SourceFiles')  
sys.path.append(srcpath)
import warnings # this is for GUI warnings
warnings.filterwarnings("ignore")
from tevisainst import TEVisaInst
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
#import keyboard
import time
#signal processing
from scipy.signal import butter,filtfilt
from scipy import signal
import commpy.filters as FIL
import threading
from scipy.optimize import fsolve
from scipy.optimize import minimize
from math import sqrt
from scipy.optimize import least_squares
dsf=2.7e9
#tabor1=0
#tabor2=0
framelen = 48000
sig1=[]
sig2=[]
sig3=[]
sig4=[]


def dc(sig1,f,sf,c):
    qvec=np.sin(2*np.pi*f*np.linspace(0,sp*len(sig1),len(sig1)))
    ivec=np.cos(2*np.pi*f*np.linspace(0,sp*len(sig1),len(sig1)))
    filt= signal.firwin(100, c, nyq=sf/2)
    out=np.convolve(ivec*sig1,filt)+np.convolve(qvec*sig1,filt)*1j
    return out
def dice(l,t,arr,v):
    ls=[]
    temp = np.array([[]])
    for i in range(int(len(arr)/t)):
        temp=[arr[i*t][v]]
        for time in range(t-1):
            a=[arr[(i)*t+time+1][v]]
            temp = np.concatenate((temp,a),0)
        ls.append(temp) 
    return ls
# Connect to and configure Tabor
#connect()

#capture(tabor2,1)
c=299792458 # Speed of light
signal_sources = 3
element_spacing=.16655136555555555/2
positions=np.linspace(0,signal_sources-1,signal_sources)*element_spacing
sf=2.7e9 # Sample frequency
sp=1/sf # Sample Period
u=0
est1=[]
est2=[]
transform=0

#pos=posf
elem=[1,3,2,4]
angle=0
d=.166
c=299_792_458
f=900e6
times=50000
siglen=10000
resfactor=10
A=2048
ant0 = 0-.29
ant1 = element_spacing*2-.29
ant2 = element_spacing*5-.29
ant3 = element_spacing*7-.29
posfarray=[]
beta = 1


for test in range(times):
    sps=np.random.randint(0, 17,1)[0]+60
    num_symbols=np.ceil(siglen/sps)
    rrc=FIL.rrcosfilter(7*sps,beta,sps,1)# filter
    h=rrc[1]
    bits = np.random.randint(0, 2, int(num_symbols)) # Our data to be transmitted, 1's and 0's
    bb =(bits*2-1) # set the first value to either a 1 or -1
    holder=np.zeros(int(sps*num_symbols))
    holder[::sps]=bb
    bb=np.convolve(holder,h)
    bb=bb[0:siglen]
    A=(400+(2048-400)*np.random.rand())*1.5+1.5
    randphase=np.random.rand()*2*np.pi
    parentsig=A*np.sin(2*np.pi*f*np.linspace(0,sp*(1/resfactor)*siglen,siglen)+randphase)+2048#*bb
    xt=(np.random.rand()-.5)*20
    yt=(np.random.rand())*10
    #yt=yt-xt*np.sin(np.deg2rad(-transform))
    #xt=xt*np.cos(np.deg2rad(-transform))
    pos=[xt,yt]
    
    arr1=[(-.29)*np.cos(np.deg2rad(transform)),(-.29)*np.sin(np.deg2rad(transform))]
    arr1a=np.rad2deg(np.angle((xt-arr1[0])+(yt-arr1[1])*1j))
    arr2=[(.125)*np.cos(np.deg2rad(transform)),.125*np.sin(np.deg2rad(transform))]
    arr2a=np.rad2deg(np.angle((xt-arr2[0])+(yt-arr2[1])*1j))
    posf=[90+transform-arr1a,90+transform-arr2a]
    #print(posf)

    A0TD=((ant0-xt)**2+(yt)**2)**.5/c
    A1TD=((ant1-xt)**2+(yt)**2)**.5/c
    A2TD=((ant2-xt)**2+(yt)**2)**.5/c
    A3TD=((ant3-xt)**2+(yt)**2)**.5/c
    sig1=np.roll(parentsig,round((A0TD/(sp/resfactor))))
    sig2=np.roll(parentsig,round((A1TD/(sp/resfactor))))
    sig3=np.roll(parentsig,round((A2TD/(sp/resfactor))))
    sig4=np.roll(parentsig,round((A3TD/(sp/resfactor))))
    signals=[np.array(sig1[::resfactor]).astype(float),np.array(sig2[::resfactor]).astype(float),np.array(sig3[::resfactor]).astype(float),np.array(sig4[::resfactor]).astype(float)]
    c1=(np.correlate(signals[0],signals[1],"same"))
    c2=(np.correlate(signals[2],signals[3],"same"))
    review=[]
    for angle1 in range(37):

        center0 = len(signals[0]) / 2 # Gets the "center" time point of the signal
        td1=(np.argmax(c1)-center0) # Get the difference between the center and max correlation samples
        td2=(np.argmax(c2)-center0)
        #td1=td1+td1/abs(td1)
        #td2=td2+td2/abs(td2)
        
        # Convert sample differences to time differences
        td1 = (td1)*sp
        td2 = (td2)*sp
        #print(td1*c)
        #print(td2*c)
        
        # Antenna coordinates for ULA setup

        angle1u=(angle1*5)-90
        #angle2u=(angle2*10)-90
        phase1=(np.sin(2*np.pi*(angle1u)/360)*d/c)*f*2*np.pi 
        shift = sf*phase1/(f*2*np.pi)
        phase2=(np.sin(2*np.pi*(angle1u)/360)*d/c)*f*2*np.pi
        shift2 = sf*phase2/(f*2*np.pi)
        sigo0=dc(signals[0],f,sf,450e6)
        sigo1=dc(signals[1],f,sf,450e6)
        sigo2=dc(signals[2],f,sf,450e6)
        sigo3=dc(signals[3],f,sf,450e6)
        sigout1=sigo0+sigo1*np.exp(phase1*1j)
        sigout2=sigo2+sigo3*np.exp(phase2*1j)
        #plt.plot(sigout2)
        #print(angle2)
        #print(np.mean(sigout2**2))
        #plt.show()
        #print(angle1u)
        #review.append(np.mean(abs(sigout2)**2))
        #print(np.mean(abs(sigout1)**2))
        #print(np.mean(abs(sigout2)**2))
        #print(np.mean(abs(sigout1)**2))
        #  print(np.mean(abs(sigout2)**2))
        a0=[[angle1u+90,ant0,ant1,td1,np.mean(abs(sigout1)**2)/1e3],[angle1u+90,ant2,ant3,td2,np.mean(abs(sigout2)**2)/1e3],pos]
        #1=np.concatenate((a1,[np.tan(a1[1]/a1[0]),np.angle(pos[0]+pos[1]*1j, deg=True)]),0)
        est1.append(a0)
        #st2.append(a1)
    #plt.plot(review)
    #print(np.argmax(review)*5-90)
    #plt.show()
    #print(posf)
    posfarray.append(posf)
print("no")
#c1=scramble(times,10,est1,0)
#c2=scramble(times,10,est1,1)

c1=dice(times,37,est1,0)
c2=dice(times,37,est1,1)
np.savez("airsimBB1.npz", data=np.array([c1,c2,list(np.reshape(posfarray,[times,2]))],dtype=object),allow_pickle=True)
#np.savez("airt{0}-{1}.npz".format(xt,yt), data=np.array([c1,c2,list(np.reshape(np.repeat(posf,times),(times,2)))],dtype=object),allow_pickle=True)
print("no")
#c1=scramble(times,10,est1,0)
#c2=scramble(times,10,est1,1)
#np.savez("d{0}-{1}tca.npz".format(xt,yt), data=np.array([c1,c2,list(np.reshape(np.tile(posf,(1,times)),(times,2)))],dtype=object),allow_pickle=True)

#p.savez("tests2.npz", data=np.array(est2, dtype=object))
