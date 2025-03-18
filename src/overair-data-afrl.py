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
global tabor1
global tabor2
tabor1= TEVisaInst('192.168.100.114')
tabor1.default_paranoia_level = 2

tabor2= TEVisaInst('192.168.100.110')
tabor2.default_paranoia_level = 2
tabor1.send_scpi_cmd('*CLS; *RST')
tabor2.send_scpi_cmd('*CLS; *RST')
filename="test"
def connect(): #connect to devices
    global tabor1
    global tabor2
    tabor1= TEVisaInst('192.168.100.114')
    tabor1.default_paranoia_level = 2
    tabor1.send_scpi_cmd(':SOUR:FREQ 5.4e9')
    tabor2= TEVisaInst('192.168.100.110')
    tabor2.default_paranoia_level = 2
def digconfig3(inst): #configure adc
    global framelen
    inst.send_scpi_cmd(':DIG:MODE DUAL')
    inst.send_scpi_cmd(':DIG:FREQ 2700MHZ')
    numframes=1
    cmd = ':DIG:ACQuire:FRAM:DEF {0},{1}'.format(numframes, framelen)
    inst.send_scpi_cmd(cmd)
    capture_first, capture_count = 1, numframes
    cmd = ":DIG:ACQuire:FRAM:CAPT {0},{1}".format(capture_first, capture_count)
    inst.send_scpi_cmd(cmd)
    inst.send_scpi_cmd(':DIG:TRIG:LEV1 0.1')
    inst.send_scpi_cmd(':DIG:CHAN:SEL 1')
    inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')
    inst.send_scpi_cmd(':DIG:TRIG:SOURCE EXT')
    inst.send_scpi_cmd(':DIG:CHAN:SEL 2')
    inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')
    inst.send_scpi_cmd(':DIG:TRIG:SOURCE EXT')
    inst.send_scpi_cmd(':DIG:ACQ:ZERO:ALL')
    resp = inst.send_scpi_query(':SYST:ERR?')
def captureall(): # wait for trigger to capture data
    global tabor1
    global tabor2
    global sig1
    global sig2
    global sig3
    global sig4
    time.sleep(.1)
    tabor1.send_scpi_cmd(':DIG:INIT ON')
    tabor2.send_scpi_cmd(':DIG:INIT ON')
    time.sleep(.6)
    tabor1.send_scpi_cmd(':DIG:INIT OFF')
    tabor2.send_scpi_cmd(':DIG:INIT OFF')
def load(): # load data from sdr units
    global tabor1
    global tabor2
    global sig1
    global sig2
    global sig3
    global sig4
    tabor1.send_scpi_cmd(':DIG:DATA:SEL ALL')
    tabor1.send_scpi_cmd(':DIG:DATA:TYPE FRAM')
    tabor2.send_scpi_cmd(':DIG:DATA:SEL ALL')
    tabor2.send_scpi_cmd(':DIG:DATA:TYPE FRAM')

    tabor1.send_scpi_cmd(':DIG:CHAN:SEL 1')
    resp = tabor1.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig1 = np.zeros(wavlen, dtype=np.uint16)
    rc=tabor1.read_binary_data(':DIG:DATA:READ?', sig1, num_bytes)
    time.sleep(.1)
    
    tabor1.send_scpi_cmd(':DIG:CHAN:SEL 2')
    resp = tabor1.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig2 = np.zeros(wavlen, dtype=np.uint16)
    tabor1.read_binary_data(':DIG:DATA:READ?', sig2, num_bytes)
    time.sleep(.1)
    
    tabor2.send_scpi_cmd(':DIG:CHAN:SEL 1')
    resp = tabor2.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig3 = np.zeros(wavlen, dtype=np.uint16)
    tabor2.read_binary_data(':DIG:DATA:READ?', sig3, num_bytes)
    time.sleep(.1)
    
    tabor2.send_scpi_cmd(':DIG:CHAN:SEL 2')
    resp = tabor2.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig4 = np.zeros(wavlen, dtype=np.uint16)
    tabor2.read_binary_data(':DIG:DATA:READ?', sig4, num_bytes)
    #time.sleep(.5)
def dice(l,t,arr,v): #repackage captures across the span of collected angles together to represent the scan
    ls=[]
    temp = np.array([[]])
    for i in range(int(len(arr)/t)-1):
        temp=[arr[i*t][v]]
        for time in range(t-1):
            a=[arr[(i)*t+time+1][v]]
            temp = np.concatenate((temp,a),0)
        ls.append(temp) 
    return ls
def dc(sig1,f,sf,c): # downconvert signal
    qvec=np.sin(2*np.pi*f*np.linspace(0,sp*len(sig1),len(sig1)))
    ivec=np.cos(2*np.pi*f*np.linspace(0,sp*len(sig1),len(sig1)))
    filt= signal.firwin(100, c, nyq=sf/2)
    out=np.convolve(ivec*sig1,filt)+np.convolve(qvec*sig1,filt)*1j
    return out

c=299792458 # Speed of light
signal_sources = 3
element_spacing=.16655136555555555/2
positions=np.linspace(0,signal_sources-1,signal_sources)*element_spacing
sf=2.7e9 # Sample frequency
sp=1/sf # Sample Period
u=0
est1=[]
est2=[]
transform=60#rotation of the antenna array
xt=0#target x pos
yt=(2.5)# target y pos
pos=[xt,yt]
arr1=[(-.29)*np.cos(np.deg2rad(transform)),(-.29)*np.sin(np.deg2rad(transform))]
arr1a=np.rad2deg(np.angle((xt-arr1[0])+(yt-arr1[1])*1j))#angle from antenna array 1 lead element
arr2=[(.125)*np.cos(np.deg2rad(transform)),.125*np.sin(np.deg2rad(transform))]
arr2a=np.rad2deg(np.angle((xt-arr2[0])+(yt-arr2[1])*1j))#angle from antenna array 2 lead element
posf=[90+transform-arr1a,90+transform-arr2a]
print(" angle relative to antenna 1: {0} angle relative to antenna 2 {1}".format(posf[0],posf[1]))
#pos=posf
elem=[1,3,2,4]
angle=0
d=.166
c=299_792_458
f=900e6
times=10
ant0 = 0-.29
ant1 = element_spacing*2-.29
ant2 = element_spacing*5-.29
ant3 = element_spacing*7-.29
for test in range(times):
    digconfig3(tabor1)
    digconfig3(tabor2)
    captureall()
    load()

    

    signals=[np.array(sig1).astype(float)-2048,np.array(sig2).astype(float)-2048,np.array(sig3).astype(float)-2048,np.array(sig4).astype(float)-2048]
    c1=(np.correlate(signals[0],signals[1],"same"))
    c2=(np.correlate(signals[2],signals[3],"same"))
    review=[]
    for angle1 in range(37):

        center0 = len(signals[0]) / 2 # Gets the "center" time point of the signal
        td1=(np.argmax(c1)-center0) # Get the difference between the center and max correlation samples
        td2=(np.argmax(c2)-center0)
        td1 = (td1)*sp
        td2 = (td2)*sp
        angle1u=(angle1*5)-90 #shift beamed angle
        # calculate phase and time shifts
        phase1=(np.sin(2*np.pi*(angle1u)/360)*d/c)*f*2*np.pi
        shift = sf*phase1/(f*2*np.pi)
        phase2=(np.sin(2*np.pi*(angle1u)/360)*d/c)*f*2*np.pi
        shift2 = sf*phase2/(f*2*np.pi)
        # downconvert
        sigo0=dc(signals[0],f,sf,450e6)
        sigo1=dc(signals[1],f,sf,450e6)
        sigo2=dc(signals[2],f,sf,450e6)
        sigo3=dc(signals[3],f,sf,450e6)
        #apply phase shift
        sigout1=sigo0+sigo1*np.exp(phase1*1j)
        sigout2=sigo2+sigo3*np.exp(phase2*1j)
        review.append(np.mean(abs(sigout1)**2)/1e3)
        #store data
        a0=[[angle1u+90,ant0,ant1,td1,np.mean(abs(sigout1)**2)/1e3],[angle1u+90,ant2,ant3,td2,np.mean(abs(sigout2)**2)/1e3],pos]
        est1.append(a0)
    plt.plot(review)
    plt.show()



times=times-1
c1=dice(times,37,est1,0)
c2=dice(times,37,est1,1)
np.savez("{0}-{1}-{2}-{3}.npz".format(filename,posf[0],posf[1],transform), data=np.array([c1,c2,list(np.reshape(np.tile(posf,(1,times)),(times,2)))],dtype=object),allow_pickle=True)
print("training samples collected")


times =times+1
est1=[]
est2=[]
for test in range(times):
    digconfig3(tabor1)
    digconfig3(tabor2)
    captureall()
    load()

    

    signals=[np.array(sig1).astype(float)-2048,np.array(sig2).astype(float)-2048,np.array(sig3).astype(float)-2048,np.array(sig4).astype(float)-2048]
    c1=(np.correlate(signals[0],signals[1],"same"))
    c2=(np.correlate(signals[2],signals[3],"same"))
    review=[]
    for angle1 in range(37):

        center0 = len(signals[0]) / 2 # Gets the "center" time point of the signal
        td1=(np.argmax(c1)-center0) # Get the difference between the center and max correlation samples
        td2=(np.argmax(c2)-center0)
        td1 = (td1)*sp
        td2 = (td2)*sp
        angle1u=(angle1*5)-90
        # calculate phase and time shifts
        phase1=(np.sin(2*np.pi*(angle1u)/360)*d/c)*f*2*np.pi
        shift = sf*phase1/(f*2*np.pi)
        phase2=(np.sin(2*np.pi*(angle1u)/360)*d/c)*f*2*np.pi
        shift2 = sf*phase2/(f*2*np.pi)
        # downconvert
        sigo0=dc(signals[0],f,sf,450e6)
        sigo1=dc(signals[1],f,sf,450e6)
        sigo2=dc(signals[2],f,sf,450e6)
        sigo3=dc(signals[3],f,sf,450e6)
        #apply phase shift
        sigout1=sigo0+sigo1*np.exp(phase1*1j)
        sigout2=sigo2+sigo3*np.exp(phase2*1j)
        review.append(np.mean(abs(sigout2)**2)/1e3)
        #store data
        a0=[[angle1u+90,ant0,ant1,td1,np.mean(abs(sigout1)**2)/1e3],[angle1u+90,ant2,ant3,td2,np.mean(abs(sigout2)**2)/1e3],pos]
        est1.append(a0)
    plt.plot(review)
    plt.show()



times=times-1
#package data
c1=dice(times,37,est1,0)
c2=dice(times,37,est1,1)
#save data
np.savez("{0}-{1}-{2}-{3}.npz".format(filename,posf[0],posf[1],transform), data=np.array([c1,c2,list(np.reshape(np.tile(posf,(1,times)),(times,2)))],dtype=object),allow_pickle=True)
print("testing samples  collected")

tabor1.close_instrument()
tabor2.close_instrument()