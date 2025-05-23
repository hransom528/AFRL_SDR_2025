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
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
global theta
global r
import threading
theta=0
r=1
# Convert polar coords to rectangular coords
def rect2polar(coord, degrees=True):
	# Get coord vals
	x = coord[0]
	y = coord[1]

	# Calculate polar coords
	r = np.sqrt(x**2 + y**2)
	theta = np.arctan2(y, x)

	# Convert to degrees if desired
	if (degrees):
		theta = np.rad2deg(theta)
	
	return (r, theta)

# Convert rectangular coords to polar coords
def polar2rect(coord, degrees=True):
	r = coord[0]
	theta = coord[1]

	# Convert theta to radians if in degrees
	if (degrees):
		theta = np.deg2rad(theta)
	
	# Convert to rectangular coords
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	return (x, y)

# Plot radar coords in polar format (static)
def staticRadarPlot(coord):
	fig = plt.figure()
	ax = fig.add_subplot(projection='polar')

	r = coord[0]
	theta = coord[1]
	c = ax.scatter(np.deg2rad(theta), r)
	plt.show()

# Dynamic radar animation update function
def radarUpdate(frame):
    global theta
    global r
    #theta += np.deg2rad(30)
    ax.clear()
    ax.scatter(np.deg2rad(theta), r)
    fig.canvas.draw()

# MAIN
if __name__ == "__main__":
	# Creates the figure and axes object
	fig = plt.figure()
	ax = fig.add_subplot(projection='polar')

	# Creates the initial coordinate point
	r = 2
	theta = np.deg2rad(60)
	#staticRadarPlot((r, theta))

	# Creates animation object
	anim = FuncAnimation(fig, radarUpdate)
	plt.show()
    
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
#tabor1.send_scpi_cmd(':SOUR:FREQ 5.4e9')
#tabor1.send_scpi_cmd(':SOUR:FREQ:OUTP ON')
tabor2= TEVisaInst('192.168.100.110')
tabor2.default_paranoia_level = 2
tabor1.send_scpi_cmd('*CLS; *RST')
tabor2.send_scpi_cmd('*CLS; *RST')
def connect():
    global tabor1
    global tabor2
    tabor1= TEVisaInst('192.168.100.114')
    tabor1.default_paranoia_level = 2
    tabor1.send_scpi_cmd(':SOUR:FREQ 5.4e9')
    #tabor1.send_scpi_cmd(':SOUR:FREQ:OUTP ON')
    tabor2= TEVisaInst('192.168.100.110')
    tabor2.default_paranoia_level = 2
def digconfig3(inst):
    global framelen
    #inst.send_scpi_cmd('*CLS; *RST')

    #tabor2.send_scpi_cmd(':DIG:FREQ:SOUR EXT')
    inst.send_scpi_cmd(':DIG:MODE DUAL')
    inst.send_scpi_cmd(':DIG:FREQ 2700MHZ')
    
    # Allocate four frames of 4800 samples
    numframes=1
    cmd = ':DIG:ACQuire:FRAM:DEF {0},{1}'.format(numframes, framelen)
    inst.send_scpi_cmd(cmd)
    
    # Select the frames for the capturing 
    # (all the four frames in this example)
    capture_first, capture_count = 1, numframes
    cmd = ":DIG:ACQuire:FRAM:CAPT {0},{1}".format(capture_first, capture_count)
    inst.send_scpi_cmd(cmd)
    
    # Set Trigger level to 0.2V
    inst.send_scpi_cmd(':DIG:TRIG:LEV1 0.1')
    
    # Enable capturing data from channel 1
    inst.send_scpi_cmd(':DIG:CHAN:SEL 1')
    inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')
    # Select the external-trigger as start-capturing trigger:
    inst.send_scpi_cmd(':DIG:TRIG:SOURCE EXT')
    
    # Enable capturing data from channel 2
    inst.send_scpi_cmd(':DIG:CHAN:SEL 2')
    inst.send_scpi_cmd(':DIG:CHAN:STATE ENAB')
    # Select the external-trigger as start-capturing trigger:
    inst.send_scpi_cmd(':DIG:TRIG:SOURCE EXT')
    
    
    # Clean memory 
    inst.send_scpi_cmd(':DIG:ACQ:ZERO:ALL')
    
    resp = inst.send_scpi_query(':SYST:ERR?')
    #print(resp)
    #print("Set Digitizer: DUAL mode; internal Trigger")
def capture(inst,chan):
    sig =[]
    inst.send_scpi_cmd(':DIG:INIT ON')
    time.sleep(.51)
    inst.send_scpi_cmd(':DIG:INIT OFF')
    inst.send_scpi_cmd(':DIG:DATA:SEL ALL')
    inst.send_scpi_cmd(':DIG:DATA:TYPE FRAM')
    resp = inst.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    inst.send_scpi_cmd(':DIG:CHAN:SEL {0}'.format(chan))
    wavlen = num_bytes // 2
    sig = np.zeros(wavlen, dtype=np.uint16)
    inst.read_binary_data(':DIG:DATA:READ?', sig, num_bytes)
    return sig
def captureall():
    global tabor1
    global tabor2
    global sig1
    global sig2
    global sig3
    global sig4
    time.sleep(.01)
    tabor1.send_scpi_cmd(':DIG:INIT ON')
    tabor2.send_scpi_cmd(':DIG:INIT ON')
    #tabor1.send_scpi_cmd(':DIG:TRIG:IMM')
    #tabor2.send_scpi_cmd(':DIG:TRIG:IMM')
    time.sleep(.1)
    tabor1.send_scpi_cmd(':DIG:INIT OFF')
    tabor2.send_scpi_cmd(':DIG:INIT OFF')
def load():
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
    time.sleep(.01)
    
    tabor1.send_scpi_cmd(':DIG:CHAN:SEL 2')
    resp = tabor1.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig2 = np.zeros(wavlen, dtype=np.uint16)
    tabor1.read_binary_data(':DIG:DATA:READ?', sig2, num_bytes)
    time.sleep(.01)
    
    tabor2.send_scpi_cmd(':DIG:CHAN:SEL 1')
    resp = tabor2.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig3 = np.zeros(wavlen, dtype=np.uint16)
    tabor2.read_binary_data(':DIG:DATA:READ?', sig3, num_bytes)
    time.sleep(.01)
    
    tabor2.send_scpi_cmd(':DIG:CHAN:SEL 2')
    resp = tabor2.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    wavlen = num_bytes // 2
    sig4 = np.zeros(wavlen, dtype=np.uint16)
    tabor2.read_binary_data(':DIG:DATA:READ?', sig4, num_bytes)
    #time.sleep(.5)
def load2(inst):
        # Choose which frames to read (all in this example)
    inst.send_scpi_cmd(':DIG:DATA:SEL ALL')
    
    # Choose what to read 
    # (only the frame-data without the header in this example)
    inst.send_scpi_cmd(':DIG:DATA:TYPE FRAM')
    
    # Get the total data size (in bytes)
    resp = inst.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)
    print('Total size in bytes: ' + resp)
    print()
    
    # Read the data that was captured by channel 1:
    inst.send_scpi_cmd(':DIG:CHAN:SEL 1')
    
    wavlen = num_bytes // 2
    
    wav1 = np.zeros(wavlen, dtype=np.uint16)
    
    rc = inst.read_binary_data(':DIG:DATA:READ?', wav1, num_bytes)
    
    # Read the data that was captured by channel 2:
    inst.send_scpi_cmd(':DIG:CHAN:SEL 2')
    
    wavlen = num_bytes // 2
    
    wav2 = np.zeros(wavlen, dtype=np.uint16)
    
    rc = inst.read_binary_data(':DIG:DATA:READ?', wav2, num_bytes)
    
    resp = inst.send_scpi_query(':SYST:ERR?')
    print(resp)
    print("read data from DDR")
def capt(inst):
    inst.send_scpi_cmd(':DIG:DATA:SEL ALL')

    inst.send_scpi_cmd(':DIG:DATA:TYPE FRAM')

    resp = inst.send_scpi_query(':DIG:DATA:SIZE?')
    num_bytes = np.uint32(resp)

    inst.send_scpi_cmd(':DIG:CHAN:SEL 1')
    wavlen = num_bytes // 2
    wav1 = np.zeros(wavlen, dtype=np.uint16)
    rc = inst.read_binary_data(':DIG:DATA:READ?', wav1, num_bytes)

    inst.send_scpi_cmd(':DIG:CHAN:SEL 2')
    wavlen = num_bytes // 2
    wav2 = np.zeros(wavlen, dtype=np.uint16)
    rc = inst.read_binary_data(':DIG:DATA:READ?', wav2, num_bytes)
def scramble(l,t,arr,v):
    ls=[]
    temp = np.array([[]])
    for i in range(l):
        select=np.floor(np.random.rand(1)*len(arr))
        temp=[arr[int(select)][v]]
        for f in range(t-1):
            select=np.floor(np.random.rand(1)*len(arr))
            temp=np.concatenate((temp,[arr[int(select)][v]]),0)
        ls.append(temp)
        temp = np.array([[]])
    return ls
def dice(l,t,arr,v):
    ls=[]
    temp = np.array([[]])
    for i in range(int(len(arr)/t)-1):
        temp=[arr[i*t][v]]
        for time in range(t-1):
            a=[arr[(i)*t+time+1][v]]
            temp = np.concatenate((temp,a),0)
        ls.append(temp) 
    return ls
def dc(sig1,f,sf,c):
    sp=1/sf
    qvec=np.sin(2*np.pi*f*np.linspace(0,sp*len(sig1),len(sig1)))
    ivec=np.cos(2*np.pi*f*np.linspace(0,sp*len(sig1),len(sig1)))
    filt= signal.firwin(50, c, nyq=sf/2)
    out=np.convolve(ivec*sig1,filt)+np.convolve(qvec*sig1,filt)*1j
    return out
def triangulate(angleVector, d1=-0.29, d2=0.126378414):
    n = len(angleVector)
    coords = np.zeros((n, 2))    
    for i in range(1):
        angles = angleVector
        a1, a2 = angles[0], angles[1]
        a1 = 90-a1 # Converts reference frame for angles
        a2 = 90-a2
        if (a1 == a2): # Edge case: Angles are the same
            r = 2.4
            x = 2.4 * np.cos(np.deg2rad(a1))
            y = 2.4 * np.sin(np.deg2rad(a1))
        else:
            m1 = np.tan(np.deg2rad(a1))
            m2 = np.tan(np.deg2rad(a2))
            x = ((m1*d1) - (m2*d2)) / (m1 - m2)
            y = m1*(x - d1)
            if (y < 0): # Keeps estimate in front of array
                x *= -1
                y *= -1
        coord = np.array([x, y])
        coords = coord
    return coords
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# Defines Neural Network structure
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(370, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        vector1, vector2 = x
        #vector = vector.view(vector.size(0), -1)  # Reshape `vector` to be 2D with shape (batch_size, 3)
        vector1[0]=vector1[0]
        vector2[0]=vector2[0]
        x = torch.cat((vector1, vector2), dim=1)
        x = self.flatten(x)

        logits = self.linear_relu_stack(x.to(device))
        return logits

# Antenna spacing calculations (x-axis)
element_spacing=.16655136555555555/2
ant0 = 0-.29
ant1 = element_spacing*2-.29
ant2 = element_spacing*5-.29
ant3 = element_spacing*7-.29
d1 = ant0
d2 = ant2

# Loads trained models
def load_model(checkpoint_path):
    # Loads trained baseband model
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

    checkpoint = torch.load(checkpoint_path, weights_only=True,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model

# Connect to and configure Tabor
#connect()
model=load_model('tiles_nobb10000')
#capture(tabor2,1)
global loc 
loc=1
def wam():
    global loc
    loc =1000
def calc(model, tabor1, tabor2):
    global r
    global theta
    global sig1
    global sig2
    global sig3
    global sig4
    global loc
    loc=0
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
    xt=0
    yt=(.29)
    #yt=yt-xt*np.sin(np.deg2rad(-transform))
    #xt=xt*np.cos(np.deg2rad(-transform))
    pos=[xt,yt]
    loc=2
    arr1=[(-.29)*np.cos(np.deg2rad(transform)),(-.29)*np.sin(np.deg2rad(transform))]
    arr1a=np.rad2deg(np.angle((xt-arr1[0])+(yt-arr1[1])*1j))
    arr2=[(.125)*np.cos(np.deg2rad(transform)),.125*np.sin(np.deg2rad(transform))]
    arr2a=np.rad2deg(np.angle((xt-arr2[0])+(yt-arr2[1])*1j))
    posf=[90+transform-arr1a,90+transform-arr2a]
    print(posf)
    #pos=posf
    loc=3
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
    while True:
        loc=4
        digconfig3(tabor1)
        digconfig3(tabor2)
        captureall()
        load()
    
        loc=5 
    
        signals=[np.array(sig1).astype(float)-2048,np.array(sig2).astype(float)-2048,np.array(sig3).astype(float)-2048,np.array(sig4).astype(float)-2048]
        #c1=(np.correlate(signals[0],signals[1],"full"))
        #c2=(np.correlate(signals[2],signals[3],"full"))
        #td1=(-(np.argmax(c1)-len(signals[0])))
        #td2=(-(np.argmax(c2)-len(signals[0])+2))
        c1=(np.correlate(signals[0],signals[1],"same"))
        c2=(np.correlate(signals[2],signals[3],"same"))
        review=[]
        review2=[]
        est1=[]
        loc=6
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
            sigout1=sigo0+sigo1*np.exp(phase1*1j)#np.roll(signals[1],round(shift))
            sigout2=sigo2+sigo3*np.exp(phase2*1j)#np.roll(signals[3],round(shift))
            #plt.plot(sigout2)
            review.append(np.mean(abs(sigout1)**2)/1e3)
            review2.append(np.mean(abs(sigout2)**2)/1e3)
            #print(angle2)
            #print(np.mean(sigout2**2))
            #plt.show()
            #print(angle1u)
            #print(np.mean(abs(sigout1)**2))
            #print(np.mean(abs(sigout2)**2))
            #print(np.mean(abs(sigout1)**2))
            #print(np.mean(abs(sigout2)**2))
            a0=[[angle1u+90,ant0,ant1,td1,np.mean(abs(sigout1)**2)/1e3],[angle1u+90,ant2,ant3,td2,np.mean(abs(sigout2)**2)/1e3],pos]
            #1=np.concatenate((a1,[np.tan(a1[1]/a1[0]),np.angle(pos[0]+pos[1]*1j, deg=True)]),0)
            est1.append(a0)
            #st2.append(a1)
            loc=7
        #plt.plot(review)
        print("triangulate")
        print(triangulate([np.argmax(review)*5-90,np.argmax(review2)*5-90]))
    
        v1=[]
        v2=[]
        for i in range(37):
            v1.append(est1[i][0])
            v2.append(est1[i][1])
        with torch.no_grad():
            pred = model([torch.tensor([v1]).to(dtype=torch.float32),torch.tensor([v2]).to(dtype=torch.float32)]).cpu().numpy().ravel()
        print("ai")
        print(triangulate(pred))
        #radial=rect2polar(triangulate([np.argmax(review)*5-90,np.argmax(review2)*5-90]))
        radial=rect2polar([triangulate(pred)[0],triangulate(pred)[1]])
        r = radial[0]
        theta = radial[1]
    #radarUpdate(anim)
   # staticRadarPlot(radial)
    #plt.show()
#t1=threading.Thread(target=wam, args=())
#t1.start()
t2=threading.Thread(target=calc, args=(model,tabor1,tabor2))
t2.start()
