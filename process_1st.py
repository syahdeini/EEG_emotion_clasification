'''
    Preprocessing EEG data
    1. Read BDF data
    2. band pass filter (4-45 Hertz)
    3. getting EPOCHS (+baseline correction)
    4. doing ICA
    5. save data as pickle

    P.S : 1-22 data from tweente and the rest from geneva (different order)
          for the sake of simplicity, I will only use tweente data

'''
import mne
import csv
from mne.preprocessing import ICA
import pywt
import math
import numpy as np
from collections import Counter
import scipy.io as sc_io
import pickle


def order(trial):
# Ordering channel from geneva participant to tweente's order
# all the order can be seen on http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
# I don't use this funct at the moment
    dataNew=[]
    trial[2],trial[3]=trial[3],trial[2]
    trial[4],trial[5]=trial[5],trial[4]
    trial[6],trial[7]=trial[7],trial[6]
    trial[8],trial[9]=trial[9],trial[8]
    trial[10],trial[11]=trial[11],trial[10]
    trial[12],trial[15]=trial[15],trial[12] 
    trial[13],trial[15]=trial[15],trial[15]
    trial[14],trial[15]=trial[15],trial[14] 
    trial[16],trial[31]=trial[31],trial[16] 
    trial[17],trial[30]=trial[30],trial[17] 
    trial[18],trial[28]=trial[28],trial[18] 
    trial[19],trial[29]=trial[29],trial[19]
    trial[20],trial[26]=trial[26],trial[20] 
    trial[21],trial[27]=trial[27],trial[21] 
    trial[22],trial[24]=trial[24],trial[22] 
    trial[23],trial[25]=trial[25],trial[23] 
    trial[24],trial[27]=trial[27],trial[24] 
    trial[25],trial[27]=trial[24],trial[27] 
    trial[26], trial[29]= trial[29],trial[26] 
    trial[27],trial[29]=trial[29],trial[27] 
    trial[28],trial[30]=trial[30],trial[28]
    trial[29],trial[31]=trial[31],trial[29] 
    return trial

# 1. Reading bdf data
path="/media/syahdeini/961C8FA51C8F7ECD/bdf_data/s"
for i in range(1,23):
    print ">> Reading data "
    if i<10: # filename problem
        raw = mne.io.read_raw_edf(path+"0"+str(i)+".bdf",preload=True)
    else:
        raw = mne.io.read_raw_edf(path+str(i)+".bdf",preload=True)

    # exclude all unused signals
    exclude=["EXG1","EXG2","EXG3","EXG4","EXG5","EXG6","EXG7","EXG8","GSR1","GSR2",
             "Erg1","Erg2","Resp","Plet","Temp","Status"]

    var_picks=mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False,misc=False,resp=False,
                         exclude=exclude)
# 2. Band-pass filter 4-45 Hertz
    raw.filter(4,45)
    print ">> finding events"
    # extract events from data
    events = mne.find_events(raw, stim_channel='STI 014')
    # extract epoch from events
    # event no.3 used (fixation screen before trial begin), information can be read at DEAP web
    event_id = 3   
    tmin = 0
    tmax = 65  
    baseline = (0, 5) # 5 seecond baseline, before the trial begin
    print ">> finding epochs"
# 3. Getting EPOCHS from data, using status code (event id) no.3 
    epochs = mne.Epochs(raw,events, event_id, tmin, tmax, picks=var_picks,baseline= baseline,
                        add_eeg_ref= True,decim=4,preload=True)

# 4. ICA
    ica=ICA(n_components=32)
    ica=ica.fit(epochs,picks=var_picks)
    # getting data as list    
    datas=ica.get_sources(epochs)
    datas=datas.get_data()
# 5. saving data as pickle
    pickle.dump(datas,open("newBasedDEAP_"+str(i)+".p","wb"))




