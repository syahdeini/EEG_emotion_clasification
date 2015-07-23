'''
    Feature extraction using wavelet 
    1. read data per-file
    ---on each channel for every trial inside a file-----------------
    2. doing Stationary Wavelet Transform (SWT) 
       using lv.5
    3. extract fitur from all details
    4. extend 5 fiturs in a list 
    ------------------------------------------------------------------
    5. extend all lists of fitur on all channel (in one Trial)
    6. ordering data based on trial number in metada file from DEAP web
    7. saving data to CSV

P.S: formula for functions on feature extraction based on paper 
"EEG-Based Emotion Recognition Using Frequency Domain Features and Support Vector Machines"

'''


import mne
import csv
from mne.preprocessing import ICA
import pywt
import math
import numpy as np
from collections import Counter
import pickle
from scipy.fftpack import fft
import matplotlib.pyplot as plt
# reading data


# function for reading csv file
def csv_reader(filename):
    with open(filename,'rb') as file_obj:
        reader=csv.reader(file_obj,delimiter=',')
        data=[]
        for row in reader:
            data.append(row)
        return data
        ret_List.append(H)
    return ret_List


## Feature extraction functions #####

def average(signal):
    return float(sum(signal))/len(signal)

def standart_dev(signal):
    Sum=0
    miu=average(signal)
    for idx,s in enumerate(signal):
        Sum=Sum+(s-miu)*(s-miu)
    Sum=math.sqrt((float(1)/(len(signal)-1))*Sum)
    return Sum

def mean_of_absolute(signal):
    sigma=0
    for idx,s in enumerate(signal[:-1]):
        sigma=sigma+abs(signal[idx+1]-s)
    sigma=sigma*(float(1)/(len(signal)-1))
    return sigma

def mean_of_absolute_2nd_dif(signal):
    sigma=0
    for idx,s in enumerate(signal[:-2]):
        sigma=sigma+abs(signal[idx+2]-s)
    sigma=sigma*(float(1)/(len(signal)-2))
    return sigma

def mean_of_absolute_value_1st_normalize(signal):
    return float(mean_of_absolute(signal))/standart_dev(signal)

def means_of_absolte_2st_normalize(signal):
    return float(mean_of_absolute_2nd_dif(signal))/standart_dev(signal)


def signal_power(signal):
    print 'powering signal'
    return np.abs(np.fft.fft(signal))**2

def cum_power(signal):
    print 'cumulative power'
    suma=0
    totSign=sum(signal)
    for s in signal:
        pj=float(s)/totSign
        suma=suma+pj*math.log(pj)
    suma=suma*-1
    return suma

def plot_fft(signal):
    val=np.abs(np.fft.fft(signal))**2
    freqs=np.fft.fftfreq(signal.size)
    plt.plot(freqs,val)
    plt.show()


feature_D3=[]
feature_D4=[]
feature_D5=[]
all_data=[]

import pdb
for i in range(1,16):
    print 'processing data-',str(i)
    dataset=pickle.load(open('../data_based_on_DEAP_ica_filter_average/newBasedDEAP_'+str(i)+'.p','rb'))
    dataset=np.array(dataset)
    print dataset


    all_feature=[]
    for one_trial in dataset:
        feature_one_trial_all=[]
        for one_channel in one_trial:
            one_channel.reshape(len(one_channel),1)
            [(cA5,cD5),(cA4,cD4),(cA3,cD3),(cA2,cD2),(cA1,cD1)]=pywt.swt(one_channel[3*128:60*128],'haar',5)
            detail3=[means_of_absolte_2st_normalize(cD5)]
            detail4=[means_of_absolte_2st_normalize(cD4)]
            detail5=[means_of_absolte_2st_normalize(cD3)]
            detail2=[means_of_absolte_2st_normalize(cD2)]
            detail1=[means_of_absolte_2st_normalize(cD1)]
            feature_one_trial_all.extend(detail3)
            feature_one_trial_all.extend(detail4)
            feature_one_trial_all.extend(detail2)

        all_feature.append(feature_one_trial_all)
    all_data.append(all_feature)



fileName="/media/syahdeini/961C8FA51C8F7ECD/TAAAA/metadata/participant_ratings_sorted.csv"
dataSetPartisipant=csv_reader(fileName)
dataSetPartisipant=dataSetPartisipant[1:] # remove header


# Data akan di dump menjadi satu file csv
# dengan tiap 40 trial * 32 orang baris data
# data diurutkan berdasarkan experiment_id, yang sudah otomatis terurut

print 'ordering'
dataset_par_order=[]
for data in dataSetPartisipant:
    person=int(data[0])-1
    trial=int(data[1])-1
    if person<15:
        dataset_par_order.append(all_data[person][trial])
print 'dataset order',np.array(dataset_par_order).shape

print 'saving file'
with open('../result/detailsWavelet.csv','wb') as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    for datum in dataset_par_order:
        print 'datum ',datum
        spamwriter.writerow(datum)

