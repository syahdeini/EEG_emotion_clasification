'''
    Feature extraction using FFT feature
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
import scipy.stats as sc

# reading csv file function
def csv_reader(filename):
    with open(filename,'rb') as file_obj:
        reader=csv.reader(file_obj,delimiter=',')
        data=[]
        for row in reader:
            data.append(row)
        return data
        ret_List.append(H)
    return ret_List

####  Feature extraction functions ###############

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


def signal_power(signal,n_bin=0):
    if n_bin!=0:
        return np.abs(np.fft.fft(signal,n=n_bin))**2
    return np.abs(np.fft.fft(signal))**2

def plot_fft(signal):
    val=np.abs(np.fft.fft(signal))**2
    freqs=np.fft.fftfreq(signal.size)
    plt.plot(freqs,val)
    plt.show()

def cum_power(signal):
    # shannon power
    suma=0
    totSign=sum(signal)
    for s in signal:
        pj=float(s)/totSign
        suma=suma+pj*math.log(pj)
    suma=suma*-1
    return suma


## another feature extraction with some modification ##
# average_with_cutting : this function will average every N-list of signal and
#                   put it into one list
# cum_power_with_cutting : this function will add cumulative power every N-list
#                   of signal, and combine it into one list
# entropy_signal : this function will calculate entropy for every N-list of
#                   signal, and combine it into one list
# spectral_entropy : this function will calculate


def average_with_cutting(signal,N):
    print '>>', len(signal)
    new_signal=[]
    sep=len(signal)/N
    print '>>>', sep,' ',N
    for i in range(N):
        new_signal.append(average(signal[sep*i:sep*(i+1)]))
    return new_signal

def cum_power_with_cutting(signal,N):
    new_signal=[]
    sep=len(signal)/N
    for i in range(sep):
        new_signal.append(cum_power(signal[N*i:N*(i+1)]))
    return new_signal


def entropy_signal(signal,N):
    new_signal=[]
    sep=len(signal)/N
    for i in range(sep):
        new_signal.append(sc.entropy(signal[N*i:N*(i+1)]))
    return new_signal

# inp = 1 berarti list of signal, inp=0 hanya satu signal
# if inp==1, it means that we are dealling with
def spectral_entropy(signal,n_bins=0,inp=1):
    if inp==1:
        sum=[]
        for s in signal:
            temp=signal_power(s,n_bin=n_bins_global)[:len(signal)/2]
            ent=0
            for d in temp:
                ent=ent+d*math.log(d,2)
            sum.append(ent)
    else:
        signal=signal_power(signal,n_bins)[:len(signal)/2]
        sum=0
        for d in signal:
            sum=sum+d*math.log(d,2)
    return sum

# this function will divide a signal into N partition and overlaping
def divide_signal(signal,N,gap=0):
    new_signal=[]
    preced=N*(float(gap)/100)
    sep=math.floor(len(signal)/N)+1
    for i in range(int(sep)-1):
        new_signal.append(signal[N*i:N*(i+1)+preced])
    new_signal.append(signal[preced*(i+1):]) # adding the last piece
    return new_signal

# this function will do hamming window for every list,
# if inp=1 than do hamming window on multi-list
# if inp=0 than do hamming wiindo on one-list
def hamming_all_signals(signal,inp=1):
    new_signal=[]
    if inp==1:
        for s in signal:
            ham_wind=sci_sig.hamming(len(s))
            new_signal.append(s*ham_wind)
    else:
        ham_wind=sci_sig.hamming(len(signal))
        return signal*ham_wind
    return new_signal

# this function wil calculate the spectral centroid
#
def spectral_centroid(signal,inp=1):
    if inp==1:
        sums=[]
        for sig in signal:
            sig=signal_power(sig)[:len(sig)/2]
            sum=0
            sumN=0
            for idx,s in enumerate(sig):
                sum=sum+s*(idx+1)
                sumN=sum+s
            sums.append(float(sum)/sumN)
        return sums
    else:
        sum=0
        sumN=0
        signal=signal_power(signal)[:len(signal)/2]
        for idx,s in enumerate(signal):
            sum=sum+s*(idx+1)
            sumN=sum+s
        return float(sum)/sumN


feature_D3=[]
feature_D4=[]
feature_D5=[]
all_data=[]
n_bins_global=96


import scipy.signal as sci_sig
for i in range(1,23):
    print 'processing data-',str(i)
#1. read data pickle
    dataset=pickle.load(open('../data_based_on_DEAP_ica_filter_average/newBasedDEAP_'+str(i)+'.p','rb'))
    dataset=np.array(dataset)

    all_feature=[]
    for id_t,one_trial in enumerate(dataset):
        fitur_one_channel=[]
        for id_c,one_channel in enumerate(one_trial):
            one_channel.reshape(len(one_channel),1)
#2, divide the signal into
            one_channel=divide_signal(one_channel,5*128,gap=25)
            one_channel_hams=hamming_all_signals(one_channel)
            spec_entrs=[]
            for s in one_channel_hams:
                spec_entrs.append(signal_power(s))
            fitur_one_channel.append(np.mean(spec_entrs))
        all_feature.append(fitur_one_channel)
    all_data.append(all_feature)



# read the medatada file, metadata can be found in DEAP Website,
# metadata consist of participant-id/particilant number, trial-id and
# experiment-id. file video-list consist VAQ-online which is the class of our
# data

fileName="/media/syahdeini/961C8FA51C8F7ECD/TAAAA/metadata/participant_ratings_sorted.csv"
dataSetPartisipant=csv_reader(fileName)
dataSetPartisipant=dataSetPartisipant[1:] # remove header

# ordering the data
dataset_par_order=[]
for data in dataSetPartisipant:
    person=int(data[0])-1 # participant-id
    trial=int(data[1])-1 # trial-id
    print person,' ',trial
    if person<22: # we try to get a limited number of people
        dataset_par_order.append(all_data[person][trial])
print 'dataset order',np.array(dataset_par_order).shape

# saving all data into csv
print 'saving file'
with open('../result/detailsFFT.csv','wb') as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    for datum in dataset_par_order:
        spamwriter.writerow(list(datum))








