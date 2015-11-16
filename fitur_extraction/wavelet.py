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


import csv
import pywt
import math
import numpy as np
import pickle
# reading data


# function for reading csv file
def csv_reader(filename):
    ret_List=[]
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

# list for collecting all feature on all data files
all_data=[]

import pdb
for i in range(1,16):
#1.  read data per-file, one file contains 40 trial and 32 EEG channels, for one person
    print 'processing data-',str(i)
    dataset=pickle.load(open('../data_based_on_DEAP_ica_filter_average/newBasedDEAP_'+str(i)+'.p','rb'))
    dataset=np.array(dataset)
    print dataset


    all_feature=[]
    for one_trial in dataset:
        feature_one_trial_all=[]
        for one_channel in one_trial:
            one_channel.reshape(len(one_channel),1)
            #2. doing SWT (Stationary wavelet transform), with mother wavelet=haar and level decomposition=5
            [(cA5,cD5),(cA4,cD4),(cA3,cD3),(cA2,cD2),(cA1,cD1)]=pywt.swt(one_channel[5*128:65*128],'haar',5)
            #3. extracting featur from 5 details. (the feature function is changable, regarding what we need)
            detail3=[means_of_absolte_2st_normalize(cD5)]
            detail4=[means_of_absolte_2st_normalize(cD4)]
            detail5=[means_of_absolte_2st_normalize(cD3)]
            detail2=[means_of_absolte_2st_normalize(cD2)]
            detail1=[means_of_absolte_2st_normalize(cD1)]
            #4. extending all 5 fitures from 32 channel to list (one_trial fitur)
            feature_one_trial_all.extend(detail1)
            feature_one_trial_all.extend(detail2)
            feature_one_trial_all.extend(detail3)
            feature_one_trial_all.extend(detail4)
            feature_one_trial_all.extend(detail5)
        # 5. appending to list the one_trial fitur
        all_feature.append(feature_one_trial_all)
# combine fitur from all file to become one big list
    all_data.append(all_feature)



# read the metadata file, metadata file can be found in DEAP website
fileName="../metadata/participant_ratings_sorted.csv"
dataSetPartisipant=csv_reader(fileName)
dataSetPartisipant=dataSetPartisipant[1:]  # remove the header


#6.  ordering data based on sortered experiment_id
print 'ordering'
dataset_par_order=[]
for data in dataSetPartisipant:
    person=int(data[0])-1 # participant-id
    trial=int(data[1])-1 # trial-id
    if person<15:
        dataset_par_order.append(all_data[person][trial])
print 'dataset order',np.array(dataset_par_order).shape


# saving file as csv
print 'saving file'
with open('../result/detailsWavelet.csv','wb') as csvfile:
    spamwriter=csv.writer(csvfile,delimiter=',')
    for datum in dataset_par_order:
        spamwriter.writerow(datum)

