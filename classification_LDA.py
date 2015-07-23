__author__ = 'root'
__author__ = 'admin'
import numpy as np
import csv

import from_old_TA.LDA_pkg.LDA as LDA_pkg
import from_old_TA.gettingClasses as gettingClasses
from from_old_TA.crossValidation import crossValidation


#### PADA PROGRAM INI DILAKUKAN MDA PADA DATA ############
## ada dua data yaitu data dengan hasil ICA dan data tanpa ICA
## data telah di ICA dan di SWT dan PCA dengan mengambil lima komponen yang terbesar
## dan mengekspan data
def rata_sample(data,divider):
    dataset=[]

    data=np.array(data).astype(np.float64)
    for dataC in data:
        dataSatuanChannel=[]
        len_data=len(dataC)
        jum_cluster=len_data/divider
        for i in range(jum_cluster):
            dataSatuanChannel.append(np.mean(dataC[i*divider:(i+1)*divider],axis=0))
        dataset.append(dataSatuanChannel)
    return dataset


dataset=[]
with open('result/detailsWavelet.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        dataset.append(row)
dataset=list(dataset)

import sklearn.decomposition as sklearn
# pca=sklearn.PCA(n_components=50)
# dataset=pca.fit_transform(dataset)
# import numpy as np
# import numpy as np
# from sklearn.lda import LDA
# kelas=gettingClasses.getClassData()
# clf=LDA()
# clf.fit(dataset,kelas)#  print

print 'dataset dimensi ',np.array(dataset).shape
# dataset=rata_sample(dataset,100)
kelas=gettingClasses.getClassData()
print 'kelas dimensi ',np.array(kelas).shape
from sklearn.lda import LDA
from sklearn.decomposition import pca
# pca_a=pca.PCA(n_components=50)
# dataset=pca_a.fit_transform(dataset)
pred_tot=0
for j in range(6):
    datasetN=list(dataset)
    kelasN=list(kelas)
    lda=LDA_pkg.LDA_Classifier()
    CV=crossValidation()
    dataTest,kelasTest,dataTrain,kelasTrain=CV.separate(datasetN,kelasN,30)
    # clf=LDA()
    # clf.fit(dataTrain,kelasTrain)
    # kelasPred=clf.predict(dataTest)
    lda.Train(np.array(dataTrain).astype(np.float64),np.array(kelasTrain).astype(np.float64))
    kelasPred=lda.Test(np.array(dataTest).astype(np.float64))
    acc=0
    for kD,kR in zip(kelasPred,kelasTest):
        print 'klasifikasi ',int(kD)+1,' ',int(kR)
        if int(kD)+1==int(kR):
            acc+=1
    pred_tot+=float(acc)/len(kelasTest)*100
    print acc,' ',len(kelasTest),' ',float(acc)/len(kelasTest)*100
print pred_tot/6
