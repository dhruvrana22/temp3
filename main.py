import wave
import struct
import sys
import csv
import numpy 
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np
import pandas as pd
from vmdpy import VMD
import sys, os, os.path

input_filename = 'input.wav'
if input_filename[-3:] != 'wav':
    print('WARNING!! Input File format should be *.wav')
    sys.exit()
samrate,data = wavfile.read(str('./' + input_filename))
np.savetxt('out.csv',data,delimiter=",")
df=pd.read_csv('out.csv')
time=df.iloc[:,0]
f=df.iloc[:,1];
alpha = 2000       # moderate bandwidth constraint
tau = 0           # noise-tolerance (no strict fidelity enforcement)
K = 5              # 3 modes
DC = 0             # no DC part imposed
init = 1           # initialize omegas uniformly
tol = 1e-7

u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)
u.to_csv('out1.csv', header=False, index=False)

df=pd.read_csv('out1.csv')
print(df.shape)
pca = PCA()
Xt = pca.fit_transform(df)
print(Xt.shape)
fcol=Xt[:,3]
scol=Xt[:,4]
msum=fcol+scol
np.savetxt('outpca.csv', msum, delimiter=",")

df=pd.read_csv('out.csv')
df2=pd.read_csv('outpca.csv')
df.iloc[:,1]=df2
df.to_csv('final.csv', header=False, index=False)
