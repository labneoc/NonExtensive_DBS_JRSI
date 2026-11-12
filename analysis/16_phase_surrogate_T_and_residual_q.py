"""16 phase surrogate T and residual q

Phase-randomized (Fourier) surrogates: T and residual q per record, for the control that preserves the power spectrum.

Reads  : (raw traces only) + RAW_ZIP
Writes : m3_phase_surrogate_T_qres.csv
Source : recovered from the execution record of the analysis session; the only edits are
         the path constants below and, where noted in README.md, the import of qstat.py.

Author : Prof. Bruno Duarte Gomes
         Laboratório de Neurofisiologia Eduardo Oswaldo Cruz, ICB-UFPA
         Laboratório de Simulação e Biologia Computacional, CCAD-UFPA
"""
import io
import os
import re
import time
import zipfile
import numpy as np
import pandas as pd
from scipy import stats, signal as sg
from scipy import stats as st

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA, RAW_ZIP

import numpy as np, pandas as pd, zipfile, io, time, json
from scipy import stats
Z=RAW_ZIP
zf=zipfile.ZipFile(Z); npys=sorted(n for n in zf.namelist() if n.endswith('.npy'))
FS=24000.0
def qstat_lnq(u,q): return np.log(u) if abs(q-1)<1e-12 else (np.power(u,1-q)-1)/(1-q)
def qstat_fit_q(s, qmin=1.01, qmax=2.60, step=0.01, nbins=120, clip=0.5):
    s=np.asarray(s,float); lo,hi=np.percentile(s,[clip,100-clip])
    h,e=np.histogram(s[(s>lo)&(s<hi)],bins=nbins,density=True); c=0.5*(e[1:]+e[:-1]); m=h>0
    xx=c[m]**2; ratio=h[m]/h[m].max(); best=(np.nan,-1.0)
    for q in np.arange(qmin,qmax,step):
        yy=qstat_lnq(ratio,q); sl,ic,rr=stats.linregress(xx,yy)[:3]
        if rr**2>best[1]: best=(float(q),float(rr**2))
    return best[0]
def kappa_curve(x,deltas,min_win=8):
    kb=np.full(len(deltas),np.nan); N=x.size
    for i,n in enumerate(deltas):
        w=N//n
        if w<min_win or n<5: continue
        y=x[:w*n].reshape(w,n); y=y-y.mean(1,keepdims=True); s2=y*y
        m2=s2.mean(1); m4=(s2*s2).mean(1); ok=m2>0; kb[i]=(m4[ok]/(m2[ok]**2)).mean()
    return kb
def cross_T(d,k,level=3.0):
    m=np.isfinite(k); d=np.asarray(d,float)[m]; kk=np.asarray(k)[m]
    if d.size<2: return np.nan,'empty'
    if kk[0]>=level: return d[0],'censored_low'
    idx=np.where(kk>=level)[0]
    if idx.size==0: return np.nan,'censored_high'
    j=idx[0]; x0,x1=np.log10(d[j-1]),np.log10(d[j]); y0,y1=kk[j-1],kk[j]
    return 10**(x0+(level-y0)*(x1-x0)/(y1-y0)),'ok'
dms=np.unique(np.round(np.logspace(np.log10(0.25),np.log10(500),50)/1000*FS).astype(int)); dms=dms[dms>=5]
deltas_ms=dms/FS*1000
def locz(x,win):
    win=max(int(win),24); w=x.size//win
    if w<4: return None
    y=x[:w*win].reshape(w,win); sd=y.std(1,keepdims=True); sd[sd==0]=1
    return ((y-y.mean(1,keepdims=True))/sd).ravel()
def ft_surr(x,rng):
    F=np.fft.rfft(x); ph=rng.uniform(0,2*np.pi,F.size); ph[0]=0
    if x.size%2==0: ph[-1]=0
    return np.fft.irfft(np.abs(F)*np.exp(1j*ph),n=x.size)
KMED=50.0/4.026
rng=np.random.default_rng(2027); rows=[]; t0=time.time()
for nm in npys:
    pat,cls,fn=nm.split('/')
    x=np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    xs=ft_surr(x,rng)
    Tr,fr=cross_T(deltas_ms,kappa_curve(x,dms)); Ts,fs=cross_T(deltas_ms,kappa_curve(xs,dms))
    rec=dict(file=nm,patient=pat,cls=cls,T_real=Tr,flag_real=fr,T_ft=Ts,flag_ft=fs,q_glob_ft=qstat_fit_q(xs))
    if fr=='ok':
        W=int(round(KMED*Tr/1000*FS))
        zr=locz(x,W); zs=locz(xs,W)
        rec['q_kmed_real']=qstat_fit_q(zr) if zr is not None else np.nan
        rec['q_kmed_ft']=qstat_fit_q(zs) if zs is not None else np.nan
    W50=int(round(0.050*FS)); rec['q_fix50_real']=qstat_fit_q(locz(x,W50)); rec['q_fix50_ft']=qstat_fit_q(locz(xs,W50))
    rows.append(rec); del x,xs
R=pd.DataFrame(rows); R.to_csv(DATA / 'm3_phase_surrogate_T_qres.csv',index=False,float_format='%.6g')
print('%.0f s n=%d'%(time.time()-t0,len(R)))
