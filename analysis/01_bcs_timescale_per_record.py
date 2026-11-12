"""01 bcs timescale per record

Beck-Cohen-Swinney time scale T per record: crossing of the window-averaged kurtosis kappa(Delta) = 3.

Reads  : (raw traces only) + RAW_ZIP
Writes : bcs_T_por_registro_184.csv
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

import zipfile, os, io, time, re, json
import numpy as np
import pandas as pd
from scipy import stats

z = RAW_ZIP
zf = zipfile.ZipFile(z)
names = zf.namelist()
npys = sorted(n for n in names if n.endswith('.npy'))

FS = 24000.0

def kappa_curve2(x, deltas, min_win=8):
    nd=len(deltas)
    kb=np.full(nd,np.nan); g2m=np.full(nd,np.nan); g2se=np.full(nd,np.nan); nw=np.zeros(nd,int)
    N=x.size
    for i,n in enumerate(deltas):
        w=N//n
        if w<min_win or n<5: continue
        y=x[:w*n].reshape(w,n); y=y-y.mean(1,keepdims=True)
        s2=y*y; m2=s2.mean(1); m4=(s2*s2).mean(1)
        ok=m2>0; r=m4[ok]/(m2[ok]**2)
        kb[i]=r.mean()
        G2=((n-1.0)/((n-2.0)*(n-3.0)))*((n+1.0)*(r-3.0)+6.0)
        g2m[i]=G2.mean(); g2se[i]=G2.std(ddof=1)/np.sqrt(G2.size); nw[i]=w
    return kb,g2m,g2se,nw

def cross_T2(dms_ms,y,thr):
    m=np.isfinite(y)&np.isfinite(thr)
    d=np.asarray(dms_ms,float)[m]; yy=np.asarray(y)[m]; tt=np.asarray(thr)[m]
    if d.size<2: return np.nan,'empty'
    dd=yy-tt
    if dd[0]>=0: return d[0],'censored_low'
    idx=np.where(dd>=0)[0]
    if idx.size==0: return np.nan,'censored_high'
    j=idx[0]; x0,x1=np.log10(d[j-1]),np.log10(d[j]); y0,y1=dd[j-1],dd[j]
    return 10**(x0+(0-y0)*(x1-x0)/(y1-y0)),'ok'

dms = np.unique(np.round(np.logspace(np.log10(0.25), np.log10(500), 50)/1000*FS).astype(int))
dms = dms[dms >= 5]
deltas_ms = dms/FS*1000

rng = np.random.default_rng(20261009)

# consume RNG state from earlier cells (calibration synthetic data, null tests, etc.)
# kappa_curve (original, used before kappa_curve2 was defined) - replicate RNG consumption
def kappa_curve(x, deltas, min_win=8):
    kb = np.full(len(deltas), np.nan); ku = np.full(len(deltas), np.nan); nw = np.zeros(len(deltas), int)
    N = x.size
    for i, n in enumerate(deltas):
        w = N // n
        if w < min_win or n < 5: continue
        y = x[:w*n].reshape(w, n)
        y = y - y.mean(1, keepdims=True)
        s2 = y*y
        m2 = s2.mean(1); m4 = (s2*s2).mean(1)
        ok = m2 > 0
        r = m4[ok]/(m2[ok]**2)
        kb[i] = r.mean()
        g2 = r - 3.0
        G2 = ((n-1.0)/((n-2.0)*(n-3.0)))*((n+1.0)*g2 + 6.0)
        ku[i] = (G2 + 3.0).mean()
        nw[i] = w
    return kb, ku, nw

def cross_T(deltas_ms, k, level=3.0):
    m = np.isfinite(k)
    d = np.asarray(deltas_ms, float)[m]; kk = np.asarray(k)[m]
    if d.size < 2: return np.nan, 'empty'
    if kk[0] >= level: return d[0], 'censored_low'
    idx = np.where(kk >= level)[0]
    if idx.size == 0: return np.nan, 'censored_high'
    j = idx[0]
    x0, x1 = np.log10(d[j-1]), np.log10(d[j]); y0, y1 = kk[j-1], kk[j]
    return 10**(x0 + (level-y0)*(x1-x0)/(y1-y0)), 'ok'

# First loop over npys (cell with rows/KB/KU)
rows = []; KB = []; KU = []; KBs = []; KUs = []
for ii, nm in enumerate(npys):
    pat, cls, fn = nm.split('/')
    x = np.load(io.BytesIO(zf.read(nm)))
    x = np.ascontiguousarray(x, dtype=np.float64)
    kb, ku, nw = kappa_curve(x, dms)
    xs = rng.permutation(x)
    kbs, kus, _ = kappa_curve(xs, dms)
    T_b, f_b = cross_T(deltas_ms, kb)
    T_u, f_u = cross_T(deltas_ms, ku)
    T_bs, f_bs = cross_T(deltas_ms, kbs)
    T_us, f_us = cross_T(deltas_ms, kus)
    rows.append(dict(file=nm, patient=pat, cls=cls, n_samples=int(x.size),
                     dur_s=x.size/FS, kurt_global=float(((x-x.mean())**4).mean()/((x-x.mean())**2).mean()**2),
                     T_ms=T_b, T_flag=f_b, T_ms_unb=T_u, T_flag_unb=f_u,
                     T_ms_shuf=T_bs, T_flag_shuf=f_bs, T_ms_unb_shuf=T_us, T_flag_unb_shuf=f_us))
    KB.append(kb); KU.append(ku); KBs.append(kbs); KUs.append(kus)
    del x, xs
KB=np.array(KB); KU=np.array(KU); KBs=np.array(KBs); KUs=np.array(KUs)
df = pd.DataFrame(rows)

# cal2 synthetic data (consumes RNG)
cal2=[]
for T0_ms in [2.0,5.0,20.0,50.0,200.0]:
    blk=int(round(T0_ms/1000*FS))
    for rep in range(20):
        nb=240000//blk
        beta=rng.gamma(shape=1.5,scale=1/1.5,size=nb)
        y=(rng.standard_normal((nb,blk))/np.sqrt(beta)[:,None]).ravel(); y=(y-y.mean())/y.std()
        kb,g2m,g2se,_=kappa_curve2(y,dms)
        Tb,fb=cross_T2(deltas_ms,kb,np.full(len(dms),3.0))
        Tt,ft=cross_T2(deltas_ms,g2m,2.0*g2se)
        cal2.append(dict(T0_ms=T0_ms,rep=rep,T_bcs_ms=Tb,T_test_ms=Tt,flag=ft))
cal2=pd.DataFrame(cal2)

# Gaussian null (consumes RNG, 20 reps)
Z=2.0
nullT=[]
for rep in range(20):
    y=rng.standard_normal(240000)
    kb,g2m,g2se,_=kappa_curve2(y,dms)
    nullT.append((cross_T2(deltas_ms,kb,np.full(len(dms),3.0))[1], cross_T2(deltas_ms,g2m,Z*g2se)[1]))

# 60 more Gaussian null reps
nullvals=[]
for rep in range(60):
    y=rng.standard_normal(240000)
    kb,g2m,g2se,_=kappa_curve2(y,dms)
    Tb,fb=cross_T2(deltas_ms,kb,np.full(len(dms),3.0)); Tt,ft=cross_T2(deltas_ms,g2m,Z*g2se)
    nullvals.append((Tb,fb,Tt,ft))

# Full re-run on the 184 records
res=[]; CB=[]; CG=[]; CS=[]
for nm in npys:
    pat,cls,fn=nm.split('/')
    x=np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    kb,g2m,g2se,nw=kappa_curve2(x,dms)
    xs=rng.permutation(x); kbs,g2ms,g2ses,_=kappa_curve2(xs,dms)
    Tb,fb=cross_T2(deltas_ms,kb,np.full(len(dms),3.0))
    Tt,ft=cross_T2(deltas_ms,g2m,Z*g2se)
    Tbs,fbs=cross_T2(deltas_ms,kbs,np.full(len(dms),3.0))
    Tts,fts=cross_T2(deltas_ms,g2ms,Z*g2ses)
    res.append(dict(file=nm,patient=pat,cls=cls,n_samples=int(x.size),dur_s=x.size/FS,
        kurt_global=float((x**4).mean()/ (x**2).mean()**2),
        T_bcs_ms=Tb,flag_bcs=fb,T_test_ms=Tt,flag_test=ft,
        T_bcs_shuf_ms=Tbs,flag_bcs_shuf=fbs,T_test_shuf_ms=Tts,flag_test_shuf=fts))
    CB.append(kb); CG.append(g2m); CS.append(kbs)
    del x,xs
CB=np.array(CB);CG=np.array(CG);CS=np.array(CS)
D=pd.DataFrame(res)

# calibration inversion
cm = cal2.groupby('T0_ms')[['T_bcs_ms','T_test_ms']].median()
def invert(Tmeas, col):
    xs=np.log10(cm[col].values); ys=np.log10(cm.index.values.astype(float))
    o=np.argsort(xs)
    return 10**np.interp(np.log10(Tmeas), xs[o], ys[o], left=np.nan, right=np.nan)
D['T0_hat_bcs_ms']=invert(D['T_bcs_ms'].values,'T_bcs_ms')
D['T0_hat_test_ms']=invert(D['T_test_ms'].values,'T_test_ms')

D2=D.copy()
D2['T0_hat_bcs_ms']=D['T0_hat_bcs_ms']; D2['T0_hat_test_ms']=D['T0_hat_test_ms']
cols=['file','patient','cls','n_samples','dur_s','kurt_global','T_bcs_ms','flag_bcs','T_test_ms','flag_test',
      'T_bcs_shuf_ms','flag_bcs_shuf','T_test_shuf_ms','flag_test_shuf','T0_hat_bcs_ms','T0_hat_test_ms']
D2[cols].to_csv(DATA / 'bcs_T_por_registro_184.csv',index=False,float_format='%.5g')
