"""06 qrel envelope protocols

Relaxation index q_rel: q-exponential fits to the autocorrelation of the 1 ms RMS envelope under four fitting protocols.

Reads  : (raw traces only) + RAW_ZIP
Writes : tripleto_qrel.csv
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

from qstat import qstat_fit_q, qstat_lnq

import numpy as np
import pandas as pd
import io
import zipfile
import time
from scipy import stats

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

def crossT(d,k,lev=3.0):
    m=np.isfinite(k); dd=np.asarray(d)[m]; kk=np.asarray(k)[m]
    if kk[0]>=lev: return dd[0],'censored_low'
    idx=np.where(kk>=lev)[0]
    if idx.size==0: return np.nan,'censored_high'
    j=idx[0]; x0,x1=np.log10(dd[j-1]),np.log10(dd[j]); y0,y1=kk[j-1],kk[j]
    return 10**(x0+(lev-y0)*(x1-x0)/(y1-y0)),'ok'

# Delta grid
dms = np.unique(np.round(np.logspace(np.log10(0.25), np.log10(500), 50)/1000*FS).astype(int))
dms = dms[dms >= 5]
deltas_ms = dms/FS*1000

ZP = RAW_ZIP
zf = zipfile.ZipFile(ZP)
npys = sorted(n for n in zf.namelist() if n.endswith('.npy'))

# Reproduce RNG state from trace
rng = np.random.default_rng(20261009)

# Cell: full re-run on 184 records (advances rng state)
rows = []
CB = []; CG = []; CS = []
t0 = time.time()
for nm in npys:
    pat,cls,fn = nm.split('/')
    x = np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    kb,g2m,g2se,nw = kappa_curve2(x,dms)
    xs = rng.permutation(x); kbs,g2ms,g2ses,_ = kappa_curve2(xs,dms)
    Tb,fb = cross_T2(deltas_ms,kb,np.full(len(dms),3.0))
    Tt,ft = cross_T2(deltas_ms,g2m,2.0*g2se)
    Tbs,fbs = cross_T2(deltas_ms,kbs,np.full(len(dms),3.0))
    Tts,fts = cross_T2(deltas_ms,g2ms,2.0*g2ses)
    rows.append(dict(file=nm,patient=pat,cls=cls,n_samples=int(x.size),dur_s=x.size/FS,
        kurt_global=float((x**4).mean()/(x**2).mean()**2),
        T_bcs_ms=Tb,flag_bcs=fb,T_test_ms=Tt,flag_test=ft,
        T_bcs_shuf_ms=Tbs,flag_bcs_shuf=fbs,T_test_shuf_ms=Tts,flag_test_shuf=fts))
    CB.append(kb); CG.append(g2m); CS.append(kbs)
    del x,xs
CB=np.array(CB);CG=np.array(CG);CS=np.array(CS)
D=pd.DataFrame(rows)

# Calibration synthetic (advances rng state)
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

# Gaussian null (advances rng state) - 60 reps
nullvals=[]
for rep in range(60):
    y=rng.standard_normal(240000)
    kb,g2m,g2se,_=kappa_curve2(y,dms)
    Tb,fb=cross_T2(deltas_ms,kb,np.full(len(dms),3.0)); Tt,ft=cross_T2(deltas_ms,g2m,2.0*g2se)
    nullvals.append((Tb,fb,Tt,ft))

# qresid_janela_proporcional_T (advances rng state)
def locz(x,win):
    win=max(int(win),24); w=x.size//win
    if w<4: return None
    y=x[:w*win].reshape(w,win); sd=y.std(1,keepdims=True); sd[sd==0]=1
    return ((y-y.mean(1,keepdims=True))/sd).ravel()
KS={'k2':2.0,'k5':5.0,'kmed':50.0/4.026,'k25':25.0}
out=[]; rgw=np.random.default_rng(2026)
for nm in npys:
    pat,cls,fn=nm.split('/')
    row=D[D.file==nm].iloc[0]
    if row.flag_bcs!='ok': continue
    x=np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    xs=rgw.permutation(x)
    rec=dict(file=nm,patient=pat,cls=cls,T_bcs_ms=row.T_bcs_ms)
    for tag,k in KS.items():
        W=int(round(k*row.T_bcs_ms/1000*FS))
        rec['win_%s_ms'%tag]=max(W,24)/FS*1000
        zr=locz(x,W); zs=locz(xs,W)
        rec['q_%s'%tag]=qstat_fit_q(zr)['q'] if zr is not None else np.nan
        rec['q_%s_shuf'%tag]=qstat_fit_q(zs)['q'] if zs is not None else np.nan
    W50=int(round(0.050*FS)); rec['win_fix50_ms']=50.0
    rec['q_fix50']=qstat_fit_q(locz(x,W50))['q']; rec['q_fix50_shuf']=qstat_fit_q(locz(xs,W50))['q']
    out.append(rec); del x,xs
WR=pd.DataFrame(out)

# tripleto_qstat advances rng state
rgq=np.random.default_rng(909); rows2=[]; t0=time.time()
for nm in npys:
    pat,cls,fn=nm.split('/')
    x=np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    f1=qstat_fit_q(x); f2=qstat_fit_q(rgq.permutation(x))
    rows2.append(dict(file=nm,patient=pat,cls=cls,n_samples=int(x.size),
                     q_stat_raw=f1['q'],r2_raw=f1['r2'],beta_raw=f1['beta'],
                     q_stat_shuf=f2['q'],r2_shuf=f2['r2']))
    del x
QS=pd.DataFrame(rows2).merge(WR[['file','T_bcs_ms','q_kmed','q_kmed_shuf','win_kmed_ms']].rename(
        columns={'q_kmed':'q_stat_resid','q_kmed_shuf':'q_stat_resid_shuf'}),on='file',how='left')
QS['edge_raw']=QS.q_stat_raw>=2.59

# Now compute qrel (advances rng state)
BIN = 24
def envelope(x,binn=BIN):
    w=x.size//binn; return np.sqrt((x[:w*binn].reshape(w,binn)**2).mean(1))

def acf(v,maxlag):
    v=v-v.mean(); n=v.size; nf=1<<int(np.ceil(np.log2(2*n)))
    F=np.fft.rfft(v,nf); c=np.fft.irfft(F*np.conj(F),nf)[:maxlag+1]
    return c/c[0]

def fit_qexp(t,C,qmin=1.0,qmax=6.0,step=0.01,cmin=0.02):
    m=np.isfinite(C)&(C>cmin)
    if m.sum()<8: return dict(q=np.nan,tau=np.nan,r2=np.nan,npts=int(m.sum()))
    k=np.argmax(~m) if (~m).any() else len(C)
    sl=slice(1,max(k,9))
    tt=t[sl]; cc=C[sl]; ok=np.isfinite(cc)&(cc>cmin); tt,cc=tt[ok],cc[ok]
    if tt.size<8: return dict(q=np.nan,tau=np.nan,r2=np.nan,npts=int(tt.size))
    best=(-np.inf,np.nan,np.nan)
    for qv in np.arange(qmin,qmax+1e-9,step):
        y=qstat_lnq(cc,qv)
        if not np.all(np.isfinite(y)): continue
        A=np.vstack([tt,np.ones_like(tt)]).T
        coef,res,_,_=np.linalg.lstsq(A,y,rcond=None)
        pred=A@coef; ss=((y-y.mean())**2).sum()
        r2=1-((y-pred)**2).sum()/ss if ss>0 else -np.inf
        if r2>best[0] and coef[0]<0: best=(r2,qv,-1/coef[0])
    return dict(q=best[1],tau=best[2],r2=best[0],npts=int(tt.size))

MAXLAG=1000
tlag=np.arange(MAXLAG+1)*1.0
rgr=np.random.default_rng(31415); out=[]; ACFs=[]; t0=time.time()
for nm in npys:
    pat,cls,fn=nm.split('/')
    x=np.ascontiguousarray(np.load(io.BytesIO(zf.read(nm))),dtype=np.float64)
    e=envelope(x); ml=min(MAXLAG,e.size//4)
    C=acf(e,ml); f=fit_qexp(tlag[:ml+1],C)
    es=envelope(rgr.permutation(x)); Cs=acf(es,ml); fs=fit_qexp(tlag[:ml+1],Cs)
    phi=np.clip(C[1],-0.95,0.95); n=e.size
    a=np.zeros(n); innov=rgr.standard_normal(n)
    for i in range(1,n): a[i]=phi*a[i-1]+innov[i]
    Ca=acf(a,ml); fa=fit_qexp(tlag[:ml+1],Ca)
    out.append(dict(file=nm,patient=pat,cls=cls,q_rel=f['q'],tau_rel_ms=f['tau'],r2_rel=f['r2'],npts_rel=f['npts'],
                    q_rel_shuf=fs['q'],tau_rel_shuf_ms=fs['tau'],r2_rel_shuf=fs['r2'],
                    q_rel_ar1=fa['q'],tau_rel_ar1_ms=fa['tau'],acf_lag1=C[1]))
    ACFs.append(np.pad(C,(0,MAXLAG+1-C.size),constant_values=np.nan))
    del x
QR=pd.DataFrame(out); ACFs=np.array(ACFs)

# wide-grid fit
res=[]
for i in range(ACFs.shape[0]):
    C=ACFs[i]; ml=np.isfinite(C).sum()-1
    f=fit_qexp(tlag[:ml+1],C[:ml+1],qmin=0.0,qmax=6.0,step=0.01)
    res.append((f['q'],f['tau'],f['r2'],f['npts']))
W2=pd.DataFrame(res,columns=['q_rel_wide','tau_wide_ms','r2_wide','npts_wide'])
QR=pd.concat([QR,W2],axis=1)

# fit from lag 5
def fit_from(start,C,qmin=0.0,qmax=6.0,cmin=0.02):
    ml=np.isfinite(C).sum()-1; t=tlag[start:ml+1]; c=C[start:ml+1]
    ok=np.isfinite(c)&(c>cmin); t,c=t[ok],c[ok]
    if t.size<8: return np.nan,np.nan,np.nan
    best=(-np.inf,np.nan,np.nan)
    for qv in np.arange(qmin,qmax+1e-9,0.01):
        y=qstat_lnq(c/c[0],qv)
        if not np.all(np.isfinite(y)): continue
        A=np.vstack([t-t[0],np.ones_like(t)]).T
        coef,_,_,_=np.linalg.lstsq(A,y,rcond=None); pred=A@coef
        ss=((y-y.mean())**2).sum(); r2=1-((y-pred)**2).sum()/ss if ss>0 else -np.inf
        if r2>best[0] and coef[0]<0: best=(r2,qv,-1/coef[0])
    return best[1],best[2],best[0]

rr=[fit_from(5,ACFs[i]) for i in range(ACFs.shape[0])]
RR=pd.DataFrame(rr,columns=['q_rel5','tau5','r25'])
QR=pd.concat([QR,RR],axis=1)
QR.to_csv(DATA / 'tripleto_qrel.csv',index=False,float_format='%.5g')
