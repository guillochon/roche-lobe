import numpy as np, sys, time
from scipy.optimize import brentq
from paths import PAPER
LD=np.longdouble
def panels(edges,n):
    g,w=np.polynomial.legendre.leggauss(n); g=g.astype(LD); w=w.astype(LD); X=[];W=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        lo,hi=LD(lo),LD(hi); X.append((hi-lo)/2*g+(hi+lo)/2); W.append((hi-lo)/2*w)
    return np.concatenate(X),np.concatenate(W)
def V_red(eps, n=90):
    eps=LD(eps)
    r1=LD(brentq(lambda r: r**3+float(eps)*r**4-1,0.5,1.0,xtol=1e-16))
    for _ in range(5): r1-= (r1**3+eps*r1**4-1)/(3*r1**2+4*eps*r1**3)
    Uc=3/r1+LD(1.5)*r1**2+eps*r1**3
    se=float(np.sqrt(eps)) if eps>0 else 1e-3
    ph_edges=[0,0.3,np.pi/2,np.pi-0.3]+[np.pi-k*se for k in (64,16,4,1) if k*se<0.3]+[np.pi]
    th_edges=[0,np.pi/2-0.3]+[np.pi/2-k*se for k in (64,16,4,1) if k*se<0.3]+[np.pi/2]
    th,thw=panels(th_edges,n); ph,phw=panels(ph_edges,n)
    TH,PH=np.meshgrid(th,ph,indexing='ij'); W=np.outer(thw*np.sin(th),phw)
    nx=(np.sin(TH)*np.cos(PH)).ravel(); nz=np.cos(TH).ravel()
    A=(3*nx**2-nz**2)/2; B=(5*nx**3-3*nx)/2
    U=lambda r: 3/r+A*r**2+eps*B*r**3
    Rg=np.linspace(LD(0.3),r1,200)
    Z=3/Rg[None,:]+A[:,None]*Rg[None,:]**2+eps*B[:,None]*Rg[None,:]**3
    below=Z<Uc; anyb=below.any(axis=1); idx=np.maximum(np.where(anyb,np.argmax(below,axis=1),199),1)
    lo,hi=Rg[idx-1].copy(),Rg[idx].copy()
    for _ in range(70):
        m=(lo+hi)/2; up=U(m)>=Uc; lo=np.where(up,m,lo); hi=np.where(up,hi,m)
    R=np.where(anyb,(lo+hi)/2,r1).reshape(TH.shape)
    return LD(4)/3*np.sum(W*R**3)
if __name__=="__main__":
    VH=LD('1.508549856131113140040991')
    print("eps=0:", V_red(0.0), " VH=", VH, " diff", V_red(0.0)-VH, flush=True)
    for n in (60,90,130): print("conv eps=1e-4 n=%d:"%n, V_red(1e-4,n), flush=True)
    out=[]
    for e in [1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5,5e-6,2e-6,1e-6,5e-7,2e-7,1e-7]:
        t=time.time(); v=V_red(e); out.append((e,v)); print(f"{e:.0e} {v:.18f}  ({time.time()-t:.0f}s)", flush=True)
    np.savetxt(PAPER / 'vred.txt', np.array([[float(e),float(v)] for e,v in out]), fmt='%.6e %.18f')
