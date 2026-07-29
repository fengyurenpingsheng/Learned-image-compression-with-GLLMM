#!/usr/bin/env python3
import csv, json, os, pathlib, struct, subprocess, tempfile, time, urllib.request, zlib
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'kodak24'
SHARD = int(os.environ.get('SHARD_INDEX', '0'))
NSHARD = int(os.environ.get('NUM_SHARDS', '1'))
OUT = ROOT / 'kodak24_rrc_results' / f'shard_{SHARD}'
DATA.mkdir(exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
URLS = ('https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png',
        'http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png')
ORDERS = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
SPECS = [None] + [(o,p) for o in ORDERS for p in (0,1,2)]

def get_data(indices):
    for i in indices:
        p = DATA / f'kodim{i:02d}.png'
        if p.exists(): continue
        error = None
        for u in URLS:
            try:
                urllib.request.urlretrieve(u.format(i), p); error = None; break
            except Exception as e: error = e
        if error is not None: raise error

def entropy(a):
    h = np.bincount(a.ravel(), minlength=256); p = h[h > 0] / a.size
    return float(-(p * np.log2(p)).sum())

def proxy(a):
    vals = sum(entropy(a[...,c]) for c in range(3))
    dh = (a[:,1:].astype(np.int16)-a[:,:-1].astype(np.int16)) & 255
    dv = (a[1:].astype(np.int16)-a[:-1].astype(np.int16)) & 255
    return 0.35*vals + 0.325*sum(entropy(dh[...,c]) for c in range(3)) + 0.325*sum(entropy(dv[...,c]) for c in range(3))

def fwd(x, tid):
    if tid == 0: return x.copy()
    order, pred_id = SPECS[tid]
    q = x.astype(np.int16)
    a = q[...,order[0]]; b0 = q[...,order[1]]; c0 = q[...,order[2]]
    b = (b0-a) & 255
    pred = a if pred_id == 0 else b0 if pred_id == 1 else (a+b0+1)//2
    c = (c0-pred) & 255
    return np.stack((a,b,c),-1).astype(np.uint8)

def inv(y, tid):
    if tid == 0: return y.copy()
    order, pred_id = SPECS[tid]
    q = y.astype(np.int16); a=q[...,0]; br=q[...,1]; cr=q[...,2]
    b0 = (br+a) & 255
    pred = a if pred_id == 0 else b0 if pred_id == 1 else (a+b0+1)//2
    c0 = (cr+pred) & 255
    out = np.empty_like(y); out[...,order[0]]=a; out[...,order[1]]=b0; out[...,order[2]]=c0
    return out.astype(np.uint8)

def local_transform(x, bs):
    h,w,_=x.shape; y=np.empty_like(x); ids=[]
    for yy in range(0,h,bs):
        for xx in range(0,w,bs):
            blk=x[yy:min(h,yy+bs),xx:min(w,xx+bs)]
            t=min(range(len(SPECS)), key=lambda k: proxy(fwd(blk,k)))
            y[yy:min(h,yy+bs),xx:min(w,xx+bs)]=fwd(blk,t); ids.append(t)
    side=struct.pack('<HHH',bs,h,w)+zlib.compress(bytes(ids),9)
    return y,side

def local_inverse(y, side):
    bs,h,w=struct.unpack('<HHH',side[:6]); ids=zlib.decompress(side[6:]); x=np.empty_like(y); k=0
    for yy in range(0,h,bs):
        for xx in range(0,w,bs):
            x[yy:min(h,yy+bs),xx:min(w,xx+bs)]=inv(y[yy:min(h,yy+bs),xx:min(w,xx+bs)],ids[k]); k+=1
    return x

def pad_shift(a, dy, dx):
    p=np.pad(a,((1,1),(1,1),(0,0)),mode='edge')
    return p[1+dy:1+dy+a.shape[0],1+dx:1+dx+a.shape[1]]

def project_smooth(src, E, iters, median=False):
    lo=np.maximum(0,src.astype(np.int16)-E); hi=np.minimum(255,src.astype(np.int16)+E)
    y=src.astype(np.int16)
    for _ in range(iters):
        n=[pad_shift(y,-1,0),pad_shift(y,1,0),pad_shift(y,0,-1),pad_shift(y,0,1),y]
        z=np.median(np.stack(n),axis=0) if median else sum(n)//len(n)
        y=np.minimum(hi,np.maximum(lo,z.astype(np.int16)))
    return y.astype(np.uint8)

def lattice_best(src,E):
    if E==0:return src.copy()
    s=2*E+1; out=np.empty_like(src)
    for c in range(3):
        x=src[...,c].astype(np.int16); lo=np.maximum(0,x-E); hi=np.minimum(255,x+E)
        best=None
        for off in range(s):
            q=np.rint((x-off)/s).astype(np.int16)*s+off
            q=np.minimum(hi,np.maximum(lo,q)).astype(np.uint8)
            score=entropy(q)+0.5*entropy(((q[:,1:].astype(np.int16)-q[:,:-1].astype(np.int16))&255).astype(np.uint8))
            if best is None or score<best[0]:best=(score,q)
        out[...,c]=best[1]
    return out

def reconstructions(src,E):
    if E==0:return [('exact',src.copy())]
    s=2*E+1
    midpoint=np.minimum(255,(src.astype(np.int16)//s)*s+E).astype(np.uint8)
    return [('midpoint',midpoint),('lattice',lattice_best(src,E)),
            ('mean2',project_smooth(src,E,2,False)),('median2',project_smooth(src,E,2,True)),
            ('mean4',project_smooth(src,E,4,False))]

def run(cmd): subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def encode_jxl(a, effort):
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); p=td/'a.png'; o=td/'a.jxl'; Image.fromarray(a).save(p)
        run(['cjxl',str(p),str(o),'-d','0','-e',str(effort),'--num_threads=1']); return o.read_bytes()
def decode_jxl(payload):
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); p=td/'a.jxl'; o=td/'o.png'; p.write_bytes(payload)
        run(['djxl',str(p),str(o),'--num_threads=1']); return np.asarray(Image.open(o).convert('RGB'),np.uint8)
def encode_webp(a):
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); p=td/'a.png'; o=td/'a.webp'; Image.fromarray(a).save(p)
        run(['cwebp','-quiet','-lossless','-z','9','-m','6',str(p),'-o',str(o)]); return o.read_bytes()
def decode_webp(payload):
    with tempfile.TemporaryDirectory() as td:
        td=pathlib.Path(td); p=td/'a.webp'; o=td/'o.png'; p.write_bytes(payload)
        run(['dwebp','-quiet',str(p),'-o',str(o)]); return np.asarray(Image.open(o).convert('RGB'),np.uint8)

def rep_candidates(rec, top_global=4):
    scores=sorted((proxy(fwd(rec,t)),t) for t in range(len(SPECS)))
    for _,t in scores[:top_global]: yield f'g{t}',fwd(rec,t),bytes([t]),lambda a,s:inv(a,s[0])
    for bs in (64,128):
        y,s=local_transform(rec,bs); yield f'l{bs}',y,s,local_inverse

def evaluate(src,E,rname,rep,car,side,inverse,effort):
    t=time.time(); payload=encode_jxl(car,effort); rec_car=decode_jxl(payload); rec=inverse(rec_car,side)
    err=int(np.abs(src.astype(np.int16)-rec.astype(np.int16)).max())
    if err>E: raise RuntimeError(('bound',E,rname,rep,err))
    return {'total':16+len(rname)+len(rep)+len(side)+len(payload),'rname':rname,'rep':rep,'backend':'jxl',
            'side':len(side),'payload':len(payload),'seconds':time.time()-t,'err':err}

def bench_one(idx,E):
    src=np.asarray(Image.open(DATA/f'kodim{idx:02d}.png').convert('RGB'),np.uint8); n=src.shape[0]*src.shape[1]
    recon=reconstructions(src,E); screened=[]; webps=[]
    for rname,rec in recon:
        t=time.time(); wp=encode_webp(rec); wr=decode_webp(wp); err=int(np.abs(src.astype(np.int16)-wr.astype(np.int16)).max())
        if err>E: raise RuntimeError(('webp-bound',idx,E,rname,err))
        webps.append({'total':16+len(rname)+4+len(wp),'rname':rname,'rep':'raw','backend':'webp','side':0,'payload':len(wp),'seconds':time.time()-t,'err':err})
        for rep,car,side,inverse in rep_candidates(rec):
            item=evaluate(src,E,rname,rep,car,side,inverse,5); screened.append((item,car,side,inverse))
    finalists=[]
    for item,car,side,inverse in sorted(screened,key=lambda z:z[0]['total'])[:5]:
        finalists.append(evaluate(src,E,item['rname'],item['rep'],car,side,inverse,9))
    all_final=finalists+webps; best=min(all_final,key=lambda z:z['total'])
    base_name='exact' if E==0 else 'midpoint'; base_rec=dict(recon)[base_name]
    raw=evaluate(src,E,base_name,'g0',base_rec,bytes([0]),lambda a,s:a,9)
    return {'image':f'kodim{idx:02d}','E':E,'pixels':n,'raw_jxl_bytes':raw['total'],'best_bytes':best['total'],
            'raw_jxl_bpsp':8*raw['total']/(3*n),'best_bpsp':8*best['total']/(3*n),
            'saving_vs_raw_jxl_pct':100*(raw['total']-best['total'])/raw['total'],
            'selected':f"{best['rname']}/{best['rep']}/{best['backend']}",'side_bytes':best['side'],
            'payload_bytes':best['payload'],'max_error':best['err'],'final_candidates':len(all_final)}

def main():
    indices=[i for i in range(1,25) if (i-1)%NSHARD==SHARD]; get_data(indices); rows=[]
    for idx in indices:
        for E in (0,1,2,4):
            row=bench_one(idx,E); rows.append(row); print(json.dumps(row),flush=True)
    with open(OUT/'results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (OUT/'meta.json').write_text(json.dumps({'shard':SHARD,'nshard':NSHARD,'indices':indices,'rows':len(rows)},indent=2))

if __name__=='__main__': main()
