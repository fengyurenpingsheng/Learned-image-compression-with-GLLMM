#!/usr/bin/env python3
import csv, json, pathlib, struct, subprocess, tempfile, time, urllib.request, zlib
import numpy as np
from PIL import Image

ROOT=pathlib.Path(__file__).resolve().parent
DATA=ROOT/'kodak24'; OUT=ROOT/'kodak24_rrc_results'; DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
URLS=(
 'https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png',
 'http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png',
)
TRANSFORMS=list(range(5))

def get_data():
  for i in range(1,25):
    p=DATA/f'kodim{i:02d}.png'
    if p.exists(): continue
    error=None
    for u in URLS:
      try: urllib.request.urlretrieve(u.format(i),p); error=None; break
      except Exception as e: error=e
    if error is not None: raise error

def entropy(a):
  h=np.bincount(a.ravel(),minlength=256); p=h[h>0]/a.size
  return float(-(p*np.log2(p)).sum())

def fwd_block(x,t):
  q=x.astype(np.int16); r,g,b=q[...,0],q[...,1],q[...,2]
  if t==0: y=(r,g,b)
  elif t==1: y=(g,(r-g)&255,(b-g)&255)
  elif t==2: y=(r,(g-r)&255,(b-r)&255)
  elif t==3: y=(b,(r-b)&255,(g-b)&255)
  else: y=(g,(r-g)&255,(b-((r+g+1)//2))&255)
  return np.stack(y,-1).astype(np.uint8)

def inv_block(y,t):
  q=y.astype(np.int16); a,c,d=q[...,0],q[...,1],q[...,2]
  if t==0: r,g,b=a,c,d
  elif t==1: g=a; r=(c+g)&255; b=(d+g)&255
  elif t==2: r=a; g=(c+r)&255; b=(d+r)&255
  elif t==3: b=a; r=(c+b)&255; g=(d+b)&255
  else: g=a; r=(c+g)&255; b=(d+((r+g+1)//2))&255
  return np.stack((r,g,b),-1).astype(np.uint8)

def global_transform(x,t): return fwd_block(x,t), bytes([t])
def global_inverse(y,side): return inv_block(y,side[0])

def local_transform(x,bs):
  h,w,_=x.shape; y=np.empty_like(x); ids=[]
  for yy in range(0,h,bs):
    for xx in range(0,w,bs):
      blk=x[yy:min(h,yy+bs),xx:min(w,xx+bs)]
      best=min(TRANSFORMS,key=lambda t: sum(entropy(fwd_block(blk,t)[...,c]) for c in range(3)))
      y[yy:min(h,yy+bs),xx:min(w,xx+bs)]=fwd_block(blk,best); ids.append(best)
  side=struct.pack('<HHH',bs,h,w)+zlib.compress(bytes(ids),9)
  return y,side

def local_inverse(y,side):
  bs,h,w=struct.unpack('<HHH',side[:6]); ids=zlib.decompress(side[6:]); x=np.empty_like(y); k=0
  for yy in range(0,h,bs):
    for xx in range(0,w,bs):
      x[yy:min(h,yy+bs),xx:min(w,xx+bs)]=inv_block(y[yy:min(h,yy+bs),xx:min(w,xx+bs)],ids[k]); k+=1
  return x

def run(cmd): subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
def encode_jxl(a,effort=9):
  with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td); png=td/'a.png'; out=td/'a.jxl'; Image.fromarray(a).save(png)
    run(['cjxl',str(png),str(out),'-d','0','-e',str(effort),'--num_threads=1'])
    return out.read_bytes()
def decode_jxl(payload):
  with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td); p=td/'a.jxl'; o=td/'o.png'; p.write_bytes(payload); run(['djxl',str(p),str(o),'--num_threads=1']); return np.asarray(Image.open(o).convert('RGB'),np.uint8)
def encode_webp(a):
  with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td); p=td/'a.png'; o=td/'a.webp'; Image.fromarray(a).save(p); run(['cwebp','-quiet','-lossless','-z','9','-m','6',str(p),'-o',str(o)]); return o.read_bytes()
def decode_webp(payload):
  with tempfile.TemporaryDirectory() as td:
    td=pathlib.Path(td); p=td/'a.webp'; o=td/'o.png'; p.write_bytes(payload); run(['dwebp','-quiet',str(p),'-o',str(o)]); return np.asarray(Image.open(o).convert('RGB'),np.uint8)

def candidates(x):
  yield 'raw-jxl',x,b'',global_inverse,encode_jxl,decode_jxl
  yield 'raw-webp',x,b'',global_inverse,encode_webp,decode_webp
  for t in TRANSFORMS[1:]:
    y,s=global_transform(x,t); yield f'global{t}-jxl',y,s,global_inverse,encode_jxl,decode_jxl
  for bs in (32,64,128):
    y,s=local_transform(x,bs); yield f'local{bs}-jxl',y,s,local_inverse,encode_jxl,decode_jxl

def quantize(x,E):
  if E==0:return x.copy()
  s=2*E+1; return np.minimum(255,(x.astype(np.int16)//s)*s+E).astype(np.uint8)

def bench():
  get_data(); rows=[]
  for idx in range(1,25):
    src=np.asarray(Image.open(DATA/f'kodim{idx:02d}.png').convert('RGB'),np.uint8); n=src.shape[0]*src.shape[1]
    for E in (0,1,2,4):
      target=quantize(src,E); best=None; allc=[]
      for name,car,side,inv,enc,dec in candidates(target):
        t=time.time(); payload=enc(car); total=12+len(side)+len(payload); rec_car=dec(payload)
        rec=rec_car if name.startswith('raw-') else inv(rec_car,side)
        err=int(np.abs(src.astype(np.int16)-rec.astype(np.int16)).max()); exact=bool(np.array_equal(target,rec))
        if not exact or err>E: raise RuntimeError((idx,E,name,err,exact))
        item=(total,name,time.time()-t,len(side),len(payload)); allc.append(item)
        if best is None or item<best: best=item
      base=next(v for v in allc if v[1]=='raw-jxl')
      row={'image':f'kodim{idx:02d}','E':E,'pixels':n,'raw_jxl_bytes':base[0],'best_bytes':best[0],'raw_jxl_bpsp':8*base[0]/(3*n),'best_bpsp':8*best[0]/(3*n),'saving_vs_raw_jxl_pct':100*(base[0]-best[0])/base[0],'selected':best[1],'side_bytes':best[3],'payload_bytes':best[4],'max_error':err}
      rows.append(row); print(row,flush=True)
  with open(OUT/'kodak24_results.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
  summary={}
  for E in (0,1,2,4):
    rr=[r for r in rows if r['E']==E]; b=sum(r['raw_jxl_bytes'] for r in rr); q=sum(r['best_bytes'] for r in rr); p=sum(r['pixels'] for r in rr)
    summary[str(E)]={'images':24,'raw_jxl_bpsp':8*b/(3*p),'best_bpsp':8*q/(3*p),'saving_vs_raw_jxl_pct':100*(b-q)/b,'wins':sum(r['best_bytes']<r['raw_jxl_bytes'] for r in rr),'ties':sum(r['best_bytes']==r['raw_jxl_bytes'] for r in rr),'selected':{n:sum(r['selected']==n for r in rr) for n in sorted(set(r['selected'] for r in rr))}}
  (OUT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': bench()
