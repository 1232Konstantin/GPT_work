"""Conservative, user-approved removal of glyphs within STRICT target rectangles.

Detection is only a suggestion: no image changes until explicit confirmation.
Coordinates are original image pixels. Raster modifications are masked and cropped
so no pixel outside the approved rectangles can ever change.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import cv2
import numpy as np
from PIL import Image

Box = tuple[int,int,int,int]

@dataclass(frozen=True)
class GlyphTarget:
    box: Box


def clip_box(box: Box, size: tuple[int,int]) -> Box:
    x,y,w,h=map(int,box)
    return (max(0,min(size[0],x)),max(0,min(size[1],y)),
            max(0,min(size[0],x+max(0,w))-max(0,min(size[0],x))),
            max(0,min(size[1],y+max(0,h))-max(0,min(size[1],y))))


def _contrast_mask(rgb: np.ndarray) -> np.ndarray:
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w=gray.shape
    sigma=max(3.0,min(11.0,min(h,w)/8.0))
    background=cv2.GaussianBlur(gray,(0,0),sigmaX=sigma)
    threshold=max(17.0,min(35.0,float(np.percentile(np.maximum(0, background.astype(np.int16)-gray.astype(np.int16)),93))*.40))
    mask=(gray.astype(np.float32)<background.astype(np.float32)-threshold).astype(np.uint8)*255
    return mask


def detect_glyph_boxes(image:Image.Image, selection:Box) -> list[Box]:
    x,y,w,h=clip_box(selection,image.size)
    if w<10 or h<10:return []
    rgb=np.asarray(image.crop((x,y,x+w,y+h)).convert('RGB'))
    raw=_contrast_mask(rgb)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(raw,8)
    out=[]
    margin=1
    for i in range(1,n):
        cx,cy,cw,ch,area=map(int,stats[i])
        if area<3 or area>min(900,w*h*.07):continue
        if cw>max(22,w*.43) or ch>max(27,h*.43):continue
        if cx <=margin or cy <=margin or cx+cw>=w-margin or cy+ch>=h-margin:continue
        if cw<=2 and ch<=2:continue
        # Pad for outline/anti-aliasing, stay entirely inside original selection.
        pad=2
        a=max(0,cx-pad);b=max(0,cy-pad)
        c=min(w,cx+cw+pad);d=min(h,cy+ch+pad)
        out.append((x+a,y+b,c-a,d-b))
    # Combine only truly overlapping pieces; do not merge entire text columns.
    merged=[]
    typical=max(8.0,float(np.median([z[3] for z in out])) if out else 8.0)
    for box in sorted(out,key=lambda z:(z[1],z[0])):
        bx,by,bw,bh=box
        touches=[j for j,(xx,yy,ww,hh) in enumerate(merged) if bx < xx+ww and bx+bw > xx and by < yy+hh and by+bh > yy]
        if touches:
            boxes=[box]+[merged[j] for j in touches]
            x0=min(z[0] for z in boxes);y0=min(z[1] for z in boxes)
            x1=max(z[0]+z[2] for z in boxes);y1=max(z[1]+z[3] for z in boxes)
            if (x1-x0 <= max(32,typical*2.3) and y1-y0<=max(27,typical*1.6)):
                for j in reversed(touches):merged.pop(j)
                merged.append((x0,y0,x1-x0,y1-y0))
            else:merged.append(box)
        else:merged.append(box)
    return merged[:150]


def remove_glyphs(image:Image.Image, approved:Iterable[Box]) -> tuple[Image.Image,int]:
    """Never writes outside approved rectangles. Source unchanged.

    Flat-tone background -> median ring colour for detected dark ink, including
    interior anti-alias. Otherwise inpaint from enlarged CONTEXT only, and copy
    repaired pixels back under each target's strict rectangular clip.
    """
    boxes=[clip_box(z,image.size) for z in approved]
    boxes=[z for z in boxes if z[2]>=3 and z[3]>=3]
    if not boxes:return image.copy(),0
    original=np.array(image.convert('RGBA'))
    output=original.copy()
    H,W=original.shape[:2]
    touched=np.zeros((H,W),dtype=np.uint8)
    for x,y,w,h in boxes:
        pad=max(5,min(15,max(w,h)//3))
        lx=max(0,x-pad);ly=max(0,y-pad)
        rx=min(W,x+w+pad);ry=min(H,y+h+pad)
        context=original[ly:ry,lx:rx,:3]
        bx,by=x-lx,y-ly
        # Candidate dark ink vs its own local surround.
        # Every approved rectangle is a self-contained removal mask.
        # Fill *the entire box*, including white halos around black glyphs.
        # The user has explicitly accepted its bounds. No pixels outside move.
        confined=np.zeros(context.shape[:2],dtype=np.uint8)
        confined[by:by+h,bx:bx+w]=255
        ring=np.ones(context.shape[:2],dtype=bool)
        ring[by:by+h,bx:bx+w]=False
        samples=context[ring]
        median=np.median(samples,axis=0) if len(samples) else np.array([255,255,255])
        # Median-filtered external ring discards dark neighbouring letters and
        # isolated highlights, so even a tinted background is not whitened.
        if len(samples)>20:
            delta=np.max(np.abs(samples.astype(np.float32)-median),axis=1)
            kept=samples[delta<=np.percentile(delta,65)]
            if len(kept)>12:samples=kept
        col=np.uint8(np.round(np.median(samples,axis=0)))
        spread=np.percentile(np.max(np.abs(samples.astype(np.float32)-col.astype(np.float32)),axis=1),85) if len(samples)>20 else 0
        if spread<18:
            rendered=np.empty((h,w,3),dtype=np.uint8)
            rendered[:]=col
        else:
            # A robust local colour plane reconstructs gradients without
            # dragging black ink from adjacent glyphs into the repaired area.
            yy,xx=np.mgrid[0:ry-ly,0:rx-lx]
            coords=np.stack((xx[ring],yy[ring],np.ones(int(ring.sum()))),axis=-1).astype(np.float64)
            vals=context[ring].astype(np.float64)
            distances=np.max(np.abs(vals-col),axis=1)
            good=distances<=np.percentile(distances,63)
            if np.count_nonzero(good)<12:
                rendered=np.empty((h,w,3),dtype=np.uint8)
                rendered[:]=col
            else:
                try:
                    plane=np.linalg.lstsq(coords[good],vals[good],rcond=None)[0]
                    jy,jx=np.mgrid[by:by+h,bx:bx+w]
                    query=np.stack((jx,jy,np.ones((h,w))),axis=-1)
                    predicted=np.matmul(query,plane)
                    rendered=np.uint8(np.clip(predicted,0,255))
                except np.linalg.LinAlgError:
                    rendered=np.empty((h,w,3),dtype=np.uint8)
                    rendered[:]=col
        current=output[y:y+h,x:x+w,:3]
        allowed=(touched[y:y+h,x:x+w]==0)
        current[allowed]=rendered[allowed]
        touched[y:y+h,x:x+w][allowed]=255
    out=Image.fromarray(output,'RGBA')
    if image.mode=='RGB':out=out.convert('RGB')
    return out,int(np.count_nonzero(touched))