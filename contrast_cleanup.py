"""Offline contrast-based text removal shared by the manga editor.

Two independent stages:
    segment_ink: discover contrast pixels, optional outline -> binary mask
    repair_background: reconstruct only inside the mask, never on its complement

All colour operations work on the original pixel grid; no OCR/recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from PIL import Image

Mode = Literal['dark', 'light', 'outline', 'outline_dark']


@dataclass
class Segmentation:
    mask: np.ndarray
    dark_seeds: np.ndarray
    light_seeds: np.ndarray
    background: np.ndarray


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _kernel(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _contrast(rgb: np.ndarray, *, estimated_bg: np.ndarray | None = None):
    # L is 0..255, local median rejects narrow strokes while retaining gradients.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    h, w = lab.shape[:2]
    k = min(35, max(9, _odd(min(h, w) // 3)))
    if estimated_bg is None:
        bg_lab = cv2.medianBlur(lab, k)
    else:
        bg_lab = cv2.cvtColor(estimated_bg, cv2.COLOR_RGB2LAB)
    lum = lab[:, :, 0].astype(np.int16)
    base = bg_lab[:, :, 0].astype(np.int16)
    # Blackhat/Tophat recover narrow lines the median may fail to remove.
    scores_dark = [np.maximum(0,base-lum)]
    scores_light = [np.maximum(0,lum-base)]
    for k in (7, 13, 21):
        if k >= min(h,w):continue
        closed=cv2.morphologyEx(lum.astype(np.uint8),cv2.MORPH_CLOSE,_kernel(k))
        opened=cv2.morphologyEx(lum.astype(np.uint8),cv2.MORPH_OPEN,_kernel(k))
        scores_dark.append(np.maximum(0,closed.astype(np.int16)-lum))
        scores_light.append(np.maximum(0,lum-opened.astype(np.int16)))
    return np.maximum.reduce(scores_dark), np.maximum.reduce(scores_light),cv2.cvtColor(bg_lab,cv2.COLOR_LAB2RGB)


def _components(raw: np.ndarray, selection:tuple[int,int] | None=None, *, min_area=2,max_fraction=.12):
    h,w=raw.shape
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw.astype('uint8'),8)
    kept=np.zeros_like(raw,dtype=np.uint8)
    for i in range(1,count):
        x,y,bw,bh,area=map(int,stats[i])
        if area<min_area or area>h*w*max_fraction:continue
        # User-selected background regions can contain real strong contours,
        # so reject components touching ROI border (unless this is a glyph crop).
        if selection is not None and (x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1):continue
        if bw>w*.8 or bh>h*.8:continue
        kept[labels==i]=255
    return kept


def segment_ink(image:Image.Image, region:tuple[int,int,int,int], mode:Mode='dark',
                *, dark_threshold:int=22, light_threshold:int=16,
                outline_color:tuple[int,int,int] | None=None, grow:int=1):
    rgb=np.asarray(image.convert('RGB'))
    H,W=rgb.shape[:2]
    xx,yy,ww,hh=(int(t) for t in region)
    x0,y0=max(0,xx),max(0,yy)
    x1,y1=min(W,xx+ww),min(H,yy+hh)
    if x1-x0<10 or y1-y0<10:
        z=np.zeros((H,W),dtype=np.uint8)
        return Segmentation(z,z.copy(),z.copy(),rgb.copy())
    crop=rgb[y0:y1,x0:x1]
    score_dark,score_light,bg=_contrast(crop)
    lab = cv2.cvtColor(crop,cv2.COLOR_RGB2LAB)
    L=lab[:,:,0].astype(np.float32)
    base=float(np.median(L))
    chroma_spread=(crop.max(axis=2).astype(np.float32)
                   -crop.min(axis=2).astype(np.float32))
    # The selection is background-dominated. Most ink is more neutral and
    # several tens of units darker than the typical local background.
    dark_global = ((L < base - 32) & (chroma_spread < 60))
    dark_local = ((score_dark >= dark_threshold) & (L < base-15)
                  & (chroma_spread < 55))
    dark=_components(dark_global | dark_local, (x0,y0), max_fraction=.27)
    bright_global = ((chroma_spread < 43) &
                     (L >= max(185.0,base + (42 if base < 130 else -22))))
    # Optional user-selected halo colour, for non-white/coloured outline.
    if outline_color is not None:
        ref_lab=cv2.cvtColor(np.uint8([[list(outline_color)]]),
                             cv2.COLOR_RGB2LAB)[0,0].astype(np.float32)
        diff=np.linalg.norm(lab.astype(np.float32)-ref_lab,axis=2)
        local_bg_lab=cv2.cvtColor(bg,cv2.COLOR_RGB2LAB).astype(np.float32)
        background_distance=np.linalg.norm(lab.astype(np.float32)-local_bg_lab,axis=2)
        bright_global |= (diff < 22) & (background_distance > light_threshold)
    light=_components(bright_global, (x0,y0), max_fraction=.27)

    if mode=='dark':
        core=dark
    elif mode=='light':
        core=light
    elif mode=='outline':
        if base > 130:
            # Light/white border around dark ink on coloured light background.
            # A halo is only trusted if it is close to a dark glyph seed.
            close_to_ink=cv2.dilate(dark,_kernel(11))>0
            rim=cv2.bitwise_and(light,np.uint8(close_to_ink)*255)
            core=cv2.bitwise_or(dark,rim)
        else:
            # White halo around almost black letters on dark background. Black
            # fill may have no contrast at all; close rim gaps and fill loops.
            rim=cv2.morphologyEx(light,cv2.MORPH_CLOSE,_kernel(5))
            contours,_=cv2.findContours(rim,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            filled=np.zeros_like(rim)
            for contour in contours:
                cx,cy,cw,ch=cv2.boundingRect(contour)
                if cw > crop.shape[1]*.75 or ch > crop.shape[0]*.75:
                    continue
                cv2.drawContours(filled,[contour],-1,255,-1)
            close_to_rim=cv2.dilate(filled,_kernel(7))>0
            core=cv2.bitwise_or(filled,
                cv2.bitwise_and(dark,np.uint8(close_to_rim)*255))
    elif mode=='outline_dark':
        # Dark outline around bright lettering. In light panels the interior
        # of a white glyph may be almost the background colour: fill enclosed
        # dark contours instead of assuming the inner pixels have contrast.
        if base > 130:
            rim=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,_kernel(5))
            contours,_=cv2.findContours(rim,cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
            enclosed=np.zeros_like(rim)
            for contour in contours:
                cx,cy,cw,ch=cv2.boundingRect(contour)
                if (cw>crop.shape[1]*.75 or ch>crop.shape[0]*.75):
                    continue
                cv2.drawContours(enclosed,[contour],-1,255,-1)
            near=cv2.dilate(rim,_kernel(9))>0
            core=cv2.bitwise_or(
                enclosed,cv2.bitwise_and(light,np.uint8(near)*255))
        else:
            # Very dark outline on dark background: use light glyph body as
            # the seed and keep only dark strokes attached to that glyph.
            near=cv2.dilate(light,_kernel(11))>0
            core=cv2.bitwise_or(light,
                cv2.bitwise_and(dark,np.uint8(near)*255))
    else:raise ValueError(mode)
    core=_components(core>0,(x0,y0),min_area=2,max_fraction=.45)
    # Include antialiased stroke edges but never grow outside selection.
    if grow>0:core=cv2.dilate(core,_kernel(2*grow+1))
    allmask=np.zeros((H,W),dtype=np.uint8)
    allmask[y0:y1,x0:x1]=core
    d=np.zeros_like(allmask);l=np.zeros_like(allmask)
    d[y0:y1,x0:x1]=dark;l[y0:y1,x0:x1]=light
    allbg=rgb.copy();allbg[y0:y1,x0:x1]=bg
    return Segmentation(allmask,d,l,allbg)


def _poly_surface(crop:np.ndarray,mask:np.ndarray,degree:int=1):
    """Fit robust per-channel plane/quadratic using unmasked neighbouring pixels.

    Polynomial regression is restricted to local *background*, not global page.
    Trim largest colour residuals to avoid dark ink/illustration border.
    """
    h,w=crop.shape[:2]
    yy,xx=np.mgrid[:h,:w]
    # Normalize coordinates; improves conditioning on thin crop.
    u=(xx-(w-1)/2)/max(1,w)
    v=(yy-(h-1)/2)/max(1,h)
    basis=[np.ones_like(u),u,v]
    if degree==2:basis += [u*u,u*v,v*v]
    a=np.stack(basis,axis=-1)
    # only samples near masked region, without masked pixels
    neighbours=cv2.dilate(mask.astype('uint8'),_kernel(min(29,_odd(max(7,min(h,w)//3)))))>0
    samples=(mask==0)&neighbours
    if samples.sum()<len(basis)*5:samples=mask==0
    if samples.sum()<len(basis)*5:return None
    X=a[samples].astype(np.float64);Y=crop[samples].astype(np.float64)
    take=np.ones(len(X),bool)
    for it in range(4):
        coef=np.linalg.lstsq(X[take],Y[take],rcond=None)[0]
        resid=np.max(np.abs(X@coef-Y),axis=-1)
        take=resid<=max(6,np.percentile(resid,77))
        if take.sum()<len(basis)*5:break
    return np.uint8(np.clip(a@coef,0,255))


def repair_background(image:Image.Image, mask:np.ndarray, method:Literal['telea','plane','quadratic']='plane'):
    if image.mode not in ('RGB', 'RGBA'):
        raise ValueError('Input must be RGB or RGBA (convert before using)')
    arr=np.asarray(image.convert('RGB'))
    if mask.shape != arr.shape[:2]:
        raise ValueError('Mask size must match original image')
    if not np.any(mask):return image.copy()
    yy,xx=np.nonzero(mask)
    # Halo with intact surrounding pixels retained in cropped inpaint input.
    x0,y0=max(0,int(xx.min())-17),max(0,int(yy.min())-17)
    x1,y1=min(arr.shape[1],int(xx.max())+18),min(arr.shape[0],int(yy.max())+18)
    crop=arr[y0:y1,x0:x1];m=mask[y0:y1,x0:x1].astype(np.uint8)
    if method=='telea':
        reconstructed=cv2.inpaint(crop,m,3,cv2.INPAINT_TELEA)
    else:
        reconstructed=_poly_surface(crop,m,1 if method=='plane' else 2)
        if reconstructed is None:reconstructed=cv2.inpaint(crop,m,3,cv2.INPAINT_TELEA)
    result=arr.copy()
    part=result[y0:y1,x0:x1]
    part[m>0]=reconstructed[m>0]
    if image.mode == 'RGBA':
        original=np.asarray(image).copy()
        original[mask>0,:3]=result[mask>0]
        return Image.fromarray(original,'RGBA')
    return Image.fromarray(result,'RGB')


def mask_components(mask: np.ndarray, region: tuple[int, int, int, int], *, max_count: int=350):
    """Map connected mask strokes to UI rectangles without replacing the pixel mask.

    Return clipped ROI, candidate rectangles in *image* coordinates, their label
    IDs, and the local label array. Only enabled labels are applied later.
    """
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Expected 8-bit 2D binary mask")
    H, W = mask.shape
    x, y, w, h = (int(v) for v in region)
    x0 = max(0, min(W, x))
    y0 = max(0, min(H, y))
    x1 = max(x0, min(W, x+max(0,w)))
    y1 = max(y0, min(H, y+max(0,h)))
    rect = (x0, y0, x1-x0, y1-y0)
    if x1==x0 or y1==y0:
        return rect, [], [], np.zeros((y1-y0,x1-x0),dtype=np.int32)
    source = np.uint8(mask[y0:y1, x0:x1]>0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(source, 8)
    items = []
    for label in range(1,n):
        cx,cy,cw,ch,area = [int(v) for v in stats[label]]
        if area < 2:
            continue
        items.append((area, (x0+cx,y0+cy,cw,ch),label))
    items.sort(key=lambda z:(-z[0], z[1][1],z[1][0]))
    selected = items[:max_count]
    return rect, [b for _,b,_ in selected], [i for _,_,i in selected], labels


def confirmed_mask(original_mask: np.ndarray, rect:tuple[int,int,int,int], labels:np.ndarray,
                   label_ids:list[int|None], boxes:list[tuple[int,int,int,int]],
                   enabled:list[bool]):
    """Build the *exact* writable mask from enabled components/manual rectangles.

    No unconfirmed pixels and no pixels outside the user selection may change.
    A manual rectangle acts as an explicitly approved mask region.
    """
    if not(len(label_ids)==len(boxes)==len(enabled)):
        raise ValueError('Mismatch in candidate lists')
    x,y,w,h=rect
    if labels.shape!=(h,w) or original_mask.ndim != 2:
        raise ValueError('Incompatible segmentation geometry')
    out=np.zeros_like(original_mask,dtype=np.uint8)
    region=out[y:y+h,x:x+w]
    keep=[key for key,on in zip(label_ids,enabled) if key is not None and on]
    if keep:
        region[np.isin(labels,keep)] = 255
    for box,key,on in zip(boxes,label_ids,enabled):
        if not on or key is not None:continue
        bx,by,bw,bh=box
        left=max(0,bx-x);top=max(0,by-y)
        right=min(w,bx-x+bw);bottom=min(h,by-y+bh)
        if right>left and bottom>top:
            region[top:bottom,left:right]=255
    return out