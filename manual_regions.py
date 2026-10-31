"""Local, user-guided Japanese OCR and speech balloon geometry. No page-wide detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
from engine import clip_xywh, configure_ocr

Rect = tuple[int,int,int,int]
Point = tuple[int,int]

@dataclass
class LocalRecognition:
    text: str
    confidence: float
    orientation: str
    glyph_boxes: list[Rect]


def _coarse_contours(image: Image.Image, rect: Rect):
    """Return plausible closed contours around a user-selected text rectangle.

    Only inspect a local ROI. Never treat full-page imagery as OCR text.
    """
    W,H=image.size
    x,y,w,h=clip_xywh(rect,W,H)
    radius=max(32,min(260,int(max(w,h)*1.15)))
    x0,y0=max(0,x-radius),max(0,y-radius)
    x1,y1=min(W,x+w+radius),min(H,y+h+radius)
    patch=np.array(image.convert('RGB').crop((x0,y0,x1,y1)))
    gray=cv2.cvtColor(patch,cv2.COLOR_RGB2GRAY)
    if gray.shape[0]<12 or gray.shape[1]<12:
        return []
    # Local ink outlines, optionally joined over small holes.
    dark=cv2.inRange(gray,0,145)
    dark=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(dark,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    result=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        bx,by,bw,bh=cv2.boundingRect(contour)
        if area < max(120, w*h*1.65) or bw>=(x1-x0)*.97 or bh>=(y1-y0)*.97:
            continue
        if bw < w*1.08 or bh < h*1.05:
            continue
        # Text rectangle must lie substantially within contour, not just its center.
        corners=[(x-x0+w*f,y-y0+h*g) for f,g in ((.12,.12),(.88,.12),(.12,.88),(.88,.88),(.5,.5))]
        count=sum(cv2.pointPolygonTest(contour,(float(cx),float(cy)),False)>=0 for cx,cy in corners)
        if count < 4:
            continue
        filled=np.zeros(gray.shape,np.uint8)
        cv2.drawContours(filled,[contour],-1,255,-1)
        inside=gray[filled>0]
        if inside.size<60 or np.quantile(inside,.60)<158:
            continue
        if area / max(bw*bh,1)<.32:
            continue
        margin=min(x-x0-bx, y-y0-by, bx+bw-x+x0-w, by+bh-y+y0-h)
        score=(count*12 + min(20,margin)) / max(1.0,area**.16)
        result.append((score,contour,x0,y0))
    return sorted(result,key=lambda c:c[0],reverse=True)


def suggest_boundary(image: Image.Image, rect: Rect) -> list[Point]:
    """Suggest an editable polygon, or [] when uncertain (rectangle fallback)."""
    candidates=_coarse_contours(image,rect)
    if not candidates:
        return []
    _,contour,x0,y0=candidates[0]
    epsilon=max(1.7,cv2.arcLength(contour,True)*.013)
    simplified=cv2.approxPolyDP(contour,epsilon,True).reshape(-1,2)
    if len(simplified)>48:
        # Constrain handle count for reasonable interactive editing.
        epsilon=cv2.arcLength(contour,True)*.028
        simplified=cv2.approxPolyDP(contour,epsilon,True).reshape(-1,2)
    if len(simplified)<4:
        return []
    return [(int(a+x0),int(b+y0)) for a,b in simplified]


def poly_mask(size: tuple[int,int], polygon: Sequence[Point]) -> np.ndarray:
    w,h=size
    mask=np.zeros((h,w),np.uint8)
    if len(polygon)>=3:
        cv2.fillPoly(mask,[np.asarray(polygon,np.int32)],255)
    return mask


def inner_text_box(polygon: Sequence[Point], fallback: Rect, size: tuple[int,int]) -> Rect:
    """Largest inscribed axis-aligned rectangle, with a small margin from polygon."""
    if len(polygon)<3:
        return clip_xywh(fallback,*size)
    W,H=size
    arr=np.asarray(polygon,dtype=np.int32)
    x,y,w,h=cv2.boundingRect(arr)
    x,y,w,h=clip_xywh((x,y,w,h),W,H)
    if min(w,h)<12:
        return clip_xywh(fallback,W,H)
    scale=min(1,420/max(w,h))
    tw,th=max(1,int(w*scale)),max(1,int(h*scale))
    translated=np.rint((arr-np.array([x,y]))*scale).astype('int32')
    mask=poly_mask((tw,th),translated)
    margin=max(2,int(min(tw,th)*.055))
    mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(margin*2+1,margin*2+1)))
    heights=[0]*tw
    best=(0,0,0,0); bestarea=0
    for yy in range(th):
        st=[]
        for xx in range(tw+1):
            if xx<tw:
                heights[xx]=heights[xx]+1 if mask[yy,xx] else 0
                ht=heights[xx]
            else: ht=0
            start=xx
            while st and st[-1][1]>ht:
                left,hh=st.pop()
                if (xx-left)*hh>bestarea:
                    best=(left,yy-hh+1,xx-left,hh);bestarea=(xx-left)*hh
                start=left
            if not st or st[-1][1]<ht:
                st.append((start,ht))
    if not bestarea:
        return clip_xywh(fallback,W,H)
    a,b,c,d=best
    return clip_xywh((x+int(a/scale),y+int(b/scale),int(c/scale),int(d/scale)),W,H)


def recognize_selection(image: Image.Image, rect: Rect, orientation='both') -> LocalRecognition:
    """Run OCR ONLY on a rectangle the user selected and return word boxes."""
    _,langs=configure_ocr()
    x,y,w,h=clip_xywh(rect,*image.size)
    if w<5 or h<5:
        return LocalRecognition('',0,'horizontal',[])
    crop=image.crop((x,y,x+w,y+h)).convert('RGB')
    scale=min(3.0,max(1.0,900/max(w,h)))
    if scale>1:
        crop=crop.resize((round(w*scale),round(h*scale)),Image.Resampling.LANCZOS)
    candidates=[]
    for lang,psm,label in [('jpn_vert',5,'vertical'),('jpn',6,'horizontal')]:
        if lang not in langs or orientation not in ('both',label):
            continue
        info=pytesseract.image_to_data(crop,lang=lang,config=f'--oem 1 --psm {psm}',output_type=Output.DICT)
        words=[]
        for i,value in enumerate(info['text']):
            value=str(value).strip()
            try: conf=float(info['conf'][i])
            except (ValueError,TypeError): continue
            if not value or conf<0: continue
            xx=int(info['left'][i]); yy=int(info['top'][i]); ww=int(info['width'][i]); hh=int(info['height'][i])
            if min(ww,hh)<2: continue
            words.append((value,conf,xx,yy,ww,hh))
        words.sort(key=(lambda a:(-a[2],a[3])) if label=='vertical' else (lambda a:(a[3],a[2])))
        text=''.join(a[0] for a in words)
        boxes=[clip_xywh((x+round(a[2]/scale),y+round(a[3]/scale),max(1,round(a[4]/scale)),max(1,round(a[5]/scale))),*image.size) for a in words]
        if not text:
            continue
        confidence=sum(a[1] for a in words)/len(words)
        # Mixed OCR output is a suggestion, not a guaranteed transcription.
        score=confidence+min(12,len(text))*.4
        candidates.append((score,LocalRecognition(text,confidence,label,boxes)))
    return max(candidates,key=lambda c:c[0])[1] if candidates else LocalRecognition('',0,'horizontal',[])