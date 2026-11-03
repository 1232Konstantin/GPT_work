"""Offline drawing primitives used by the Paint-3D-style editor.

Pixels are edited at native source resolution. Each stroke uses local
patches, alpha compositing and deterministic texture (no fuzzy image scaling).
"""
from __future__ import annotations

import math
import random
from PIL import Image, ImageDraw, ImageFilter
import cv2
import numpy as np

TOOLS = (
    "marker", "calligraphy", "oil", "watercolor", "pixel",
    "pencil", "crayon", "restore", "spray", "fill", "smudge"
)


def _clip(value, maximum):
    return min(max(0, int(round(value))), maximum - 1)


def _dabs(draw, p0, p1, radius, fill, *, ellipse=True, step=1.0):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = max(1, int(math.hypot(dx, dy) / step))
    for i in range(n + 1):
        t = i / n
        x, y = p0[0] + dx * t, p0[1] + dy * t
        if ellipse:
            draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=fill)
        else:
            draw.rectangle((x-radius,y-radius,x+radius,y+radius),fill=fill)


def fill_region(image: Image.Image, point, color, opacity=100, *, tolerance=20) -> Image.Image:
    """Contiguous color fill; tolerance limits it to the tapped color region."""
    original = np.asarray(image.convert("RGBA")).copy()
    h, w = original.shape[:2]
    if not w or not h:
        return image.copy()
    x, y = _clip(point[0],w), _clip(point[1],h)
    work = np.ascontiguousarray(original[:,:,:3].copy())
    flood_mask = np.zeros((h+2,w+2),dtype=np.uint8)
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(work,flood_mask,(x,y),(0,0,0),
                 (tolerance,)*3,(tolerance,)*3,flags=flags)
    selected = flood_mask[1:-1,1:-1] != 0
    if not np.any(selected):
        return image.copy()
    target=np.asarray(color[:3],dtype=np.float32)
    ratio=max(0,min(100,int(opacity)))/100.0
    original[selected,:3] = np.uint8(np.round(
        original[selected,:3].astype(np.float32)*(1-ratio)+target*ratio))
    out=Image.fromarray(original,"RGBA")
    return out.convert("RGB") if image.mode=="RGB" else out


def draw_stroke(image: Image.Image, original: Image.Image, start, end,
                diameter:int, color:tuple[int,int,int], tool="marker",
                opacity=100) -> Image.Image:
    """Return a new PIL image; brush opacity and diameter apply to every tool.

    Eraser/restore recovers the exact corresponding pixels from the source.
    """
    if tool=="fill":
        return fill_region(image,end,color,opacity)
    if tool=="smudge":
        from paint_core import palette_smudge
        smeared=palette_smudge(image,start,end,diameter)
        alpha=max(0,min(100,int(opacity)))/100.0
        mixed=Image.blend(image.convert('RGBA'),smeared.convert('RGBA'),alpha)
        return mixed.convert('RGB') if image.mode=='RGB' else mixed
    if tool not in TOOLS:
        raise ValueError("Unknown tool: "+tool)
    size=max(1,min(180,int(diameter)))
    if tool=="pencil":size=max(1,min(size,18))
    if tool=="pixel":size=max(1,min(size,48))
    r=max(0.5,size/2)
    W,H=image.size
    margin=int(size*2+9)
    left=max(0,int(min(start[0],end[0]))-margin)
    top=max(0,int(min(start[1],end[1]))-margin)
    right=min(W,int(max(start[0],end[0]))+margin+2)
    bottom=min(H,int(max(start[1],end[1]))+margin+2)
    if left>=right or top>=bottom:return image.copy()
    local0=(start[0]-left,start[1]-top)
    local1=(end[0]-left,end[1]-top)
    patch=image.crop((left,top,right,bottom)).convert("RGBA")
    mask=Image.new("L",patch.size,0)
    draw=ImageDraw.Draw(mask)
    if tool=="pixel":
        _dabs(draw,local0,local1,r,255,ellipse=False,step=max(1,r))
    elif tool=="calligraphy":
        # Nib slanted at a fixed angle, as with a calligraphic pen.
        dx,dy=local1[0]-local0[0],local1[1]-local0[1]
        steps=max(1,int(math.hypot(dx,dy)/max(1,size/5)))
        for i in range(steps+1):
            t=i/steps
            x,y=local0[0]+dx*t,local0[1]+dy*t
            draw.line((x-r*.65,y+r*.7,x+r*.65,y-r*.7),
                      fill=255,width=max(1,int(size*.38)))
    elif tool=="spray":
        rng=random.Random((int(start[0])*733+int(start[1])*1511+
                           int(end[0])*593+int(end[1])*1223+size*37)&0xffffffff)
        dx,dy=local1[0]-local0[0],local1[1]-local0[1]
        samples=max(15,int(math.hypot(dx,dy)*2)+int(size*.7))
        for i in range(samples):
            t=i/max(1,samples-1)
            cx,cy=local0[0]+dx*t,local0[1]+dy*t
            theta=rng.random()*math.tau
            distance=math.sqrt(rng.random())*r
            px,py=cx+math.cos(theta)*distance,cy+math.sin(theta)*distance
            draw.ellipse((px-1,py-1,px+1,py+1),fill=255)
    else:
        draw.line((*local0,*local1),fill=255,width=max(1,size),joint="curve")
        _dabs(draw,local0,local1,r,255,step=max(1,r*.65))
    if tool=="watercolor":
        mask=mask.filter(ImageFilter.GaussianBlur(max(1,r*.55)))
    elif tool=="crayon":
        # Sparse deterministic grain, retained over a solid crayon stroke.
        ar=np.asarray(mask).copy()
        yy,xx=np.indices(ar.shape)
        noise=((xx*73+yy*157+(xx*yy)%89)%23)
        ar[noise<5]=0
        mask=Image.fromarray(ar,"L")
    elif tool=="oil":
        ar=np.asarray(mask).copy()
        yy,xx=np.indices(ar.shape)
        ar[(xx*31+yy*11)%17==0]=np.uint8(ar[(xx*31+yy*11)%17==0]*.55)
        mask=Image.fromarray(ar,"L")
    if tool=="restore":
        source=original.crop((left,top,right,bottom)).convert("RGBA")
        alpha=max(0,min(100,int(opacity)))/100.0
        patch.paste(source,(0,0),mask.point(lambda v:int(round(v*alpha))))
    else:
        alpha=max(0,min(100,int(opacity)))/100.0
        if tool=="watercolor":alpha*=.52
        rgb=tuple(int(x) for x in color[:3])
        overlay=Image.new("RGBA",patch.size,(*rgb,0))
        overlay.putalpha(mask.point(lambda v:int(round(v*alpha))))
        patch=Image.alpha_composite(patch,overlay)
    result=image.copy()
    result.paste(patch,(left,top))
    return result.convert("RGB") if image.mode=="RGB" else result
