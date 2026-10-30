"""Offline manga page OCR, ink removal and typesetting. No network calls."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import os
import re
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pytesseract
from pytesseract import Output

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}


@dataclass
class Region:
    id: int
    ink: tuple[int, int, int, int]  # x,y,w,h - OCR ink to erase
    box: tuple[int, int, int, int]  # x,y,w,h - translation layout area
    source: str = ''
    translated: str = ''
    confidence: float = 0.0
    orientation: str = 'horizontal'
    action: str = 'translate'  # translate | hide | keep
    confirmed: bool = False  # User has checked this OCR area manually
    balloon: bool = False  # False = free-standing text, with no speech bubble


@dataclass
class Page:
    path: str
    regions: list[Region] = field(default_factory=list)


def open_image(path: str) -> Image.Image:
    with Image.open(path) as inp:
        image = ImageOps.exif_transpose(inp)
        if image.mode in ('RGBA', 'LA') or ('transparency' in image.info):
            return image.convert('RGBA').copy()
        return image.convert('RGB').copy()


def find_tesseract() -> str | None:
    """Prefer bundled engine, then conventional Windows install, then PATH."""
    bases = [Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent)),
             Path(__file__).resolve().parent, Path(sys.executable).resolve().parent]
    for base in bases:
        for candidate in (base / 'vendor' / 'tesseract' / 'tesseract.exe',
                          base / 'tesseract' / 'tesseract.exe'):
            if candidate.is_file():
                return str(candidate)
    if os.name == 'nt':
        for candidate in (Path(os.environ.get('PROGRAMFILES', 'C:/Program Files')) / 'Tesseract-OCR' / 'tesseract.exe',
                          Path(os.environ.get('PROGRAMFILES(X86)', 'C:/Program Files (x86)')) / 'Tesseract-OCR' / 'tesseract.exe'):
            if candidate.is_file():
                return str(candidate)
    import shutil
    return shutil.which('tesseract')


def configure_ocr() -> tuple[str, set[str]]:
    engine = find_tesseract()
    if not engine:
        raise RuntimeError('Не найден Tesseract. Установите его с jpn и jpn_vert либо поместите в vendor/tesseract/.')
    pytesseract.pytesseract.tesseract_cmd = engine
    # If a portable installation is present, explicitly point at its models.
    tessdata = Path(engine).parent / 'tessdata'
    if tessdata.is_dir():
        os.environ['TESSDATA_PREFIX'] = str(tessdata)
    langs = set(pytesseract.get_languages(config=''))
    if not ({'jpn', 'jpn_vert'} & langs):
        raise RuntimeError('Не найдены языковые модели jpn.traineddata и/или jpn_vert.traineddata.')
    return engine, langs


def clip_xywh(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v) for v in box)
    x1, y1 = max(0, min(width, x)), max(0, min(height, y))
    x2, y2 = max(x1, min(width, x + max(0, w))), max(y1, min(height, y + max(0, h)))
    return x1, y1, x2-x1, y2-y1


def _rect_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter
    return inter / union if union else 0.0


def _join_ocr_words(words: list[str], vertical: bool) -> str:
    words = [str(w).strip() for w in words if str(w).strip()]
    if vertical:
        return ''.join(words).replace(' ', '')
    result = ''
    for word in words:
        if result and result[-1].isascii() and word[0].isascii() and result[-1].isalnum() and word[0].isalnum():
            result += ' '
        result += word
    return result


def _ocr_pass(image: Image.Image, language: str, psm: int, scale: float) -> list[Region]:
    input_image = image
    if scale != 1.0:
        input_image = image.resize((max(1, round(image.width*scale)), max(1, round(image.height*scale))), Image.Resampling.LANCZOS)
    data = pytesseract.image_to_data(input_image, lang=language, config=f'--oem 1 --psm {psm}', output_type=Output.DICT)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, word in enumerate(data['text']):
        if not str(word).strip():
            continue
        try:
            conf = float(data['conf'][i])
        except (ValueError, TypeError):
            continue
        if conf < 15:
            continue
        key = (int(data['block_num'][i]), int(data['par_num'][i]), int(data['line_num'][i]))
        groups.setdefault(key, []).append(i)
    vertical = language.endswith('_vert')
    found = []
    for inds in groups.values():
        inds.sort(key=(lambda i: (-data['left'][i], data['top'][i])) if vertical else (lambda i: (data['left'][i], data['top'][i])))
        text = _join_ocr_words([data['text'][i] for i in inds], vertical)
        if not text or not any(c.isalnum() for c in text):
            continue
        l = min(data['left'][i] for i in inds)
        t = min(data['top'][i] for i in inds)
        r = max(data['left'][i]+data['width'][i] for i in inds)
        b = max(data['top'][i]+data['height'][i] for i in inds)
        rect = clip_xywh((round(l/scale), round(t/scale), round((r-l)/scale), round((b-t)/scale)), image.width, image.height)
        if rect[2] < 3 or rect[3] < 3:
            continue
        confidence = sum(float(data['conf'][i]) for i in inds)/len(inds)
        found.append(Region(0, rect, rect, text, confidence=confidence,
                            orientation='vertical' if vertical else 'horizontal'))
    return found


def estimate_layout_box(image: Image.Image, ink: tuple[int, int, int, int], vertical: bool = False) -> tuple[int, int, int, int]:
    """Find a likely light-filled balloon around recognized ink; conservative fallback."""
    x, y, w, h = ink
    W, H = image.size
    if w <= 0 or h <= 0:
        return ink
    # Process a nearby crop only to avoid huge connected-component matrices.
    radius = max(80, min(600, round(max(w,h)*1.5)))
    x0, y0 = max(0, x-radius), max(0, y-radius)
    x1, y1 = min(W, x+w+radius), min(H, y+h+radius)
    rgb = np.asarray(image.convert('RGB').crop((x0,y0,x1,y1)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Close holes (letter strokes) in white background. Solid ink outline remains.
    candidate = (gray >= 177).astype('uint8') * 255
    k = max(3, min(13, (round(min(W,H)/250)*2+1)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    # Inspect white pixels around original ink. The ink itself can be dark.
    ax0, ay0 = max(0,x-x0-5), max(0,y-y0-5)
    ax1, ay1 = min(x1-x0,x-x0+w+5), min(y1-y0,y-y0+h+5)
    samples = labels[ay0:ay1, ax0:ax1]
    if samples.size:
        ids, counts = np.unique(samples[samples != 0], return_counts=True)
        for idx in ids[np.argsort(-counts)]:
            ix, iy, iw, ih, area = (int(q) for q in stats[int(idx)])
            # Reject the page background and implausibly small regions.
            if area < w*h*0.5 or iw*ih > W*H*0.24 or iw*ih > (x1-x0)*(y1-y0)*0.80:
                continue
            if iw < max(w*1.08, w+6) or ih < max(h*0.85, h-4):
                continue
            if ix > x-x0+4 or iy > y-y0+4 or ix+iw < x-x0+w-4 or iy+ih < y-y0+h-4:
                continue
            inset = min(12, max(4, round(min(iw,ih)*0.06)))
            result = clip_xywh((x0+ix+inset, y0+iy+inset, iw-2*inset, ih-2*inset), W, H)
            if result[2] >= w*0.9 and result[3] >= h*0.75:
                return result
    # Fallback: expand more in width for original Japanese vertical lettering.
    pad_x = max(9, round(w*(0.95 if vertical else 0.22)))
    pad_y = max(7, round(h*(0.12 if vertical else 0.27)))
    return clip_xywh((x-pad_x,y-pad_y,w+2*pad_x,h+2*pad_y), W,H)



def _balloon_candidates(image: Image.Image) -> list[tuple[tuple[int,int,int,int], np.ndarray]]:
    """Find likely closed, lightly filled speech balloons using their dark outline."""
    work = image.convert('RGB')
    scale = min(1.0, 2200/max(work.size))
    if scale != 1.0:
        work = work.resize((round(work.width*scale), round(work.height*scale)),Image.Resampling.BILINEAR)
    arr=np.asarray(work)
    gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
    dark=(gray<145).astype(np.uint8)*255
    contours,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    W,H=gray.shape[1],gray.shape[0]
    candidates=[]
    for cnt in contours:
        x,y,w,h=cv2.boundingRect(cnt)
        if w<70*scale or h<55*scale or w*h > W*H*.85:
            continue
        area=cv2.contourArea(cnt)
        if area<.35*w*h:
            continue
        roi=gray[y:y+h,x:x+w]
        if float((roi>178).mean()) < .53:
            continue
        original=clip_xywh((round(x/scale),round(y/scale),round(w/scale),round(h/scale)),image.width,image.height)
        contour_full=np.rint(cnt.astype(np.float64)/scale).astype(np.int32)
        candidates.append((original,contour_full))
    candidates.sort(key=lambda t:t[0][2]*t[0][3],reverse=True)
    return candidates[:100]



def _inscribed_box(outline: tuple[int,int,int,int], contour: np.ndarray) -> tuple[int,int,int,int] | None:
    """Largest axis-aligned rectangle safely inside a speech balloon, not its bbox.

    Downsample first so this stays fast on high-resolution pages.
    """
    x,y,w,h=outline
    scale=min(1.0,480/max(w,h))
    tw,th=max(1,round(w*scale)),max(1,round(h*scale))
    if tw<12 or th<12:
        return None
    mask=np.zeros((th,tw),dtype=np.uint8)
    pts=contour.astype(np.float64).copy()
    pts[:,:,0]=(pts[:,:,0]-x)*scale
    pts[:,:,1]=(pts[:,:,1]-y)*scale
    cv2.fillPoly(mask,[np.rint(pts).astype('int32')],255)
    radius=max(3,min(14,round(min(tw,th)*.06)))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    mask=cv2.erode(mask,k,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    heights=[0]*tw
    best_area=0
    best=None
    for row in range(th):
        for i in range(tw):
            heights[i]=heights[i]+1 if mask[row,i] else 0
        stack=[]
        for i in range(tw+1):
            current=heights[i] if i<tw else 0
            start=i
            while stack and stack[-1][1]>current:
                sx,bar_h=stack.pop()
                area=(i-sx)*bar_h
                if area>best_area:
                    best_area=area
                    best=(sx,row-bar_h+1,i-sx,bar_h)
                start=sx
            if not stack or stack[-1][1]<current:
                stack.append((start,current))
    if best is None:
        return None
    bx,by,bw,bh=best
    # Rounding inward keeps the reconstructed rectangle within the outline.
    return (x+round(bx/scale)+2,y+round(by/scale)+2,
            max(0,round(bw/scale)-4),max(0,round(bh/scale)-4))

def _ocr_balloon(image: Image.Image, outline: tuple[int,int,int,int], contour: np.ndarray, mode: str, langs: set[str]) -> Region | None:
    x,y,w,h=outline
    # Keep the black outline out of the OCR input; it can make OCR page segmentation fail.
    mx=max(6,round(w*.055))
    my=max(6,round(h*.075))
    layout=_inscribed_box(outline,contour)
    if not layout or layout[2]<20 or layout[3]<20:
        layout=clip_xywh((x+mx,y+my,w-2*mx,h-2*my),image.width,image.height)
    # Mask away everything outside the balloon and its black contour before OCR.
    # Rectangular crops alone let OCR mistake curved borders for Japanese letters.
    roi=clip_xywh((x,y,w,h),image.width,image.height)
    if roi[2]<20 or roi[3]<20:
        return None
    pixels=np.asarray(image.convert('RGB').crop((roi[0],roi[1],roi[0]+roi[2],roi[1]+roi[3]))).copy()
    mask=np.zeros((roi[3],roi[2]),dtype=np.uint8)
    local=contour.copy()
    local[:,:,0]-=roi[0]
    local[:,:,1]-=roi[1]
    cv2.drawContours(mask,[local],-1,255,-1)
    radius=max(5,min(22,round(min(w,h)*.07)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1))
    mask=cv2.erode(mask,kernel,iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    pixels[mask==0]=(255,255,255)
    part=Image.fromarray(pixels,'RGB')
    scale=min(2.0, max(.4, 1250/max(roi[2],roi[3])))
    variants=[]
    if mode in ('both','horizontal') and 'jpn' in langs:
        variants.append(('jpn',11))
    if mode in ('both','vertical') and 'jpn_vert' in langs:
        variants.append(('jpn_vert',5))
    options=[]
    for lang,psm in variants:
        pieces=_ocr_pass(part,lang,psm,scale)
        if not pieces:
            continue
        # Recognized text within one bubble is treated as one translation unit.
        pieces.sort(key=(lambda r:(-r.ink[0],r.ink[1])) if lang.endswith('_vert') else (lambda r:(r.ink[1],r.ink[0])))
        source=''.join(r.source for r in pieces)
        conf=sum(r.confidence*max(1,len(r.source)) for r in pieces)/sum(max(1,len(r.source)) for r in pieces)
        left=min(r.ink[0] for r in pieces)+roi[0]
        top=min(r.ink[1] for r in pieces)+roi[1]
        right=max(r.ink[0]+r.ink[2] for r in pieces)+roi[0]
        bottom=max(r.ink[1]+r.ink[3] for r in pieces)+roi[1]
        ink=clip_xywh((left,top,right-left,bottom-top),image.width,image.height)
        if ink[2]<3 or ink[3]<3:
            continue
        option=Region(0,ink,layout,source,confidence=conf,
                      orientation='vertical' if lang.endswith('_vert') else 'horizontal',
                      balloon=True)
        # Recognition quality / volume, penalize low confidence.
        score=conf * min(10,max(1,len(source)))**.28
        options.append((score,option))
    return max(options,key=lambda item:item[0])[1] if options else None

def detect_regions(image: Image.Image, mode: str = 'both') -> list[Region]:
    _, langs = configure_ocr()
    work = image.convert('RGB')
    # Keep OCR tractable on very large scans; all coordinates map back to original.
    scale = min(1.0, 3400 / max(work.size))
    regions: list[Region] = []
    # Closed balloons: OCR on a crop avoids the balloon outline confusing Tesseract.
    for outline,contour in _balloon_candidates(work):
        recognized=_ocr_balloon(work,outline,contour,mode,langs)
        if recognized:
            regions.append(recognized)
    # Remaining captions, signs and unclosed balloons: whole-page sparse OCR.
    if mode in ('both', 'horizontal') and 'jpn' in langs:
        regions += _ocr_pass(work, 'jpn', 11, scale)
    if mode in ('both', 'vertical') and 'jpn_vert' in langs:
        regions += _ocr_pass(work, 'jpn_vert', 3, scale)
    # Remove competing recognitions of the same pixels, favoring quality.
    regions.sort(key=lambda r: (r.confidence + min(len(r.source),20)*0.7), reverse=True)
    selected: list[Region] = []
    for region in regions:
        if any(_rect_iou(region.ink, old.ink) > .32 or
               (region.ink[0] >= old.ink[0] and region.ink[1] >= old.ink[1] and
                region.ink[0]+region.ink[2] <= old.ink[0]+old.ink[2] and
                region.ink[1]+region.ink[3] <= old.ink[1]+old.ink[3])
               for old in selected):
            continue
        selected.append(region)
    for r in selected:
        if r.box == r.ink:
            r.box = estimate_layout_box(work, r.ink, r.orientation == 'vertical')
    # Simple visual reading order. Bubbles on the same band: rightmost first (Japanese original).
    selected.sort(key=lambda r: (round(r.ink[1]/max(1,work.height/12)), -r.ink[0], r.ink[1]))
    return selected


def clean_image(image: Image.Image, regions: Iterable[Region], strength: int = 50) -> Image.Image:
    """Remove high-contrast strokes inside tight OCR rectangles, inpaint from surroundings."""
    has_alpha = image.mode == 'RGBA'
    base = np.asarray(image.convert('RGB')).copy()
    H,W = base.shape[:2]
    # Strength 0..100: high strength = lower discrimination threshold.
    threshold = round(65 - .49 * max(0,min(100,strength)))
    for region in regions:
        if region.action == 'keep':
            continue
        ix,iy,iw,ih = clip_xywh(region.ink, W,H)
        if iw < 2 or ih < 2:
            continue
        margin = max(8, min(18, round(min(iw,ih)*.16)))
        x0,y0 = max(0,ix-margin), max(0,iy-margin)
        x1,y1 = min(W,ix+iw+margin), min(H,iy+ih+margin)
        patch = base[y0:y1,x0:x1]
        if patch.size == 0:
            continue
        # Estimate color of the surrounding balloon from the padding ring.
        border = np.ones(patch.shape[:2], dtype=bool)
        border[iy-y0:iy-y0+ih, ix-x0:ix-x0+iw] = False
        colors = patch[border]
        if len(colors) < 5:
            continue
        bg = np.median(colors, axis=0)
        delta = np.max(np.abs(patch.astype(np.int16)-bg.astype(np.int16)),axis=2)
        mask = ((delta > threshold).astype(np.uint8)*255)
        # Restrict erasing to ink rectangle, +2px for antialiased edges.
        safe = np.zeros(patch.shape[:2],dtype=np.uint8)
        ext = 2
        yy0,yy1 = max(0,iy-y0-ext), min(y1-y0,iy-y0+ih+ext)
        xx0,xx1 = max(0,ix-x0-ext), min(x1-x0,ix-x0+iw+ext)
        safe[yy0:yy1,xx0:xx1]=255
        mask = cv2.bitwise_and(mask,safe)
        mask = cv2.dilate(mask,np.ones((3,3),np.uint8),iterations=1)
        mask = cv2.bitwise_and(mask,safe)
        base[y0:y1,x0:x1] = cv2.inpaint(patch,mask,3,cv2.INPAINT_TELEA)
    out = Image.fromarray(base,'RGB')
    if has_alpha:
        out.putalpha(image.getchannel('A'))
    return out


def default_font_path() -> str:
    candidates = [
        Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'arial.ttf',
        Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/'segoeui.ttf',
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf'),
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return 'DejaVuSans.ttf'


def _font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.truetype(default_font_path(), size)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=font)


def _wrap_line(line: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    if not line:
        return ['']
    result=[]
    current=''
    for word in line.split(' '):
        trial = (current+' '+word).strip() if current else word
        if _text_width(draw,trial,font) <= width:
            current=trial
        else:
            if current:
                result.append(current)
                current=''
            if _text_width(draw,word,font)<=width:
                current=word
            else:
                # Split unusually long words so no line is clipped.
                for char in word:
                    trial=current+char
                    if current and _text_width(draw,trial,font)>width:
                        result.append(current)
                        current=char
                    else:
                        current=trial
    if current:
        result.append(current)
    return result or ['']


def layout_text(text: str, width: int, height: int, font_path: str, size: int) -> list[str] | None:
    if width <= 4 or height <= 4:
        return None
    font = _font(font_path,size)
    draw=ImageDraw.Draw(Image.new('RGB',(1,1)))
    area_width=max(1,width-8)
    lines=[]
    for line in text.replace('\r','').split('\n'):
        lines.extend(_wrap_line(line,draw,font,area_width))
    row_height=max(1,round(size*1.24))
    if len(lines)*row_height > height-4:
        return None
    if any(_text_width(draw,l,font) > area_width+.2 for l in lines):
        return None
    return lines


def fitting_size(text: str, box: tuple[int,int,int,int], font_path: str, cap: int = 72) -> int:
    _,_,w,h=box
    low,high=1,max(1,min(cap,round(h*.9)))
    best=1
    while low<=high:
        mid=(low+high)//2
        if layout_text(text,w,h,font_path,mid) is not None:
            best=mid
            low=mid+1
        else:
            high=mid-1
    return best


def add_translations(cleaned: Image.Image, regions: Iterable[Region], font_path: str = '',
                     size_percent: int = 100) -> Image.Image:
    """Render left-to-right, centered text in each assigned layout box."""
    image=cleaned.convert('RGBA')
    W,H=image.size
    active=[r for r in regions if r.action == 'translate' and r.translated.strip()]
    if not active:
        return image if cleaned.mode=='RGBA' else image.convert('RGB')
    font_path=font_path or default_font_path()
    maximums=[]
    for r in active:
        # Free-standing lettering should remain close to the original glyph size.
        # Its fitted region may be wider than the original ink (especially Japanese vertical writing).
        cap=72
        if not r.balloon:
            _,_,iw,ih=r.ink
            cap=max(7, min(110, round(iw*1.25 if r.orientation == 'vertical' else ih*.95)))
        maximums.append(fitting_size(r.translated,clip_xywh(r.box,W,H),font_path,cap))
    # Shared typical font size. Rare long texts may use their individual smaller size.
    preferred=int(round(float(np.percentile(maximums,35))*size_percent/100))
    preferred=max(1,preferred)
    for r,maxsize in zip(active,maximums):
        x,y,w,h=clip_xywh(r.box,W,H)
        if w<3 or h<3:
            continue
        # Balloon lettering is harmonised per page. Captions have their own size.
        size=max(1,min(maxsize,preferred if r.balloon else round(maxsize*size_percent/100)))
        lines=layout_text(r.translated,w,h,font_path,size)
        if lines is None:
            continue
        font=_font(font_path,size)
        # Sample behind the text. Use light-on-dark / dark-on-light with fine outline.
        sample=np.asarray(image.crop((x,y,x+w,y+h)).convert('RGB'))
        light=float(np.median(sample[:,:,0]*.299+sample[:,:,1]*.587+sample[:,:,2]*.114))
        foreground=(0,0,0,255) if light>130 else (255,255,255,255)
        outline=(255,255,255,210) if light>130 else (0,0,0,210)
        layer=Image.new('RGBA',(w,h),(0,0,0,0))
        draw=ImageDraw.Draw(layer)
        row_height=max(1,round(size*1.24))
        top=(h-row_height*len(lines))/2
        for i,line in enumerate(lines):
            draw.text((w/2,top+i*row_height+row_height/2),line,font=font,
                      anchor='mm',align='center',fill=foreground,
                      stroke_width=1 if size>=11 else 0, stroke_fill=outline)
        image.alpha_composite(layer,(x,y))
    return image if cleaned.mode=='RGBA' else image.convert('RGB')


def numbered_source(pages: list[Page]) -> str:
    lines=[]
    for page in pages:
        for r in page.regions:
            if r.action == 'translate':
                lines.append(f'[{r.id:04d}] {r.source.replace(chr(10)," ").strip()}')
    return '\n'.join(lines)


def numbered_translations(pages: list[Page]) -> str:
    lines=[]
    for page in pages:
        for r in page.regions:
            if r.action == 'translate':
                lines.append(f'[{r.id:04d}] {r.translated.strip()}')
    return '\n'.join(lines)


def parse_numbered(text: str) -> dict[int,str]:
    """[0001] text; next lines belong to the same block until next marker."""
    pattern=re.compile(r'^\s*\[(\d+)\]\s*',re.MULTILINE)
    matches=list(pattern.finditer(text))
    if not matches:
        raise ValueError('Не найдены номера строк вида [0001].')
    if text[:matches[0].start()].strip():
        raise ValueError('Перед первым номером обнаружен текст без номера.')
    result={}
    for i,m in enumerate(matches):
        n=int(m.group(1))
        if n in result:
            raise ValueError(f'Номер [{n:04d}] повторяется.')
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        result[n]=text[m.end():end].strip()
    return result


def to_json_pages(pages: list[Page]) -> list[dict]:
    return [{'path':p.path,'regions':[asdict(r) for r in p.regions]} for p in pages]


def from_json_pages(data: list[dict]) -> list[Page]:
    return [Page(path=item['path'],regions=[Region(**r) for r in item.get('regions',[])]) for item in data]


def save_image(image: Image.Image, filepath: str) -> None:
    ext=Path(filepath).suffix.lower()
    if ext in ('.jpg','.jpeg'):
        if image.mode!='RGB':
            rgba=image.convert('RGBA')
            bg=Image.new('RGBA',rgba.size,'white')
            image=Image.alpha_composite(bg,rgba).convert('RGB')
        image.save(filepath,'JPEG',quality=100,subsampling=0,optimize=True)
    elif ext=='.png':
        image.save(filepath,'PNG',compress_level=6)
    else:
        raise ValueError('Поддерживается сохранение в PNG или JPG.')