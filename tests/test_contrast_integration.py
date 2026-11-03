"""Editor-independent regression tests: real pixel mask, enabled components, repair."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from contrast_cleanup import segment_ink, repair_background, mask_components, confirmed_mask


def _background(mode):
    h,w=175,215
    yy,xx=np.mgrid[:h,:w]
    if mode=='outline':
        base=np.stack((30+xx*.03, 43+yy*.045, 58+xx*.03+yy*.02),axis=-1)
    else:
        base=np.stack((201+xx*.1, 161+xx*.06+yy*.02, 115+xx*.07),axis=-1)
    return Image.fromarray(np.uint8(np.clip(base,0,255)),'RGB')


@pytest.mark.parametrize('mode',['dark','outline'])
def test_contours_repaired_without_touching_other_pixels(mode):
    base=_background(mode)
    img=base.copy(); d=ImageDraw.Draw(img)
    for x in (83,117,151):
        if mode=='outline':
            d.rectangle((x-3,54-3,x+13,116),fill='white')
        d.rectangle((x,54,x+10,113),fill=(12,12,16))
    selection=(52,36,140,105)
    segment=segment_ink(img,selection,mode,grow=2)
    roi,boxes,ids,labels=mask_components(segment.mask,selection)
    assert boxes
    agreed=confirmed_mask(segment.mask,roi,labels,ids,boxes,[True]*len(boxes))
    assert np.count_nonzero(agreed)>200
    out=repair_background(img,agreed,'telea')
    a=np.asarray(img);b=np.asarray(out);clean=np.asarray(base)
    assert np.array_equal(a[agreed==0],b[agreed==0])
    assert np.mean(np.abs(b.astype(float)-clean.astype(float))) < 6


def test_excluding_components_and_manual_repair_strictly_bounded():
    image=Image.new('RGB',(155,125),'#e0b17a')
    dr=ImageDraw.Draw(image)
    dr.rectangle((46,40,54,70),fill='black')
    dr.rectangle((91,40,99,70),fill='black')
    sel=(25,20,112,90)
    segmented=segment_ink(image,sel,'dark',grow=1)
    rect,boxes,ids,labels=mask_components(segmented.mask,sel)
    assert len(boxes)>=2
    on=[True]*len(boxes); on[0]=False
    mask=confirmed_mask(segmented.mask,rect,labels,ids,boxes,on)
    x,y,w,h=boxes[0]
    assert not np.any(mask[y:y+h,x:x+w])
    out=repair_background(image,mask,'plane')
    before=np.asarray(image);after=np.asarray(out)
    assert np.array_equal(before[mask==0],after[mask==0])
    assert np.array_equal(before[y:y+h,x:x+w],after[y:y+h,x:x+w])
    manual=(119,33,9,12)
    updated=confirmed_mask(segmented.mask,rect,labels,ids+[None],boxes+[manual],on+[True])
    assert np.count_nonzero(updated)>np.count_nonzero(mask)
    assert updated[36,122]==255
    assert updated[0,0]==0


def test_no_text_and_no_enabled_regions_leave_image_untouched():
    image=Image.new('RGB',(135,100),'#a5b2c3')
    seg=segment_ink(image,(15,15,100,70),'outline')
    r,boxes,ids,labels=mask_components(seg.mask,(15,15,100,70))
    mask=confirmed_mask(seg.mask,r,labels,ids,boxes,[False]*len(boxes))
    assert not np.any(mask)
    assert repair_background(image,mask,'telea').tobytes()==image.tobytes()


def test_repeat_same_source_is_deterministic():
    image=_background('dark')
    dr=ImageDraw.Draw(image)
    dr.rectangle((88,49,107,108),fill='black')
    seg=segment_ink(image,(60,24,100,119),'dark',grow=1)
    r,boxes,ids,labels=mask_components(seg.mask,(60,24,100,119))
    mask=confirmed_mask(seg.mask,r,labels,ids,boxes,[True]*len(boxes))
    a=repair_background(image,mask,'telea')
    b=repair_background(image,mask,'telea')
    assert a.tobytes()==b.tobytes()


def test_real_japanese_page_if_available():
    src=Path(__file__).resolve().parents[1]/'изображение(1).png'
    if not src.exists():
        pytest.skip('Human test image not available in checkout')
    image=Image.open(src).convert('RGB')
    seg=segment_ink(image,(263,98,59,175),'outline',grow=2)
    r,boxes,ids,labels=mask_components(seg.mask,(263,98,59,175))
    mask=confirmed_mask(seg.mask,r,labels,ids,boxes,[True]*len(boxes))
    assert np.count_nonzero(mask)>1000
    out=repair_background(image,mask,'telea')
    assert np.array_equal(np.asarray(image)[mask==0],np.asarray(out)[mask==0])

def test_dark_outline_around_light_lettering():
    """Reverse polarity: a dark outline and a light glyph interior."""
    bg=Image.new('RGB',(235,145),(188,172,154))
    image=bg.copy()
    draw=ImageDraw.Draw(image)
    for x in (72,106,140):
        draw.rectangle((x-4,42,x+20,102),fill=(7,10,15))
        draw.rectangle((x,46,x+16,98),fill=(246,246,244))
    selection=(47,25,152,102)
    seg=segment_ink(image,selection,'outline_dark',grow=1)
    roi,boxes,ids,labels=mask_components(seg.mask,selection)
    assert len(boxes)>=3
    agreed=confirmed_mask(seg.mask,roi,labels,ids,boxes,[True]*len(boxes))
    assert agreed[65,82]>0, 'Light interior of outlined glyph must also be removed'
    result=repair_background(image,agreed,'telea')
    before=np.asarray(image)
    after=np.asarray(result)
    assert np.array_equal(before[agreed==0],after[agreed==0])
    assert result.getpixel((0,0))==bg.getpixel((0,0))
