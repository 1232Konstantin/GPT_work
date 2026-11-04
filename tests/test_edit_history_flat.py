"""Independent history semantics and small flat-background glyph recovery."""
from copy import deepcopy
import numpy as np
from PIL import Image,ImageDraw
import pytest

from edit_history import raster_command,text_command,CommandHistory
from contrast_cleanup import segment_flat_ink,repair_flat_background


def test_history_undo_cleanup_does_not_rollback_unrelated_text():
    a=Image.new("RGB",(50,40),"white")
    b=a.copy()
    ImageDraw.Draw(b).rectangle((10,10,15,20),fill=(230,225,220))
    history=CommandHistory()
    text0=[dict(text="原文",x=4)]
    text1=[dict(text="перевод",x=4)]
    assert history.push(text_command(text0,text1,0,0,"text"))
    assert history.push(raster_command(a,b,"clean"))
    img,txt,index=history.undo(b,deepcopy(text1))
    assert img.tobytes()==a.tobytes()
    assert txt==text1, "Undo cleaning must not revert translated text"
    img,txt,index=history.undo(img,txt)
    assert txt==text0
    assert img.tobytes()==a.tobytes()
    img,txt,index=history.redo(img,txt)
    assert txt==text1
    img,txt,index=history.redo(img,txt)
    assert img.tobytes()==b.tobytes()
    assert txt==text1


def test_history_new_command_invalidates_redo_without_merging_kinds():
    image=Image.new("RGB",(24,24),(0,0,0))
    changed=image.copy()
    changed.putpixel((5,5),(255,0,0))
    h=CommandHistory()
    assert h.push(raster_command(image,changed,"brush"))
    im,_,_=h.undo(changed,[])
    assert im.tobytes()==image.tobytes()
    h.push(text_command([],["hello"],-1,0,"create"))
    assert not h.future
    assert h.steps[-1].kind=="text"
    assert h.steps[0].kind=="text"


def test_flat_small_glyph_without_reliable_central_background():
    bg=(215,207,185)
    image=Image.new("RGB",(56,44),bg)
    draw=ImageDraw.Draw(image)
    draw.rectangle((21,17,26,28),fill=(24,25,24))
    # 12x16 rectangle with generous border but most central area is ink.
    region=(18,14,13,18)
    seg=segment_flat_ink(image,region,mode='dark',grow=0)
    mask=seg.mask
    assert mask[22,23]>0
    assert np.count_nonzero(mask)>0
    assert not np.any(mask[:14]) and not np.any(mask[:,31:])
    out=repair_flat_background(image,mask,region)
    assert out.getpixel((23,22))==bg
    before=np.asarray(image)
    after=np.asarray(out)
    assert np.array_equal(before[mask==0],after[mask==0])


def test_flat_background_preserves_outside_selection_even_for_bad_mask():
    base=Image.new("RGB",(45,35),(120,160,200))
    r=(14,9,11,12)
    arr=np.full((35,45),255,dtype=np.uint8)
    actual=repair_flat_background(base,arr,r)
    pre=np.asarray(base)
    post=np.asarray(actual)
    outside=np.ones((35,45),bool)
    outside[9:21,14:25]=False
    assert np.array_equal(pre[outside],post[outside])


def test_flat_background_light_ink_on_dark_border():
    bg=(36,46,55)
    im=Image.new("RGB",(58,55),bg)
    ImageDraw.Draw(im).rectangle((22,23,27,35),fill=(248,240,238))
    region=(18,19,14,21)
    mask=segment_flat_ink(im,region,'light',grow=0).mask
    assert mask[30,25]>0
    out=repair_flat_background(im,mask,region)
    assert out.getpixel((25,30))==bg
