import numpy as np
from PIL import Image, ImageDraw
from cleanup_core import detect_glyph_boxes,remove_glyphs

def test_uniform_color_no_white_or_black_blobs():
    c=(182,147,92)
    im=Image.new('RGB',(260,150),c)
    dr=ImageDraw.Draw(im)
    for x in (60,85,110,135):
        dr.rectangle((x,50,x+10,82),fill='black')
        dr.rectangle((x+2,49,x+8,49),fill='white')
    boxes=[(56,46,19,42),(81,46,19,42),(106,46,19,42),(131,46,19,42)]
    out,touched=remove_glyphs(im,boxes)
    assert touched==sum(w*h for x,y,w,h in boxes)
    rgb=np.asarray(out)
    for x,y,w,h in boxes:
        p=rgb[y:y+h,x:x+w]
        assert np.mean(np.abs(p.astype('int32')-np.array(c)))<4, p.reshape(-1,3).mean(0)
    mask=np.zeros(rgb.shape[:2],bool)
    for x,y,w,h in boxes:mask[y:y+h,x:x+w]=1
    assert np.array_equal(np.asarray(im)[~mask],rgb[~mask])
    assert im.getpixel((64,60))==(0,0,0)

def test_empty_and_image_immutability_and_idempotence_at_unchanged_source():
    im=Image.new('RGB',(120,120),(90,110,130))
    out,n=remove_glyphs(im,[])
    assert n==0 and out.tobytes()==im.tobytes()
    a,n=remove_glyphs(im,[(20,20,10,10)])
    b,m=remove_glyphs(im,[(20,20,10,10)])
    assert a.tobytes()==b.tobytes()

def test_candidate_boxes_inside_selection_and_not_column_sized():
    img=Image.new('RGB',(240,250),'#dcc2a4')
    d=ImageDraw.Draw(img)
    for y in (55,77,99,121,143):
        d.rectangle((85,y,99,y+17),fill='black')
    selection=(70,45,60,140)
    boxes=detect_glyph_boxes(img,selection)
    assert len(boxes)>=4,boxes
    assert all(70<=x and 45<=y and x+w<=130 and y+h<=185 for x,y,w,h in boxes)
    assert all(h<85 for _,_,_,h in boxes)

def test_real_page_outside_approved_boxes_exact():
    from pathlib import Path
    file=Path('/mnt/data/изображение(1).png')
    if not file.exists():return
    im=Image.open(file).convert('RGB')
    boxes=detect_glyph_boxes(im,(255,90,75,175))
    assert len(boxes)>=6
    out,n=remove_glyphs(im,boxes)
    assert n>0
    mask=np.zeros((im.height,im.width),dtype=bool)
    for x,y,w,h in boxes:mask[y:y+h,x:x+w]=True
    assert np.array_equal(np.asarray(im)[~mask],np.asarray(out)[~mask])