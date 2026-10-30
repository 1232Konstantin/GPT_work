from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import engine
import pytest


def test_numbered_roundtrip():
    p1=engine.Page('one.png',[engine.Region(1,(1,2,30,20),(0,0,60,40),'おはよう'),
                                  engine.Region(2,(1,2,30,20),(0,0,60,40),'はい')])
    p2=engine.Page('two.jpg',[engine.Region(3,(1,2,30,20),(0,0,60,40),'ありがとう')])
    assert engine.numbered_source([p1,p2]).count('\n')==2
    assert engine.parse_numbered('[0001] Привет\n[0002] Да\n[0003] Первая\nвторая') == {
        1:'Привет',2:'Да',3:'Первая\nвторая'}
    with pytest.raises(ValueError):
        engine.parse_numbered('[0001] Да\n[0001] Нет')
    with pytest.raises(ValueError):
        engine.parse_numbered('Привет, [0001] да')


def test_clean_only_changes_text_bounds():
    image=Image.new('RGB',(360,160),(245,245,240))
    draw=ImageDraw.Draw(image)
    font=ImageFont.truetype(engine.default_font_path(),26)
    draw.text((95,45),'TEST',font=font,fill=(10,10,10))
    r=engine.Region(1,(92,42,90,38),(85,36,150,80))
    cleaned=engine.clean_image(image,[r],strength=60)
    before=np.asarray(image).astype(np.int16)
    after=np.asarray(cleaned).astype(np.int16)
    assert np.sum(np.abs(before-after))>0
    assert np.array_equal(before[:38],after[:38])
    assert np.array_equal(before[:, :80],after[:, :80])


def test_typesetting_outside_box_unchanged():
    im=Image.new('RGB',(400,220),'white')
    r=engine.Region(1,(80,20,70,50),(60,30,160,110),translated='Очень длинное предложение, которое должно переноситься.')
    result=engine.add_translations(im,[r],font_path=engine.default_font_path())
    before=np.asarray(im)
    after=np.asarray(result)
    assert np.any(np.any(before!=after,axis=2))
    assert np.array_equal(before[:29],after[:29])
    assert np.array_equal(before[:, :59],after[:, :59])
    assert np.array_equal(before[141:],after[141:])
    assert np.array_equal(before[:, 221:],after[:, 221:])


def test_export_lossless(tmp_path):
    image=Image.fromarray(np.random.default_rng(100).integers(0,255,(50,60,3),dtype=np.uint8),'RGB')
    png=tmp_path/'out.png'
    engine.save_image(image,str(png))
    assert np.array_equal(np.asarray(Image.open(png)),np.asarray(image))
    jpeg=tmp_path/'out.jpg'
    engine.save_image(image,str(jpeg))
    assert jpeg.is_file()


def test_clip_and_layout():
    assert engine.clip_xywh((-20,-10,50,40),100,100)==(0,0,30,30)
    assert engine.fitting_size('Long english translated text',(0,0,80,60),engine.default_font_path())>=1
    assert engine.layout_text('Большой перевод',120,70,engine.default_font_path(),12) is not None


def test_balloon_layout_is_inside_outline():
    im=Image.new('RGB',(700,430),'#cccccc')
    d=ImageDraw.Draw(im)
    d.ellipse((50,40,640,380),fill='white',outline='black',width=6)
    candidates=engine._balloon_candidates(im)
    assert candidates
    outline,contour=candidates[0]
    rect=engine._inscribed_box(outline,contour)
    assert rect is not None
    x,y,w,h=rect
    assert 50 < x < x+w < 640
    assert 40 < y < y+h < 380
    assert w>200 and h>100