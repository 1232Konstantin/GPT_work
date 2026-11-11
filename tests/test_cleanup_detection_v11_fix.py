"""Regressions for v0.11 preview.3: missing dark/light/colour glyph pixels.

The tests deliberately check whole glyph interiors, not just whether a few
edge pixels are found. Every detected mask remains clipped to its selection.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtWidgets import QApplication

from cleanup_v11 import segment_text
from contrast_cleanup import mask_components, confirmed_mask
from qt_editor import EditorWindow


@pytest.mark.parametrize('reference', ['area', 'border'])
def test_light_dark_and_isoluminant_pink_all_detected(reference):
    background=(180,180,180)
    picture=Image.new('RGB',(245,145),background)
    d=ImageDraw.Draw(picture)
    # Contrast to midgray is modest, but visible; old 25-L threshold loses it.
    d.rectangle((36,35,64,89),fill=(160,160,160))
    d.rectangle((100,35,127,89),fill=(199,199,199))
    # Nearly same luminance as gray; independently different Lab chroma.
    d.rectangle((163,35,204,89),fill=(206,165,181))
    # The chromatic glyph should not require a separate brightness edge.
    bg_lab=cv2.cvtColor(np.uint8([[background]]),cv2.COLOR_RGB2LAB)[0,0]
    pink_lab=cv2.cvtColor(np.uint8([[(206,165,181)]]),cv2.COLOR_RGB2LAB)[0,0]
    assert abs(int(pink_lab[0])-int(bg_lab[0])) < 10
    seg=segment_text(picture,(18,16,215,107),reference=reference,
                     outlined=False,grow=0)
    for point in ((48,65),(112,65),(184,65)):
        assert seg.mask[point[1],point[0]]==255, (reference,point)
    # Every flat-filled interior must be marked, not only the stroke edges.
    assert np.all(seg.mask[42:83,171:196]==255)
    assert seg.mask[10,10]==0


@pytest.mark.parametrize('reference',['area','border'])
def test_clipped_character_touching_selection_border_is_kept(reference):
    picture=Image.new('RGB',(230,150),'white')
    d=ImageDraw.Draw(picture)
    d.rectangle((30,45,85,96),fill=(20,20,20))
    selection=(40,22,120,100)  # cuts left edge of the foreground object
    seg=segment_text(picture,selection,reference=reference,
                     outlined=False,grow=0)
    assert seg.mask[60,40]==255
    assert seg.mask[60,80]==255
    roi,boxes,ids,labels=mask_components(seg.mask,selection)
    assert boxes, 'A clipped character must remain a clickable component'
    mask=confirmed_mask(seg.mask,roi,labels,ids,boxes,[True]*len(boxes))
    assert mask[60,40]==255


def test_component_spanning_most_of_tight_crop_is_not_automatically_rejected():
    picture=Image.new('RGB',(240,150),'white')
    d=ImageDraw.Draw(picture)
    d.rectangle((58,35,150,95),fill='black')
    # Foreground > 52% of selected region, but does not touch its perimeter.
    selection=(50,25,110,82)
    seg=segment_text(picture,selection,reference='border',grow=0)
    assert seg.mask[60,100]==255
    assert seg.mask[60,152]==0


def test_outline_only_adds_enclosed_interior_not_open_u():
    picture=Image.new('RGB',(240,155),'white')
    d=ImageDraw.Draw(picture)
    d.rectangle((40,32,87,104),outline='black',width=3)
    d.line(((133,35),(133,105),(185,105),(185,35)),fill='black',width=3)
    region=(18,15,202,127)
    base=segment_text(picture,region,reference='border',
                      outlined=False,grow=0)
    outlined=segment_text(picture,region,reference='border',
                          outlined=True,grow=0)
    assert base.mask[65,61]==0
    assert outlined.mask[65,61]==255
    assert outlined.mask[60,160]==0, 'Open U is not an enclosed glyph'
    assert np.count_nonzero(outlined.mask)>np.count_nonzero(base.mask)


def test_small_break_in_outline_is_sealed_but_large_break_is_not():
    picture=Image.new('RGB',(240,155),'white')
    d=ImageDraw.Draw(picture)
    d.rectangle((35,35,85,95),outline='black',width=3)
    # Tiny 1-pixel gap from anti-aliasing of an outlined letter.
    d.point((60,35),fill='white')
    d.rectangle((135,35,185,95),outline='black',width=3)
    d.rectangle((155,35,168,39),fill='white')  # intentional large gap
    outlined=segment_text(picture,(15,12,211,127),reference='border',
                          outlined=True,grow=0)
    assert outlined.mask[65,60]==255
    assert outlined.mask[65,160]==0


def test_noise_only_is_not_misclassified_as_symbols():
    rng=np.random.default_rng(52)
    noise=rng.integers(-2,3,size=(95,135,1))
    crop=np.uint8(np.clip(183+noise,0,255))
    picture=Image.fromarray(np.repeat(crop,3,axis=2),'RGB')
    for reference in ('border','area'):
        seg=segment_text(picture,(5,5,125,85),reference=reference,
                         outlined=False,grow=0)
        assert np.count_nonzero(seg.mask)==0


def test_mode_names_and_persistent_keys(tmp_path):
    app=QApplication.instance() or QApplication([])
    file=tmp_path/'page.png'
    Image.new('RGB',(140,100),'white').save(file)
    editor=EditorWindow(file)
    try:
        assert editor.cleanup_mode.itemText(
            editor.cleanup_mode.findData('border'))=='По границам'
        assert editor.cleanup_mode.itemText(
            editor.cleanup_mode.findData('area'))=='По фону'
        assert editor.cleanup_outline is not None
    finally:
        editor.dirty=False
        editor.close()


@pytest.mark.parametrize('reference',['border','area'])
def test_antialiased_coloured_text_on_subtle_gradient(reference):
    """Actual rasterized font, not just solid rectangular test glyphs."""
    height,width=160,300
    yy,xx=np.mgrid[:height,:width]
    shade=np.uint8(np.clip(180+7*np.sin(xx/37)+5*np.cos(yy/32),0,255))
    rgb=np.repeat(shade[:,:,None],3,axis=2)
    picture=Image.fromarray(rgb,'RGB')
    font=ImageFont.load_default(size=31)
    pos=(50,55)
    ink=Image.new('L',(width,height),0)
    ImageDraw.Draw(ink).text(pos,'PINK LETTERS',font=font,fill=255)
    foreground=Image.new('RGB',(width,height),(206,165,181))
    picture.paste(foreground,(0,0),ink)
    seg=segment_text(picture,(25,25,250,112),reference=reference,
                     outlined=False,grow=1)
    high_confidence=np.asarray(ink)>220
    assert high_confidence.sum()>400
    recall=np.mean((seg.mask>0)[high_confidence])
    assert recall>0.93,(reference,recall)
    assert seg.mask[8,8]==0
