"""Regression tests for Paint-3D-like local brush tools (no GUI required)."""
from PIL import Image, ImageDraw
import numpy as np
import pytest
from paint3d_tools import draw_stroke, fill_region, TOOLS


@pytest.mark.parametrize("tool",[
    "marker","calligraphy","oil","watercolor","pixel",
    "pencil","brush","crayon","spray"
])
def test_brush_stroke_local_and_source_untouched(tool):
    src=Image.new("RGB",(130,96),(240,235,225))
    before=src.tobytes()
    out=draw_stroke(src,src,(22,45),(88,45),12,(20,80,190),tool,100)
    assert out.size==src.size
    assert src.tobytes()==before
    assert out.tobytes()!=before,tool
    assert out.getpixel((0,0))==src.getpixel((0,0))
    assert out.getpixel((129,95))==src.getpixel((129,95))


def test_opacity_is_respected_for_marker():
    src=Image.new('RGB',(80,80),'white')
    full=draw_stroke(src,src,(20,40),(60,40),18,(0,0,0),'marker',100)
    partial=draw_stroke(src,src,(20,40),(60,40),18,(0,0,0),'marker',20)
    assert 0<=full.getpixel((40,40))[0]<partial.getpixel((40,40))[0]<255


def test_restore_reveals_source_after_drawing():
    src=Image.new('RGB',(100,100),'#afcede')
    painted=draw_stroke(src,src,(20,45),(80,45),13,(210,10,30),'marker',100)
    restored=draw_stroke(painted,src,(20,45),(80,45),17,(0,0,0),'restore',100)
    assert restored.getpixel((50,45))==src.getpixel((50,45))
    assert painted.getpixel((50,45))!=src.getpixel((50,45))


def test_fill_respects_contiguous_border_and_opacity():
    src=Image.new('RGB',(100,70),'white')
    ImageDraw.Draw(src).rectangle((10,10,58,55),outline='black',width=3)
    out=fill_region(src,(30,30),(20,70,200),100)
    assert out.getpixel((30,30))==(20,70,200)
    assert out.getpixel((5,5))==(255,255,255)
    assert out.getpixel((10,30))==(0,0,0)
    out2=fill_region(src,(30,30),(20,70,200),50)
    assert out2.getpixel((30,30))[0]>out.getpixel((30,30))[0]
    assert src.getpixel((30,30))==(255,255,255)


def test_smudge_respects_opacity():
    original=Image.new('RGB',(100,85),'white')
    ImageDraw.Draw(original).rectangle((28,33,40,45),fill='black')
    full=draw_stroke(original,original,(33,39),(57,39),20,(0,0,0),'smudge',100)
    zero=draw_stroke(original,original,(33,39),(57,39),20,(0,0,0),'smudge',0)
    assert zero.tobytes()==original.tobytes()
    assert full.tobytes()!=original.tobytes()
