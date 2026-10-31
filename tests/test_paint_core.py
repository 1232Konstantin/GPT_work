import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from paint_core import clear_letters, ink_mask, clip_box, brush_segment, palette_smudge


def test_only_glyphs_cleared_and_outside_exactly_preserved():
    picture = Image.new('RGB',(260,190),'#f7f3db')
    d=ImageDraw.Draw(picture)
    d.ellipse((17,14,242,175), outline='black', width=5)
    for x in [92,112,132,152]:
        d.rectangle((x,70,x+9,103),fill='#080808')
    after, pixels, mask=clear_letters(picture,(63,46,130,84))
    assert pixels>0
    assert pixels<130*84*.3
    a=np.asarray(picture); b=np.asarray(after);m=np.asarray(mask)
    assert np.array_equal(a[14,128], b[14,128]),'Balloon outline must be preserved'
    outside=np.ones((190,260),dtype=bool);outside[46:130,63:193]=False
    assert np.array_equal(a[outside],b[outside])
    assert np.sum(a!=b)>0


def test_blank_selection_remains_unchanged():
    im=Image.new('RGB',(120,100),'white')
    cleared,pixels,mask=clear_letters(im,(12,15,50,40))
    assert pixels==0 and im.tobytes()==cleared.tobytes()
    assert not np.asarray(mask).any()


def test_user_clip_and_brush_restore():
    im=Image.new('RGB',(30,30),'white')
    edited=brush_segment(im, im,(4,4),(19,19),5,(0,0,0,255))
    assert edited.getpixel((10,10))==(0,0,0)
    restored=brush_segment(edited,im,(4,4),(19,19),7,(0,0,0,255),restore=True)
    assert restored.getpixel((10,10))==(255,255,255)
    assert im.getpixel((10,10))==(255,255,255)
    assert clip_box((-12,-1,18,9),im.size)==(0,0,6,8)


def test_smudge_only_local_and_does_not_change_source():
    im=Image.new('RGB',(80,80),'white')
    ImageDraw.Draw(im).rectangle((18,23,29,34),fill='black')
    result=palette_smudge(im,(24,28),(39,28),16)
    assert result.tobytes()!=im.tobytes()
    assert im.getpixel((39,28))==(255,255,255)
    assert result.getpixel((70,70))==im.getpixel((70,70))