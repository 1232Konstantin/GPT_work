"""v0.11 regressions: two search models, opaque masks, bounded restoration and folder DnD."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QPointF, QMimeData, QUrl
from PySide6.QtGui import QDropEvent, QDragEnterEvent
from PySide6.QtWidgets import QApplication

from cleanup_v11 import segment_text, restore_text_background, _reference_colour
from contrast_cleanup import mask_components, confirmed_mask
from qt_editor import EditorWindow
from qt_viewer import _dropped_folder, PictureView


def _rings():
    src=Image.new('RGB',(260,155),'white')
    d=ImageDraw.Draw(src)
    for x in (64,145):
        d.rectangle((x-4,35,x+30,107),fill='black')
        d.rectangle((x+1,40,x+25,102),fill='white')
    return src


@pytest.mark.parametrize('reference',['border','area'])
def test_outline_fills_holes_independently_of_reference(reference):
    picture=_rings()
    rect=(20,12,203,128)
    ordinary=segment_text(picture,rect,reference=reference,outlined=False,grow=0)
    outlined=segment_text(picture,rect,reference=reference,outlined=True,grow=0)
    assert outlined.mask[60,75]==255
    assert outlined.mask[60,157]==255
    assert ordinary.mask[60,75]==0, 'Background-coloured glyph hole is not ordinary ink'
    assert ordinary.mask[60,157]==0
    assert outlined.mask[0,0]==0
    assert np.count_nonzero(outlined.mask)>np.count_nonzero(ordinary.mask)
    # Building the approved mask must preserve the exact opaque pixel geometry.
    roi,boxes,ids,labels=mask_components(outlined.mask,rect)
    assert len(boxes)==2
    approved=confirmed_mask(outlined.mask,roi,labels,ids,boxes,[True,False])
    assert 0<np.count_nonzero(approved)<np.count_nonzero(outlined.mask)
    assert approved[60,75] != approved[60,157]


@pytest.mark.parametrize('reference',['border','area'])
def test_plain_dark_and_coloured_letters_detected(reference):
    src=Image.new('RGB',(250,145),(205,190,174))
    d=ImageDraw.Draw(src)
    d.rectangle((54,45,70,88),fill=(15,15,15))
    d.rectangle((123,45,143,88),fill=(210,45,45))
    segment=segment_text(src,(28,22,190,99),reference=reference,
                         outlined=False,grow=0)
    assert segment.mask[60,62]==255
    assert segment.mask[60,132]==255
    assert segment.mask[0,0]==0


def _gradient(kind, size=(240,155)):
    w,h=size
    yy,xx=np.mgrid[:h,:w]
    if kind=='horizontal':
        base=np.stack((70+xx*.60, 105+xx*.31, 180-xx*.25),axis=2)
    elif kind=='vertical':
        base=np.stack((60+yy*.88, 185-yy*.46, 110+yy*.52),axis=2)
    elif kind=='flat':
        base=np.full((h,w,3),[182,174,161],dtype=float)
    else:
        base=np.stack((143+24*np.sin(xx/21)+7*np.cos(yy/19),
                       120+28*np.sin(xx/26+yy/40),
                       159+24*np.cos(yy/23)+5*np.sin(xx/13)),axis=2)
    return Image.fromarray(np.uint8(np.clip(base,0,255)),'RGB')


@pytest.mark.parametrize('kind,method,max_error',[
    ('horizontal','horizontal',2.0),
    ('vertical','vertical',2.0),
    ('flat','solid',1.0),
    ('horizontal','plane',7.0),
    ('vertical','quadratic',9.0),
])
@pytest.mark.parametrize('average',[False,True])
def test_repair_reconstructs_known_background_without_touching_other_pixels(
        kind,method,max_error,average):
    original=_gradient(kind)
    source=np.asarray(original).copy()
    mask=np.zeros(source.shape[:2],dtype=np.uint8)
    mask[52:99,89:136]=255
    source[mask>0]=[0,0,0]
    image=Image.fromarray(source,'RGB')
    result=restore_text_background(image,mask,(35,20,165,115),
                                   method=method,average=average)
    output=np.asarray(result)
    expected=np.asarray(original)
    assert np.array_equal(output[mask==0],source[mask==0])
    mae=np.mean(abs(output[mask>0].astype(float)-expected[mask>0]))
    assert mae<=max_error, (kind,method,average,mae)


@pytest.mark.parametrize('method',['horizontal','vertical','plane','quadratic','telea','solid'])
def test_complex_background_filling_does_not_create_white_blob(method):
    original=_gradient('complex')
    arr=np.asarray(original).copy()
    mask=np.zeros(arr.shape[:2],dtype=np.uint8)
    mask[45:100,93:151]=255
    arr[mask>0]=[0,0,0]
    result=restore_text_background(Image.fromarray(arr,'RGB'),mask,
                                   (28,20,182,111),method,average=True)
    rgb=np.asarray(result)
    assert np.array_equal(rgb[mask==0],arr[mask==0])
    inside=rgb[mask>0]
    assert np.all(inside.mean(axis=0)<225), (method, inside.mean(axis=0))
    assert np.all(inside.mean(axis=0)>40)


def test_average_uses_only_unmasked_grid_nodes():
    image=_gradient('flat')
    array=np.asarray(image)
    mask=np.zeros(array.shape[:2],dtype=bool)
    mask[18:130,25:206]=True
    colour,samples=_reference_colour(array,mask,(0,0,240,155),grid=True)
    assert samples.shape[0] < 18*18
    assert np.allclose(colour,[182,174,161],atol=1)


def test_rgba_alpha_and_outside_mask_are_unchanged():
    src=np.asarray(_gradient('horizontal')).copy()
    alpha=np.full(src.shape[:2]+(1,),175,dtype=np.uint8)
    rgba=np.concatenate((src,alpha),axis=2)
    image=Image.fromarray(rgba,'RGBA')
    mask=np.zeros(src.shape[:2],np.uint8)
    mask[70:85,110:135]=255
    patched=image.copy()
    draw=ImageDraw.Draw(patched)
    draw.rectangle((110,70,134,84),fill=(4,4,4,175))
    out=restore_text_background(patched,mask,(30,30,150,95),'horizontal')
    assert out.mode=='RGBA'
    assert np.array_equal(np.asarray(out)[mask==0],np.asarray(patched)[mask==0])
    assert np.array_equal(np.asarray(out)[:,:,3],np.asarray(patched)[:,:,3])


def test_opaque_component_overlay_and_pixel_clicks(tmp_path):
    app=QApplication.instance() or QApplication([])
    file=tmp_path/'rings.png'
    _rings().save(file)
    editor=EditorWindow(file)
    try:
        editor.cleanup_mode.setCurrentIndex(editor.cleanup_mode.findData('border'))
        editor.cleanup_outline.setChecked(True)
        editor.cleanup_grow.setValue(0)
        editor._selected((20,12,203,128))
        assert len(editor._cleanup_candidates)==2
        assert editor.canvas.cleanup_mask_item is not None
        assert not editor.canvas.cleanup_items, 'No bounding rectangles'
        assert editor.canvas.cleanup_component_at(QPointF(75,60)) is not None
        first=editor.canvas.cleanup_component_at(QPointF(75,60))
        other=editor.canvas.cleanup_component_at(QPointF(157,60))
        assert first!=other
        assert editor.canvas.cleanup_component_at(QPointF(115,75)) is None
        pix=editor.canvas.cleanup_mask_item.pixmap().toImage()
        local_x,local_y=75-20,60-12
        active=pix.pixelColor(local_x,local_y)
        assert (active.red(),active.green(),active.blue(),active.alpha()) == (
            57,255,20,255)
        editor._toggle_cleanup_box(first)
        assert not editor._cleanup_enabled[first]
        pix=editor.canvas.cleanup_mask_item.pixmap().toImage()
        rejected=pix.pixelColor(local_x,local_y)
        assert rejected.name()=='#ff00ff' and rejected.alpha()==255
        assert not editor._approved_cleanup_mask()[60,75]
        assert editor._approved_cleanup_mask()[60,157]
        editor.cleanup_mask_color.setCurrentIndex(
            editor.cleanup_mask_color.findData('#ffffff'))
        assert editor.canvas.cleanup_mask_item.pixmap().toImage().pixelColor(
            157-20,60-12).name()=='#ffffff'
        assert editor.working.tobytes()==editor.original.tobytes()
    finally:
        editor.dirty=False
        editor.close()


def test_drop_accepts_local_folder_not_file_or_remote(tmp_path):
    folder=tmp_path/'pages'
    folder.mkdir()
    file=tmp_path/'image.png'
    Image.new('RGB',(15,15)).save(file)
    mime=QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(folder))])
    assert _dropped_folder(mime)==folder
    view=PictureView()
    captured=[]
    view.folder_dropped.connect(captured.append)
    event=QDragEnterEvent(__import__('PySide6.QtCore',fromlist=['QPoint']).QPoint(10,10),
                           Qt.DropAction.CopyAction,mime,
                           Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
    view.dragEnterEvent(event)
    assert event.isAccepted()
    drop=QDropEvent(QPointF(10,10),Qt.DropAction.CopyAction,mime,
                    Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier)
    view.dropEvent(drop)
    assert drop.isAccepted() and captured==[folder]
    mime.setUrls([QUrl.fromLocalFile(str(file))])
    assert _dropped_folder(mime) is None
    mime.setUrls([QUrl('https://example.org/pages')])
    assert _dropped_folder(mime) is None
    mime.setUrls([QUrl.fromLocalFile(str(folder)),QUrl.fromLocalFile(str(file))])
    assert _dropped_folder(mime) is None
    view.close()
