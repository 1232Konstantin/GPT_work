"""Regression checks for editor v0.9 commands, modes and flat background."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtWidgets import QApplication

from contrast_cleanup import segment_ink
from qt_editor import EditorWindow, TextLayer


def test_flat_background_small_fragment():
    image = Image.new('RGB', (60, 50), (220, 220, 220))
    ImageDraw.Draw(image).rectangle((21, 18, 26, 25), fill=(15, 15, 15))
    selection = (18, 15, 12, 13)
    seg = segment_ink(image, selection, mode='flat', grow=0)
    assert seg.mask[21, 23] == 255
    assert seg.mask[15, 18] == 0
    assert seg.mask[0, 0] == 0
    assert np.count_nonzero(seg.mask) > 20


def test_flat_background_handles_very_small_boxes():
    im = Image.new('RGB', (30, 30), 'white')
    ImageDraw.Draw(im).rectangle((12, 12, 14, 14), fill='black')
    seg = segment_ink(im, (10, 10, 7, 7), mode='flat', grow=0)
    assert seg.mask[13, 13] == 255
    assert seg.mask[9, 13] == 0


def test_image_and_text_commands_are_independent(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / 'page.png'
    Image.new('RGB',(100,80),'white').save(source)
    editor = EditorWindow(source)
    try:
        assert editor.sidebar_modes.count() == 3
        assert editor.cleanup_mode.findData('flat') >= 0
        editor._checkpoint('image')
        editor.working.putpixel((10, 10), (0,0,0))
        editor._checkpoint('text')
        editor.layers.append(TextLayer((20,20,50,25),'hello'))
        editor._checkpoint('image')
        editor.working.putpixel((11, 10), (0,0,0))
        editor.undo()
        assert editor.working.getpixel((11,10)) == (255,255,255)
        assert editor.working.getpixel((10,10)) == (0,0,0)
        assert editor.layers[0].text == 'hello'
        editor.undo()
        assert not editor.layers
        assert editor.working.getpixel((10,10)) == (0,0,0)
        editor.redo()
        assert editor.layers[0].text == 'hello'
        editor.redo()
        assert editor.working.getpixel((11,10)) == (0,0,0)
        assert editor.layers[0].text == 'hello'
    finally:
        editor.dirty = False
        editor.close()


def test_text_edit_does_not_checkpoint_raster(tmp_path):
    app = QApplication.instance() or QApplication([])
    source = tmp_path / 'page.png'
    Image.new('RGB',(100,80),'white').save(source)
    editor = EditorWindow(source)
    try:
        editor._create_text_box((10,10,50,30))
        editor._text_in_canvas('one')
        editor._text_in_canvas('two')
        assert editor.layers[-1].text == 'two'
        editor.undo()
        assert editor.layers[-1].text == 'one'
        editor.undo()
        assert editor.layers[-1].text == ''
        editor.redo()
        assert editor.layers[-1].text == 'one'
    finally:
        editor.dirty = False
        editor.close()
