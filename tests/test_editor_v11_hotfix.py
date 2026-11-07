"""Regression tests for 0.11.0-preview.2: pixel pencil and live folder lists."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import time
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtTest import QTest

from paint3d_tools import draw_stroke
from qt_editor import EditorWindow
from qt_viewer import ViewerWindow


def _app():
    return QApplication.instance() or QApplication([])


def _wait_until(predicate, timeout=4500):
    deadline=time.monotonic()+timeout/1000
    while time.monotonic()<deadline:
        _app().processEvents()
        if predicate():
            return
        QTest.qWait(20)
    assert predicate(), 'Timed out waiting for folder scan'


def test_one_pixel_pencil_modifies_exactly_one_raster_row():
    src=Image.new('RGB',(140,100),'white')
    expected=(9,103,217)
    out=draw_stroke(src,src,(21.8,36.3),(99.9,36.3),1,
                    expected,'pencil',100)
    ar=np.asarray(out)
    different=np.any(ar!=255,axis=2)
    assert np.flatnonzero(different.any(axis=1)).tolist()==[36]
    assert np.flatnonzero(different[36]).tolist()==list(range(21,100))
    assert np.all(ar[different]==np.array(expected,dtype=np.uint8))
    assert src.getpixel((22,36))==(255,255,255)


def test_one_pixel_pencil_diagonal_has_no_antialiased_neighbours():
    src=Image.new('RGB',(110,95),'white')
    out=draw_stroke(src,src,(19.2,17.7),(66.8,64.2),1,
                    (0,0,0),'pencil',100)
    ar=np.asarray(out)
    assert np.all((ar==0)|(ar==255))
    assert set(map(tuple,np.unique(ar.reshape(-1,3),axis=0)))=={
        (0,0,0),(255,255,255)}
    for y in range(ar.shape[0]):
        line=np.flatnonzero(np.any(ar[y]!=255,axis=1))
        assert len(line)<=1
    assert out.getpixel((19,18))==(255,255,255)


def test_brush_remains_old_soft_tool_and_is_separate_from_pixel_pencil():
    src=Image.new('RGB',(120,85),'white')
    pencil=draw_stroke(src,src,(21,36),(88,36),5,(12,35,90),'pencil',100)
    brush=draw_stroke(src,src,(21,36),(88,36),5,(12,35,90),'brush',100)
    assert pencil.tobytes()!=src.tobytes()
    assert brush.tobytes()!=src.tobytes()
    assert pencil.getpixel((20,20))==src.getpixel((20,20))
    assert brush.getpixel((20,20))==src.getpixel((20,20))


def test_pencil_uses_fast_non_resampled_canvas_and_own_brush_size(tmp_path):
    _app()
    file=tmp_path/'page.png'
    Image.new('RGB',(220,150),'white').save(file)
    editor=EditorWindow(file)
    try:
        assert 'pencil' in editor.brush_tool_buttons
        assert 'brush' in editor.brush_tool_buttons
        editor.set_tool('pencil')
        editor.brush_diameter.setValue(1)
        assert editor.canvas.pixmap_item.transformationMode()==(
            Qt.TransformationMode.FastTransformation)
        editor._painted((QPointF(12,16),QPointF(65,16),'pencil',1,True))
        editor._painted((None,None,'pencil',1,False))
        arr=np.asarray(editor.working)
        assert np.flatnonzero(np.any(arr!=255,axis=2).any(axis=1)).tolist()==[16]
        assert len(editor.history)==1
        assert editor.canvas._rendered_size==editor.working.size
        editor.set_tool('brush')
        assert editor.canvas.pixmap_item.transformationMode()==(
            Qt.TransformationMode.SmoothTransformation)
        editor.brush_diameter.setValue(14)
        editor.set_tool('pencil')
        assert editor.brush_diameter.value()==1
        editor.set_tool('brush')
        assert editor.brush_diameter.value()==14
        editor.undo()
        assert editor.working.tobytes()==editor.original.tobytes()
        editor.redo()
        assert editor.working.getpixel((35,16))!=editor.original.getpixel((35,16))
    finally:
        editor.dirty=False
        editor.close()


def test_saved_png_appears_in_live_viewer_without_losing_selection(tmp_path,monkeypatch):
    _app()
    original=tmp_path/'page1.png'
    exported=tmp_path/'page1_edited.png'
    Image.new('RGB',(140,85),'white').save(original)
    viewer=ViewerWindow()
    try:
        viewer.set_folder(tmp_path)
        _wait_until(lambda: viewer.current==original and len(viewer.paths)==1)
        assert str(tmp_path) in viewer._folder_watcher.directories()
        viewer.open_editor()
        editor=viewer.editor_by_path[original]
        monkeypatch.setattr(QFileDialog,'getSaveFileName',
                            lambda *args: (str(exported),'PNG (*.png)'))
        editor.save_png()
        assert exported.is_file()
        _wait_until(lambda: exported in viewer.paths)
        assert viewer.current==original
        assert viewer.counter_label.text()=='1 / 2'
        assert viewer.visible==[original,exported]
        viewer._return_to_viewer()
        assert viewer.current==original
        assert exported in viewer.visible
    finally:
        for ed in viewer.editor_by_path.values():
            ed.dirty=False
        viewer.close()


def test_directory_change_refreshes_added_file_without_reopening_page(tmp_path):
    _app()
    first=tmp_path/'page1.png'
    Image.new('RGB',(80,50),'white').save(first)
    viewer=ViewerWindow()
    try:
        viewer.set_folder(tmp_path)
        _wait_until(lambda:viewer.current==first)
        current_generation=viewer.image_generation
        second=tmp_path/'page2.png'
        Image.new('RGB',(80,50),'red').save(second)
        # Native watcher normally delivers this; emit explicitly so its
        # asynchronous callback is also covered in headless Linux CI.
        viewer._folder_watcher.directoryChanged.emit(str(tmp_path))
        _wait_until(lambda: second in viewer.visible)
        assert viewer.current==first
        assert viewer.image_generation==current_generation
        assert viewer.counter_label.text()=='1 / 2'
        assert viewer.folder==tmp_path
    finally:
        viewer.close()
