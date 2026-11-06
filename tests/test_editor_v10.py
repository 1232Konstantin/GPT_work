"""v0.10 user-acceptance regressions: committed operations only."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from qt_editor import EditorWindow, composite_text, qimage_from_pil
from paint3d_tools import draw_stroke


def _editor(tmp_path):
    app = QApplication.instance() or QApplication([])
    filename=tmp_path/'page.png'
    Image.new('RGB',(260,170),'white').save(filename)
    return EditorWindow(filename)


def test_enter_commits_one_caption_without_intermediate_undo(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.set_tool('text')
        editor._create_text_box((10,10,90,60))
        editor._text_in_canvas('abc')
        editor._text_in_canvas('abcd')
        assert len(editor.history)==0
        # Enter remains a newline in both text editors.
        editor.text_input.setPlainText('abc')
        editor.text_input.setFocus()
        QTest.keyClick(editor.text_input,Qt.Key.Key_Return)
        assert editor.text_input.toPlainText() == 'abc\\n'
        assert editor.layers[0].text == 'abc\\n'
        assert len(editor.history)==0
        QTest.keyClick(editor.text_input,Qt.Key.Key_Return,
                       Qt.KeyboardModifier.ControlModifier)
        assert len(editor.history)==1
        assert editor.selected_text==-1
        assert editor.layers[0].text=='abcd'
        assert editor.canvas.text_frame is None
        assert editor.canvas.text_items
        image=composite_text(editor.working,editor.layers)
        assert image.crop((10,10,100,70)).tobytes()!=editor.original.crop((10,10,100,70)).tobytes()
        editor.undo()
        assert not editor.layers
        editor.redo()
        assert editor.layers[0].text=='abcd'
    finally:
        editor.dirty=False
        editor.close()


def test_enter_in_canvas_text_editor_is_newline(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.set_tool('text')
        editor._create_text_box((10,10,90,60))
        item=editor.canvas.text_edit_item
        item.setPlainText('line1')
        item.setFocus()
        QTest.keyClick(editor.canvas.viewport(),Qt.Key.Key_Return)
        assert item.toPlainText() == 'line1\\n'
        assert not editor.history
        QTest.keyClick(editor.canvas.viewport(),Qt.Key.Key_Return,
                       Qt.KeyboardModifier.ControlModifier)
        assert editor.selected_text == -1
        assert len(editor.history)==1
    finally:
        editor.dirty=False
        editor.close()


def test_second_caption_preserves_first_and_switch_commits(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.set_tool('text')
        editor._create_text_box((10,10,90,60))
        editor._text_in_canvas('FIRST')
        editor._create_text_box((130,10,90,60))
        assert len(editor.history)==1
        assert editor.layers[0].text=='FIRST'
        assert editor.layers[1].text==''
        editor._text_in_canvas('SECOND')
        editor.set_tool('pencil')
        assert editor.selected_text==-1
        assert len(editor.history)==2
        assert [layer.text for layer in editor.layers]==['FIRST','SECOND']
        assert editor.canvas.text_frame is None
        editor.undo()
        assert [layer.text for layer in editor.layers]==['FIRST']
    finally:
        editor.dirty=False
        editor.close()


def test_cleanup_selection_switch_does_not_edit_or_enter_history(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.set_tool('cleanup')
        editor._selected((20,20,80,60))
        assert editor.selection is not None
        assert not editor.history
        editor.set_tool('pencil')
        assert editor.selection is None
        assert not editor.canvas.cleanup_boxes
        assert editor.canvas.cleanup_mask_item is None
        assert not editor.history
    finally:
        editor.dirty=False
        editor.close()


def test_paste_preview_is_opaque_and_commits_only_when_confirmed(tmp_path):
    editor=_editor(tmp_path)
    try:
        QApplication.clipboard().setImage(qimage_from_pil(
            Image.new('RGB',(18,15),'#ab3c20')))
        editor.paste_fragment(force_image=True)
        assert editor.canvas.paste_item.opacity()==1.0
        assert not editor.history
        editor.commit_paste()
        assert len(editor.history)==1
        editor.undo()
        assert editor.working.tobytes()==editor.original.tobytes()
    finally:
        editor.dirty=False
        editor.close()


def test_brush_stroke_is_single_transaction(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.set_tool('pencil')
        assert not editor.history
        editor._painted((QPointF(15,25),QPointF(15,25),'pencil',1,True))
        editor._painted((QPointF(15,25),QPointF(55,25),'pencil',1,False))
        assert not editor.history
        editor._painted((None,None,'pencil',1,False))
        assert len(editor.history)==1
        assert editor.working.getpixel((25,25))!=editor.original.getpixel((25,25))
        editor.undo()
        assert editor.working.tobytes()==editor.original.tobytes()
        editor.redo()
        assert editor.working.getpixel((25,25))!=editor.original.getpixel((25,25))
    finally:
        editor.dirty=False
        editor.close()


def test_pencil_one_pixel_and_blur_act_differently():
    original=Image.new('RGB',(110,90),'white')
    one=draw_stroke(original,original,(15,35),(85,35),1,(0,0,0),'pencil',100)
    assert one.getpixel((40,35))==(0,0,0)
    assert one.getpixel((40,34))==(255,255,255)
    assert one.getpixel((40,36))==(255,255,255)
    target=original.copy()
    ImageDraw.Draw(target).rectangle((40,25,64,55),fill='black')
    blurry=draw_stroke(target,target,(50,40),(55,40),28,(0,0,0),'blur',100)
    assert blurry.tobytes()!=target.tobytes()
    assert blurry.getpixel((0,0))==target.getpixel((0,0))
