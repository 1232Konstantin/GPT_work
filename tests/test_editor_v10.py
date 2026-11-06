"""v0.10 user-acceptance regressions: committed operations only."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")

from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QTextCursor
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
        editor.text_input.moveCursor(QTextCursor.MoveOperation.End)
        editor.text_input.setFocus()
        QTest.keyClick(editor.text_input,Qt.Key.Key_Return)
        assert editor.text_input.toPlainText() == 'abc\n'
        assert editor.layers[0].text == 'abc\n'
        assert len(editor.history)==0
        QTest.keyClick(editor.text_input,Qt.Key.Key_Return,
                       Qt.KeyboardModifier.ControlModifier)
        assert len(editor.history)==1
        assert editor.selected_text==-1
        assert editor.layers[0].text=='abc\n'
        assert editor.canvas.text_frame is None
        assert editor.canvas.text_items
        image=composite_text(editor.working,editor.layers)
        assert image.crop((10,10,100,70)).tobytes()!=editor.original.crop((10,10,100,70)).tobytes()
        editor.undo()
        assert not editor.layers
        editor.redo()
        assert editor.layers[0].text=='abc\n'
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
        cursor=item.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        item.setTextCursor(cursor)
        item.setFocus()
        QTest.keyClick(editor.canvas.viewport(),Qt.Key.Key_Return)
        assert item.toPlainText() == 'line1\n'
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


def test_cleanup_after_text_brush_and_overlapping_previous_cleanup(tmp_path):
    """Previously cleaned rectangles must not suppress newly written ink."""
    app = QApplication.instance() or QApplication([])
    source=tmp_path/'page_with_glyph.png'
    original=Image.new('RGB',(260,170),'white')
    ImageDraw.Draw(original).rectangle((32,27,45,52),fill='black')
    original.save(source)
    editor=EditorWindow(source)
    try:
        editor.cleanup_mode.setCurrentIndex(
            editor.cleanup_mode.findData('flat'))
        selection=(20,15,60,52)
        editor._selected(selection)
        assert editor._cleanup_candidates
        assert editor._approved_cleanup_mask().any()
        editor.clear_selection()
        assert editor.selection is None
        assert editor.working.getpixel((38,40)) != (0,0,0)

        editor._create_text_box((140,20,90,60))
        editor._text_in_canvas('translation')
        editor.commit_text()
        editor.set_tool('pencil')
        editor._set_brush_color('#000000')
        editor._painted((QPointF(38,27),QPointF(38,51),'pencil',8,True))
        editor._painted((None,None,'pencil',8,False))
        assert editor.working.getpixel((38,40))==(0,0,0)

        editor.set_tool('cleanup')
        # Same area previously used for cleanup, before brush+text.
        editor._selected(selection)
        assert editor._cleanup_candidates, 'New dark ink must be detected'
        assert all(editor._cleanup_enabled), 'Previous cleanup must not disable it'
        assert editor.clear_button.isEnabled()
        assert editor.canvas.cleanup_boxes
        # Different but overlapping selection must also work.
        editor._selected((22,16,60,52))
        assert editor._cleanup_candidates
        assert all(editor._cleanup_enabled)
        editor.clear_selection()
        assert editor.working.getpixel((38,40))!=(0,0,0)
        editor.undo()
        assert editor.working.getpixel((38,40))==(0,0,0)
        editor.redo()
        assert editor.working.getpixel((38,40))!=(0,0,0)
    finally:
        editor.dirty=False
        editor.close()


def test_undo_keeps_more_than_twelve_compressed_image_states(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.set_tool('pencil')
        editor._set_brush_color('#000000')
        for i in range(25):
            p=QPointF(14+i*6,33)
            editor._painted((p,p,'pencil',1,True))
            editor._painted((None,None,'pencil',1,False))
        assert len(editor.history)==25
        assert len(editor.history[0][1][2]) < len(editor.original.tobytes())
        for _ in range(25):
            editor.undo()
        assert editor.working.tobytes()==editor.original.tobytes()
        for _ in range(25):
            editor.redo()
        assert editor.working.getpixel((14+24*6,33))==(0,0,0)
    finally:
        editor.dirty=False
        editor.close()


def test_picker_preview_and_click_sample_same_fractional_pixel(tmp_path):
    editor=_editor(tmp_path)
    try:
        editor.working.putpixel((40,50),(245,15,15))
        editor.working.putpixel((41,50),(15,225,15))
        editor._sync()
        editor.set_tool('picker')
        position=QPointF(40.8,50.3)
        preview=editor.canvas.picker_color(position)
        assert preview.name()=='#f50f0f'
        editor._pick(preview)
        assert editor.brush_color.name()==preview.name()
        editor.set_tool('picker')
        editor._pick((40.8,50.3))
        assert editor.brush_color.name()==preview.name()
    finally:
        editor.dirty=False
        editor.close()
