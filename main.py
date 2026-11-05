"""Entrypoint for the offline Qt image viewer and manga translator."""
import sys
from PySide6.QtWidgets import QApplication
from qt_common import STYLE
from qt_viewer import ViewerWindow


def gui_self_test(app):
    """Run a real Qt smoke test both before and after Windows installation."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from PIL import Image, ImageDraw
    from qt_editor import EditorWindow
    from qt_editor import composite_text
    with TemporaryDirectory(prefix='manga-qt-smoke-') as directory:
        path=Path(directory)/'example.png'
        img=Image.new('RGB',(440,340),'#e6dfd1')
        draw=ImageDraw.Draw(img)
        draw.ellipse((75,30,360,300), fill='#fdfbf4',outline='#26292c',width=4)
        for gx in (188,216,244):
            draw.rectangle((gx,173,gx+11,201),fill='#161616')
        for gx in (346,379):
            draw.rectangle((gx,204,gx+11,234),fill='#111111')
        img.save(path)
        viewer=ViewerWindow();viewer.show()
        viewer.current=path
        viewer.open_editor()
        editor=viewer.editor_by_path[path]
        # The GUI test is executed three times (Python, packaged and installed).
        # Never assume user defaults: QSettings deliberately persists controls.
        editor.underline_button.setChecked(False)
        editor.align_buttons['left'].click()
        try:
            app.processEvents()
            assert viewer.isVisible() and editor.isVisible()
            assert editor.canvas.scene().sceneRect().width()==440
            assert viewer.canvas.scene() is not None
            viewer.canvas.set_picture(img)
            if viewer.canvas.pixmap_item.transformationMode() != __import__('PySide6.QtCore', fromlist=['Qt']).Qt.TransformationMode.SmoothTransformation:
                raise RuntimeError('Viewer image item is using low-quality scaling')
            if viewer.title_label.isVisible():
                raise RuntimeError('Viewer must not consume space for a permanent filename header')
            if viewer.actual_action.text() != '100%':
                raise RuntimeError('Viewer must expose a clearly labelled 100% button')
            assert not viewer.library_panel.isVisible(), 'Viewer must start image-first'
            assert editor.canvas.tool=='cleanup', 'Editor must start in cleanup rectangle mode'
            # Full paint-style editor smoke: rectangle goes to image clipboard,
            # text stays inside its box, other pixels remain unchanged.
            print('Single-window diagnostic:',viewer.mode_stack.currentWidget() is editor,
                  'editor.isWindow=',editor.isWindow(),flush=True)
            if viewer.mode_stack.currentWidget() is not editor or editor.isWindow():
                raise RuntimeError('Editor must open inside the same top-level window')
            editor.set_tool('cleanup')
            editor._selected((130, 80, 125, 80))
            if not app.clipboard().mimeData().hasImage():
                raise RuntimeError('Cleanup selection must copy fragment automatically')
            editor.copy_action.trigger()
            if not app.clipboard().mimeData().hasImage():
                raise RuntimeError('Copy action did not copy selected image')
            editor.text_input.setPlainText('Hello\\nworld!')
            editor.insert_text()
            if editor.canvas.tool != 'text':
                raise RuntimeError('Text tool must wait for independent rectangle')
            editor._create_text_box((145, 90, 110, 70))
            editor._text_in_canvas('Hello\\nworld!')
            if len(editor.layers) != 1 or editor.canvas.text_edit_item is None:
                raise RuntimeError('Bounded text editor failed to initialize')
            editor.font_size.setValue(36)
            editor._create_text_box((270, 130, 95, 75))
            editor._text_in_canvas('SECOND CAPTION')
            editor.font_size.setValue(22)
            editor._choose_text_layer(0)
            if editor.font_size.value() != 36 or editor.selected_text != 0:
                raise RuntimeError('Text-layer selection must restore its own font settings')
            # Actual mouse interaction: drag a text frame by the top border,
            # resize at its bottom-right corner, undo and redo the geometry.
            from PySide6.QtTest import QTest
            from PySide6.QtCore import Qt, QPoint, QPointF
            editor.set_tool('text')
            app.processEvents()
            initial=tuple(editor.layers[0].box)
            x,y,w,h=initial
            p=editor.canvas.mapFromScene(QPointF(x+w/2,y+1))
            q=editor.canvas.mapFromScene(QPointF(x+w/2+20,y+14))
            QTest.mousePress(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,p)
            QTest.mouseMove(editor.canvas.viewport(),q)
            QTest.mouseRelease(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier,q)
            moved=tuple(editor.layers[0].box)
            if moved[:2]==initial[:2]:
                raise RuntimeError('Dragging upper text border must move text frame')
            editor.undo()
            if tuple(editor.layers[0].box)!=initial:
                raise RuntimeError('Undo must restore text position')
            editor.redo()
            if tuple(editor.layers[0].box)!=moved:
                raise RuntimeError('Redo must restore moved text frame')
            editor._choose_text_layer(0)
            x,y,w,h=moved
            p=editor.canvas.mapFromScene(QPointF(x+w,y+h))
            q=editor.canvas.mapFromScene(QPointF(x+w+22,y+h+14))
            QTest.mousePress(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,p)
            QTest.mouseMove(editor.canvas.viewport(),q)
            QTest.mouseRelease(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier,q)
            if editor.layers[0].box[2:] == moved[2:]:
                raise RuntimeError('Dragging text corner must resize text frame')
            editor.undo()
            if tuple(editor.layers[0].box)!=moved:
                raise RuntimeError('Resize text must be undoable')
            editor.redo()
            if tuple(editor.layers[0].box[2:])==moved[2:]:
                raise RuntimeError('Resize text must be redoable')
            viewer._return_to_viewer()
            if viewer.mode_stack.currentIndex() != 0:
                raise RuntimeError('Back must show the original viewer')
            viewer.open_editor()
            if viewer.mode_stack.currentWidget() is not editor or len(editor.layers)!=2:
                raise RuntimeError('Returning to editor must preserve unsaved layers')
            # External picture pasted from clipboard is a movable image object.
            from qt_editor import qimage_from_pil
            app.clipboard().setImage(qimage_from_pil(Image.new('RGBA',(12,12),'#e45d45')))
            editor.paste_fragment(force_image=True)
            if editor.canvas.paste_item is None:
                raise RuntimeError('Paste image from clipboard failed')
            editor.canvas.cancel_paste()
            rendered = composite_text(editor.working, editor.layers)
            if rendered.getpixel((0, 0)) != editor.original.getpixel((0, 0)):
                raise RuntimeError('Rendering text corrupted pixels outside its bounds')
            if rendered.crop((145, 90, 255, 160)).tobytes() == editor.original.crop((145, 90, 255, 160)).tobytes():
                raise RuntimeError('Translated text not drawn in selected area')
            # Cleanup remains a suggestion until the user confirms boxes.
            import numpy as np
            editor.cleanup_mode.setCurrentIndex(2) # outlined text
            editor.cleanup_grow.setValue(2)
            editor.set_tool('cleanup')
            editor._selected((170,155,116,68))
            if not editor._cleanup_candidates:
                raise RuntimeError('No contrast components suggested for synthetic characters')
            selected_count=sum(editor._cleanup_enabled)
            full_mask=editor._approved_cleanup_mask()
            if not np.any(full_mask) or editor.canvas.cleanup_mask_item is None:
                raise RuntimeError('No visible pixel mask constructed')
            editor._toggle_cleanup_box(0)
            if np.count_nonzero(editor._approved_cleanup_mask()) >= np.count_nonzero(full_mask):
                raise RuntimeError('Disabling a component did not remove pixels from mask')
            # Changing the repair method must preserve the user's decisions.
            editor.cleanup_method.setCurrentIndex(1)
            if editor._cleanup_enabled[0] or sum(editor._cleanup_enabled)!=selected_count-1:
                raise RuntimeError('Changing repair method reset approved regions')
            editor.cleanup_method.setCurrentIndex(0)
            editor._toggle_cleanup_box(0)
            if not np.array_equal(full_mask,editor._approved_cleanup_mask()):
                raise RuntimeError('Enabling a component did not restore original mask')
            before=editor.working.copy()
            mask=editor._approved_cleanup_mask().copy()
            boxes=[box for box,chosen in zip(editor._cleanup_candidates,editor._cleanup_enabled) if chosen]
            editor.preview_button.click()
            if not editor._cleanup_preview_active or editor.working.tobytes()!=before.tobytes():
                raise RuntimeError('Cleanup preview must not change working pixels')
            shown=editor.canvas.current_image.tobytes()
            if shown==before.tobytes():
                raise RuntimeError('Preview itself did not show any cleanup')
            # Crucially, click the ACTUAL QPushButton while PREVIEW IS SHOWN.
            if not editor.clear_button.isEnabled():
                raise RuntimeError('Apply button disabled while approved boxes are present')
            editor.clear_button.click()
            after=editor.working.copy()
            if after.tobytes()!=shown:
                raise RuntimeError('Clicking Apply must commit EXACTLY what preview displayed')
            if editor.canvas.current_image.tobytes()!=after.tobytes():
                raise RuntimeError('Qt canvas kept showing the old image after Apply')
            if editor.canvas.pixmap_item.pixmap().size().width()!=editor.working.width:
                raise RuntimeError('Pixmap geometry is not updated after applying cleanup')
            protected=np.ones((before.height,before.width),dtype=bool)
            for x,y,w,h in boxes:protected[y:y+h,x:x+w]=False
            if not np.array_equal(np.asarray(before)[protected],np.asarray(after)[protected]):
                raise RuntimeError('Cleanup modified pixels outside confirmed rectangles')
            if not np.array_equal(np.asarray(before)[mask==0],np.asarray(after)[mask==0]):
                raise RuntimeError('Cleanup modified any pixels outside its fine-grained ink mask')
            if not np.array_equal(editor._cleanup_last_applied_mask,mask):
                raise RuntimeError('Applied mask differs from previewed selection')
            if editor.clear_button.isEnabled():
                raise RuntimeError('Apply must disable itself after successful cleanup')
            editor.clear_selection()
            if editor.working.tobytes()!=after.tobytes():
                raise RuntimeError('Repeated cleanup must be a no-op')
            editor.canvas.setFocus()
            editor.undo()
            if editor.working.tobytes()!=before.tobytes():
                raise RuntimeError('Undo did not restore pixels before cleanup')
            editor.redo()
            if editor.working.tobytes()!=after.tobytes():
                raise RuntimeError('Redo did not restore the approved cleanup')
            if editor.selection is not None or editor.canvas.cleanup_boxes:
                raise RuntimeError('Successful cleanup must remove all stale selections')
            first_done=editor.working.crop((170,155,286,223)).tobytes()
            editor.insert_text()
            editor._create_text_box((170,150,116,70))
            editor.set_tool('cleanup')
            editor._selected((325,185,105,80))
            if not editor._cleanup_candidates:
                raise RuntimeError('Second cleanup region did not detect glyphs')
            second_before=editor.working.copy()
            editor.clear_button.click()
            if editor.working.tobytes()==second_before.tobytes():
                raise RuntimeError('Second cleanup did not modify new region')
            if editor.working.crop((170,155,286,223)).tobytes()!=first_done:
                raise RuntimeError('Second cleanup unexpectedly modified first cleaned region')
            if editor.selection is not None:
                raise RuntimeError('Second cleaned selection was not cleared')
            # UI regressions: simplified cleanup and a distinct Paint-3D-like
            # brushes page. Test actual controls, not only the backing methods.
            from PySide6.QtCore import QPointF
            editor.set_tool('cleanup')
            assert editor.sidebar_modes.currentWidget() is editor.selection_panel
            assert editor.sidebar_modes.count() == 3
            assert editor.selection_panel.styleSheet().find('#1b2430')>=0
            assert editor.brush_panel.widget().styleSheet().find('#1b2430')>=0
            assert not editor.undo_action.icon().isNull()
            assert not editor.redo_action.icon().isNull()
            editor._apply_zoom_text('100%')
            assert abs(editor.canvas.transform().m11()-1)<0.01
            editor._apply_zoom_text('75%')
            assert abs(editor.canvas.transform().m11()-.75)<0.01
            assert editor.zoom_combo.currentText()=='75%'
            assert editor.preview_button.isHidden()
            assert editor.cleanup_stats.isHidden()
            assert editor.clear_button.text() == 'Очистить'
            assert editor.cleanup_mode.findData('outline_dark')>=0

            editor.set_tool('pencil')
            assert editor.sidebar_modes.currentWidget() is editor.brush_panel
            assert len(editor.brush_tool_buttons)==6
            assert editor.actions['brushes'].isChecked()
            editor.brush_diameter.setValue(23)
            editor.brush_opacity.setValue(55)
            editor._set_brush_color('#ed355e')
            assert editor.brush_color.name()=='#ed355e'
            before_brush=editor.working.copy()
            editor._painted((QPointF(30,22),QPointF(76,22),'pencil',23,True))
            editor._painted((None,None,'pencil',23,False))
            assert editor.working.getpixel((48,22))!=before_brush.getpixel((48,22))
            assert editor.working.getpixel((5,5))==before_brush.getpixel((5,5))
            editor.set_tool('spray')
            editor.brush_diameter.setValue(7)
            editor.set_tool('pencil')
            assert editor.brush_diameter.value()==23, 'Brush thickness was not remembered per tool'
            assert editor.brush_opacity.value()==55
            editor.settings.sync()
            assert int(editor.settings.value('brushSize/pencil'))==23
            assert int(editor.settings.value('brushOpacity'))==55
            editor.set_tool('picker')
            editor._pick((48,22))
            assert editor.canvas.tool=='pencil', 'Picker must return to the previous brush'
            assert editor.brush_color.isValid()
            assert not editor.canvas.viewport().cursor().pixmap().isNull(), (
                'Picker must restore the chosen brush cursor after sampling')

            # Every text layer owns underline, alignment and other formatting.
            editor._choose_text_layer(0)
            editor.underline_button.setChecked(True)
            editor.align_buttons['center'].click()
            assert editor.layers[0].underline
            assert editor.layers[0].alignment=='center'
            editor._choose_text_layer(1)
            assert not editor.underline_button.isChecked()
            assert editor.layers[1].alignment=='left'
            editor._choose_text_layer(0)
            assert editor.underline_button.isChecked()
            assert editor.align_buttons['center'].isChecked()
            editor.settings.sync()
            assert str(editor.settings.value('fontAlign'))=='center'
            assert str(editor.settings.value('fontUnderline')).lower()=='true'

            # An image crop is copied as image even when the text field owns focus.
            editor.set_tool('cleanup')
            editor.text_input.setFocus()
            from qt_editor import qimage_from_pil
            stored=Image.new('RGBA',(9,9),'#123456')
            app.clipboard().setImage(qimage_from_pil(stored))
            editor._selected((27,22,55,41))
            assert app.clipboard().image().size().width()==55, 'Cleanup selection failed to overwrite clipboard'
            menu=editor._make_context_menu(QPoint(100,100))
            item_labels=[a.text() for a in menu.actions() if not a.isSeparator()]
            assert item_labels==['Копировать','Вырезать','Вставить','Снять выделение'], item_labels
            editor.copy_action.trigger()
            assert app.clipboard().mimeData().hasImage()
            assert app.clipboard().image().size().width()==55
            from qt_editor import qimage_from_pil
            fragment=Image.new('RGBA',(15,12),'#e45d45')
            app.clipboard().setImage(qimage_from_pil(fragment))
            editor.paste_action.trigger()
            assert editor.canvas.paste_item is not None
            editor.canvas.paste_item.setPos(8,8)
            editor.canvas.paste_item.setScale(2.0)
            editor.commit_paste()
            assert editor.working.getpixel((15,15))[:3]==(228,93,69), 'Resized pasted image was not committed'
            editor.canvas.setFocus()
            editor.undo()
            assert editor.working.getpixel((15,15))[:3]!=(228,93,69)
            editor.redo()
            assert editor.working.getpixel((15,15))[:3]==(228,93,69)
            editor.undo()
            assert editor.working.getpixel((15,15))[:3]!=(228,93,69)
            print('GUI smoke test: Paint-style brushes, persistent tool settings, text formatting and image clipboard succeeded')
            editor.dirty = False
            print('GUI smoke test: contrast pixel mask, exclusion, mode selection, preview/application and undo/redo succeeded')
        finally:
            editor.dirty=False
            viewer.close();app.processEvents()
    return 0


def main():
    if '--self-test' in sys.argv:
        from engine import configure_ocr
        command,languages=configure_ocr()
        if not {'jpn','jpn_vert'}<=languages:
            raise RuntimeError('Bundled Japanese OCR models are missing')
        print('OCR ready:',command,sorted(languages))
        return 0
    app=QApplication(sys.argv)
    app.setApplicationName('Manga Offline Translator')
    app.setOrganizationName('MangaOffline')
    app.setStyle('Fusion');app.setStyleSheet(STYLE)
    if '--gui-self-test' in sys.argv:
        try:
            return gui_self_test(app)
        except BaseException:
            from pathlib import Path
            import tempfile
            import traceback
            diagnostic=Path(tempfile.gettempdir())/'manga_gui_smoke_error.txt'
            diagnostic.write_text(traceback.format_exc(),encoding='utf-8')
            return 17
    window=ViewerWindow();window.show()
    return app.exec()

if __name__=='__main__':sys.exit(main())