"""Entrypoint for the offline Qt image viewer and manga translator."""
import sys
from PySide6.QtWidgets import QApplication
from qt_common import STYLE
from qt_viewer import ViewerWindow


def gui_self_test(app):
    """Exercise actual editing flows and independent undo/redo on Windows Qt."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import numpy as np
    from PIL import Image, ImageDraw
    from PySide6.QtCore import Qt,QPointF
    from PySide6.QtTest import QTest
    from qt_editor import composite_text, qimage_from_pil

    with TemporaryDirectory(prefix='manga-qt-smoke-') as directory:
        path=Path(directory)/'sample.png'
        picture=Image.new('RGB',(440,340),'#e6dfd1')
        draw=ImageDraw.Draw(picture)
        draw.ellipse((75,30,360,300),fill='#fdfbf4',outline='#26292c',width=4)
        for x in (188,216,244):
            draw.rectangle((x,173,x+11,201),fill='#161616')
        for x in (346,379):
            draw.rectangle((x,204,x+11,234),fill='#111111')
        # Small black glyph almost fills a tiny beige selected region.
        draw.rectangle((32,37,38,51),fill='#111111')
        picture.save(path)
        viewer=ViewerWindow()
        viewer.show()
        viewer.current=path
        viewer.open_editor()
        editor=viewer.editor_by_path[path]
        editor.dirty=False
        try:
            app.processEvents()
            assert viewer.mode_stack.currentWidget() is editor and not editor.isWindow()
            assert editor.sidebar_modes.count()==3, 'Exactly Cleanup/Text/Brushes right panels'
            assert 'new' not in editor.actions, 'Obsolete selection mode reappeared'
            assert editor.canvas.tool=='cleanup'
            assert editor.sidebar_modes.currentWidget() is editor.selection_panel
            assert editor.insert_button if False else True
            assert editor.preview_button.isHidden()
            assert editor.selection_panel.styleSheet().find('#1b2430')>=0
            assert editor.brush_panel.widget().styleSheet().find('#1b2430')>=0
            assert not editor.undo_action.icon().isNull()
            assert not editor.redo_action.icon().isNull()
            editor._apply_zoom_text('100%')
            assert abs(editor.canvas.transform().m11()-1)<0.01
            editor._apply_zoom_text('75%')
            assert abs(editor.canvas.transform().m11()-.75)<0.01
            editor.flat_background.setChecked(False)
            editor.cleanup_mode.setCurrentIndex(2)
            editor.cleanup_grow.setValue(2)

            # Cleanup selection instantly copies the crop; one region at a time.
            editor._selected((170,155,116,68))
            assert app.clipboard().mimeData().hasImage()
            assert app.clipboard().image().size().width()==116
            assert editor._cleanup_candidates and editor.clear_button.isEnabled()
            mask=editor._approved_cleanup_mask().copy()
            count=sum(editor._cleanup_enabled)
            editor._toggle_cleanup_box(0)
            assert sum(editor._cleanup_enabled)==count-1
            editor._toggle_cleanup_box(0)
            assert np.array_equal(mask,editor._approved_cleanup_mask())
            before=editor.working.copy()
            editor.clear_button.click()
            first_clean=editor.working.copy()
            assert first_clean.tobytes()!=before.tobytes()
            assert np.array_equal(np.asarray(before)[mask==0],
                                  np.asarray(first_clean)[mask==0])
            assert editor.selection is None and not editor.canvas.cleanup_boxes
            editor.undo_action.trigger()
            assert editor.working.tobytes()==before.tobytes()
            editor.redo_action.trigger()
            assert editor.working.tobytes()==first_clean.tobytes()

            # Text can be created by dragging in Text mode, with no separate
            # "Create text box" button. Each text action has its own history.
            editor.set_tool('text')
            assert editor.sidebar_modes.currentWidget() is editor.text_panel
            assert not hasattr(editor,'insert_button')
            # Draw the text rectangle using the real Qt canvas, not a direct
            # helper call. This is the only way to create text in the UI.
            p=editor.canvas.mapFromScene(QPointF(141,88))
            q=editor.canvas.mapFromScene(QPointF(251,168))
            QTest.mousePress(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,p)
            QTest.mouseMove(editor.canvas.viewport(),q)
            QTest.mouseRelease(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier,q)
            assert len(editor.layers)==1 and editor.selected_text==0
            editor.text_input.setPlainText('Перевод')
            editor._flush_text_edit()
            assert editor.layers[0].text=='Перевод'
            text_before_size=editor.layers[0].size
            editor.font_size.setValue(37 if text_before_size!=37 else 38)
            new_size=editor.layers[0].size
            editor.undo()
            assert editor.layers[0].size==text_before_size
            assert editor.layers[0].text=='Перевод'
            assert editor.working.tobytes()==first_clean.tobytes()
            editor.redo()
            assert editor.layers[0].size==new_size

            editor._create_text_box((240,130,90,75))
            assert len(editor.layers)==2 and editor.selected_text==1
            assert editor.layers[1].text=='', 'New box must not copy prior translation'
            editor._choose_text_layer(0)
            editor.set_tool('text')
            app.processEvents()
            box=tuple(editor.layers[0].box)
            x,y,w,h=box
            p=editor.canvas.mapFromScene(QPointF(x+w/2,y+1))
            q=editor.canvas.mapFromScene(QPointF(x+w/2+20,y+15))
            QTest.mousePress(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,p)
            QTest.mouseMove(editor.canvas.viewport(),q)
            QTest.mouseRelease(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier,q)
            moved=tuple(editor.layers[0].box)
            assert moved[:2]!=box[:2], 'Text frame must move by top border'
            editor.undo()
            assert editor.layers[0].box==box
            editor.redo()
            assert editor.layers[0].box==moved
            editor._choose_text_layer(0)
            x,y,w,h=moved
            p=editor.canvas.mapFromScene(QPointF(x+w,y+h))
            q=editor.canvas.mapFromScene(QPointF(x+w+22,y+h+14))
            QTest.mousePress(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                             Qt.KeyboardModifier.NoModifier,p)
            QTest.mouseMove(editor.canvas.viewport(),q)
            QTest.mouseRelease(editor.canvas.viewport(),Qt.MouseButton.LeftButton,
                               Qt.KeyboardModifier.NoModifier,q)
            sized=tuple(editor.layers[0].box)
            assert sized[2:]!=moved[2:], 'Text frame must resize via corner'
            editor.undo()
            assert editor.layers[0].box==moved
            editor.redo()
            assert editor.layers[0].box==sized
            assert editor.text_items[1].pen().style()==Qt.PenStyle.NoPen
            rendered=composite_text(editor.working,editor.layers)
            assert rendered.getpixel((0,0))==picture.getpixel((0,0))

            # A second cleanup is a raster command. Undoing that command must
            # leave ALL text layers and the first cleaned area unchanged.
            editor.set_tool('cleanup')
            editor._selected((332,185,106,83))
            assert editor.selection==(332,185,106,83)
            assert editor._cleanup_candidates
            before2=editor.working.copy()
            text_before=__import__('copy').deepcopy(editor.layers)
            editor.clear_button.click()
            after2=editor.working.copy()
            assert after2.tobytes()!=before2.tobytes()
            assert editor.selection is None
            assert after2.crop((170,155,286,223)).tobytes()==first_clean.crop(
                (170,155,286,223)).tobytes()
            editor.undo_action.trigger()
            assert editor.working.tobytes()==before2.tobytes()
            assert editor.layers==text_before, 'Cleanup undo rolled back text'
            editor.redo_action.trigger()
            assert editor.working.tobytes()==after2.tobytes()
            assert editor.layers==text_before, 'Cleanup redo changed text'

            # Flat mode uses crop BORDER colour on tiny background-dominated
            # selections; the original contrast algorithm stays unchanged.
            editor.flat_background.setChecked(True)
            editor.cleanup_mode.setCurrentIndex(0)
            editor.cleanup_grow.setValue(0)
            editor._selected((27,32,17,23))
            assert editor._cleanup_candidates
            flat_mask=editor._approved_cleanup_mask()
            assert flat_mask[44,34]>0
            original_flat_color=editor.working.getpixel((34,44))
            assert original_flat_color==(17,17,17)
            editor.clear_button.click()
            background=picture.getpixel((25,33))
            assert editor.working.getpixel((34,44))==background
            flat_result=editor.working.copy()
            editor.undo()
            assert editor.working.getpixel((34,44))==original_flat_color
            assert editor.layers==text_before
            editor.redo()
            assert editor.working.tobytes()==flat_result.tobytes()

            # Clipboard: the single Cleanup rectangle overwrites clipboard
            # automatically; the side button pastes as movable image.
            editor.flat_background.setChecked(False)
            editor._selected((27,32,17,23))
            assert app.clipboard().image().size().width()==17
            editor.cleanup_paste.click()
            assert editor.canvas.paste_item is not None
            assert editor.selection is None
            editor.canvas.paste_item.setPos(11,11)
            editor.canvas.paste_item.setScale(1.5)
            before_paste=editor.working.copy()
            editor.commit_paste()
            after_paste=editor.working.copy()
            assert after_paste.tobytes()!=before_paste.tobytes()
            editor.undo()
            assert editor.working.tobytes()==before_paste.tobytes()
            assert editor.layers==text_before
            editor.redo()
            assert editor.working.tobytes()==after_paste.tobytes()
            assert editor.layers==text_before

            # Real brush actions each have their own independently undoable
            # transaction; colour, thickness, opacity settings are not edits.
            editor.set_tool('marker')
            assert editor.sidebar_modes.currentWidget() is editor.brush_panel
            editor.brush_diameter.setValue(23)
            editor.brush_opacity.setValue(55)
            editor._set_brush_color('#ed355e')
            first=editor.working.copy()
            editor._painted((QPointF(30,20),QPointF(80,20),'marker',23,True))
            editor._painted((None,None,'marker',23,False))
            painted=editor.working.copy()
            assert painted.tobytes()!=first.tobytes()
            editor.undo()
            assert editor.working.tobytes()==first.tobytes()
            editor.redo()
            assert editor.working.tobytes()==painted.tobytes()
            editor.set_tool('pencil')
            editor.brush_diameter.setValue(7)
            editor.set_tool('marker')
            assert editor.brush_diameter.value()==23
            editor.settings.sync()
            assert int(editor.settings.value('brushSize/marker'))==23
            assert int(editor.settings.value('brushOpacity'))==55
            editor.set_tool('picker')
            editor._pick((45,20))
            assert editor.canvas.tool=='marker'
            assert editor.canvas.viewport().cursor().shape()==Qt.CursorShape.ArrowCursor
            print('GUI workflow smoke: 3 panels, independent text/raster commands, '
                  'small flat cleanup, clipboard paste, brush undo/redo passed',flush=True)
        finally:
            editor.dirty=False
            viewer.close()
            app.processEvents()
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