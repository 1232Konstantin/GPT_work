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
        img.save(path)
        viewer=ViewerWindow();viewer.show()
        viewer.current=path
        viewer.open_editor()
        editor=viewer.editor_by_path[path]
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
            assert editor.canvas.tool=='new', 'Editor must start in manual rectangle mode'
            # Full paint-style editor smoke: rectangle goes to image clipboard,
            # text stays inside its box, other pixels remain unchanged.
            if viewer.mode_stack.currentWidget() is not editor or editor.isWindow():
                raise RuntimeError('Editor must open inside the same top-level window')
            editor._selected((130, 80, 125, 80))
            if not app.clipboard().mimeData().hasImage():
                raise RuntimeError('Selected fragment was not copied as an image')
            editor.text_input.setPlainText('Hello\\nworld!')
            editor.insert_text()
            if editor.canvas.tool != 'text_new':
                raise RuntimeError('Text tool must wait for independent rectangle')
            editor._create_text_box((145, 90, 110, 70))
            if len(editor.layers) != 1 or editor.canvas.text_edit_item is None:
                raise RuntimeError('Bounded text editor failed to initialize')
            editor.font_size.setValue(36)
            editor._create_text_box((270, 130, 95, 75))
            editor.font_size.setValue(22)
            editor._choose_text_layer(0)
            if editor.font_size.value() != 36 or editor.selected_text != 0:
                raise RuntimeError('Text-layer selection must restore its own font settings')
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
            editor.dirty = False
            print('GUI smoke test: one-window navigation, independent text layers, clipboard paste and compositing succeeded')
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
    if '--gui-self-test' in sys.argv:return gui_self_test(app)
    window=ViewerWindow();window.show()
    return app.exec()

if __name__=='__main__':sys.exit(main())