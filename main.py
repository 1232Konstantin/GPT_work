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
    from manual_regions import suggest_boundary, inner_text_box
    with TemporaryDirectory(prefix='manga-qt-smoke-') as directory:
        path=Path(directory)/'example.png'
        img=Image.new('RGB',(440,340),'#e6dfd1')
        draw=ImageDraw.Draw(img)
        draw.ellipse((75,30,360,300), fill='#fdfbf4',outline='#26292c',width=4)
        img.save(path)
        viewer=ViewerWindow();viewer.show()
        editor=EditorWindow(path);editor.show()
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
            poly=suggest_boundary(img,(175,120,30,35))
            if poly:
                box=inner_text_box(poly,(175,120,30,35),img.size)
                assert box[2]>0 and box[3]>0
            print('GUI smoke test: viewer, editor and manual tools initialized')
        finally:
            editor.close();viewer.close();app.processEvents()
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