"""Windows application entrypoint (no network services)."""
import sys
from PySide6.QtWidgets import QApplication
from qt_common import STYLE
from qt_viewer import ViewerWindow


def gui_self_test(app):
    """Exercise the actual Qt viewer and editor in a Windows build runner."""
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from PIL import Image, ImageDraw
    from qt_editor import EditorWindow

    with TemporaryDirectory(prefix='manga-ui-smoke-') as directory:
        path = Path(directory) / 'example.png'
        canvas = Image.new('RGB', (400, 280), '#e6dfd2')
        draw = ImageDraw.Draw(canvas)
        draw.ellipse((90, 35, 310, 240), fill='#fffdf6', outline='#202933', width=3)
        canvas.save(path)

        viewer = ViewerWindow()
        viewer.show()
        editor = EditorWindow(path)
        editor.show()
        try:
            app.processEvents()
            if not viewer.isVisible() or not editor.isVisible():
                raise RuntimeError('Qt viewer/editor windows did not become visible')
            if editor.canvas.scene().sceneRect().width() != 400:
                raise RuntimeError('Qt editor failed to load a test page')
            if not viewer.canvas.scene():
                raise RuntimeError('Qt viewer scene was not initialized')
            print('GUI smoke test: Qt viewer and editor created successfully')
        finally:
            editor.close()
            viewer.close()
            app.processEvents()
    return 0


def main():
    if '--self-test' in sys.argv:
        # Runs on the packaged executable without opening a GUI. Validates OCR assets.
        from engine import configure_ocr
        command, languages = configure_ocr()
        if not {'jpn', 'jpn_vert'} <= languages:
            raise RuntimeError('Missing bundled jpn/jpn_vert OCR languages')
        print(f'OCR ready: {command} | {sorted(languages)}')
        return 0
    app = QApplication(sys.argv)
    app.setApplicationName('Manga Offline Translator')
    app.setOrganizationName('MangaOffline')
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    if '--gui-self-test' in sys.argv:
        return gui_self_test(app)
    win = ViewerWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())