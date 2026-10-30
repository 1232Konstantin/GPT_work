"""Windows application entrypoint (no network services)."""
import sys
from PySide6.QtWidgets import QApplication
from qt_common import STYLE
from qt_viewer import ViewerWindow


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
    win = ViewerWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())