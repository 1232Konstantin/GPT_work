# Manga Offline Translator — manual editor 0.5.0

A personal, entirely offline image viewer and paint-style manga lettering editor for Windows 64-bit. Built with Python, Qt (PySide6), OpenCV, and Pillow. No subscription, account, cloud service or Google API is required.

## Workflow

1. In the viewer, open an image, then choose **Перевести страницу** via the file menu (or `Ctrl+E`).
2. In the editor, drag a rectangle over a Japanese inscription. **The crop is immediately copied to the Windows clipboard as an IMAGE**. Paste it into an image translator in your browser (external step; the editor itself stays offline).
3. Press **Очистить буквы**. Only a conservative mask of detected ink strokes inside your rectangle is inpainted; the selection rectangle and whole balloon are never passed as a cleanup mask. If needed undo with `Ctrl+Z`, refine the selection, or use the **Восстановить** brush to recover original pixels.
4. Paste your translation in the right-hand text input, and press **Вставить текст в зону** (or choose **Текст**). A text box is placed over the same rectangle. Edit the text right on the image, insert spaces and manual line breaks; the text clips at the box's edges. Resize its bottom-right corner. Click an older text region in text mode to select it again.
5. Choose font family, size, bold, italic, and text color. These defaults are stored in Qt `QSettings` and survive restarting the app.
6. Touch up with **Кисть**, **Восстановить**, **Мастихин** (smudge), and **Пипетка**. Image fragments can be copied with `Ctrl+C` and pasted with `Ctrl+V`; move a floating pasted fragment without affecting the current zoom, apply with `Ctrl+Enter` or discard with `Esc`.
7. Save using **PNG** / `Ctrl+S`. The original source file is never overwritten. Text, including out-of-bounds content, is clipped to its rectangle. PNG output is lossless.

## Shortcuts

* `Ctrl+C`: copy selected image rectangle (or highlighted text when typing).
* `Ctrl+V`: paste image as movable fragment (or paste text into the active text editor).
* `Ctrl+X`: copy selection and restore original pixels in it.
* `Ctrl+Z` / `Ctrl+Y`: undo / redo.
* `Delete`: remove active text box or cancel an empty selection.
* `Alt+S/B/E/M/I/T/C`: Select / Brush / Restore / Smudge / Pick color / Text / Clean ink.
* `Alt+F` / `Alt+1`: Fit page / 100% actual size. Mouse wheel zooms at cursor.

## Key guarantees

* The editor never calls OCR or an external program during its workflow, avoiding the flashing Tesseract console window.
* Viewer and editor never save or rescale the original on zoom. Zoom is display-only; editor display uses source-pixel Lanczos downsampling, with Qt smooth interpolation during interaction.
* All manual edits happen in a working copy, and the original is never overwritten by the export dialog.
* The cleanup step is deliberately conservative and may leave residue around stylized lettering. Use manual touch-up rather than expecting perfect automatic restoration on patterned backgrounds.

## Windows setup

The GitHub Actions `Windows installer` workflow builds a self-contained 64-bit installer with PyInstaller/Inno Setup, executes all algorithm tests, launches real GUI smoke tests, installs and runs the finished package and finally uninstalls it before publishing a preview GitHub release.

The bundled Tesseract from earlier versions is kept for legacy API compatibility and packaging checks, but **the new manual editor does not use it**.