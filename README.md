# Manga Offline Translator 0.4 (preview)

Windows x64, fully offline viewer and manually guided manga translator. **Version 0.4 changes the editor workflow and intentionally disables full-page text detection** because it frequently marked irrelevant picture areas and missed Japanese text.

## Viewer: Windows 7 inspired

- Picture occupies nearly the whole window by default. The small navigation toolbar is at the bottom.
- `Ctrl+O` open folder; arrows previous / next; `R` / `Shift+R` rotate; `F` fit; `1` 100%; wheel zoom; mouse drag pan; `F11` fullscreen / `Esc` exit.
- `Ctrl+L` toggles file/library panel. Search exact duplicate files/pixels via `Ctrl+D`; no automatic deletion. Show nested folders using the panel checkbox.
- `Ctrl+E` opens the current page in the translator editor.

## Editor: manual areas, local OCR

1. **Select text with the mouse:** the default tool creates a rectangle around the Japanese characters. The program will recognize only this selection, using bundled vertical/horizontal Japanese OCR, and suggest a text-placement boundary based on local balloon contours.
2. **Adjust the boundary:** a red polygon shows the insertion boundary. Use **Edit red contour** and drag white handles; click a red edge to add a handle. Blue dashed rectangle = removal mask; yellow rectangles = OCR glyph/word candidates. If automatic geometry is wrong, choose **Contour → rectangle** and draw a new rectangular boundary. Use **Change cleaning area** to change only the removal rectangle.
3. **Choose Translate, Hide, or Leave unchanged.** Edit the recognized text if OCR made mistakes, paste the translation, choose text color, font size (`0` = auto), bold or italic. Click **Apply changes**.
4. View **Original / After cleaning / With translation** before exporting. Export the cleaned image separately or the translated one. PNG is lossless; JPEG quality=100 still has lossy compression. Original images are never overwritten.
5. **Batch text exchange** retains `[0001]` style numbering for copying to an external translator and pasting back. It works on **one page** (the active editor).
6. **Save project / Open project** allows resuming all areas, boundaries and translations.

### Shortcuts

- `Ctrl+S` project save; `Ctrl+Z` / `Ctrl+Y` undo/redo; `Delete` remove selected area (when not typing); `Ctrl+C` copy OCR text; `Ctrl+V` paste translation; `Ctrl+T` bulk translations; `Ctrl+Enter` apply. Standard text-editor shortcuts work inside text fields.

### Constraints

- No automatic offline slang translator is included: small local models do not reliably preserve Japanese slang and meaning, and large LLMs would substantially increase package size. External translator copy/paste and direct user translation remain supported.
- OCR remains imperfect, even when limited to a manual area. Bounding boxes are **suggestions**; users can correct text and geometry. Automatic contour detection is conservative and can fall back to a rectangle. Complex textured backgrounds may show cleaning artifacts.
- No network calls occur during use. Windows installer includes Python runtime, Qt, OpenCV, Tesseract and `jpn`/`jpn_vert` OCR language data. No separate installation is required.

## Development and release

Requirements: Python 3.12, PySide6, Pillow, OpenCV, NumPy, pytesseract, pytest. To run source: `py -m pip install -r requirements.txt`, `py main.py`. Local OCR requires Tesseract + Japanese data.

Workflow `.github/workflows/windows-release.yml` uses a hosted Windows VM to run tests, launch Qt windows in offscreen smoke mode, package PyInstaller, compile Inno Setup, **install the generated installer in a temporary location**, test OCR/Qt from installed EXE, uninstall, and publish a preview release. The release is a standard HTTPS download on GitHub, independent of ChatGPT.

Files: `qt_viewer.py` picture-first viewer; `qt_editor.py` editor; `manual_regions.py` local OCR and contours; `engine.py` image processing; `viewer_core.py` exact duplicates; `main.py` entrypoint/tests. The project is work-in-progress and not digitally signed.