# Manga Offline Translator 0.7.0 — offline viewer and Paint-style editor

Windows x64 desktop app based on PySide6, Pillow and OpenCV. Fully offline; no cloud or OCR is called during manual editing. The bundled legacy OCR engine is not needed for manual cleanup.

## Workflow

1. Open an image in the viewer and choose **Перевести страницу** (Ctrl+E). The editor opens **in the same window**.
2. Drag a selection around the original lettering. Its original-size crop is **immediately copied as an image to the Windows clipboard** for manual translation in a browser.
3. The editor highlights the detected **actual ink pixels** in pink and draws green candidate boxes. Click a box to exclude it (grey); click again to restore. Choose text type: dark, light, light-outlined or dark-outlined. Press **Очистить**. Ctrl+Z undoes the result. No preview step is required.
4. Optional: **⋯ > Расширенные настройки очистки** contains Telea/linear/quadratic reconstruction and the 0–3 pixel mask-edge expansion. Settings persist between sessions. The accepted contrast-cleanup algorithm remains OCR-free and **never changes any pixel outside its approved mask**.
5. Choose **Текст** on the top toolbar, then **draw an independent rectangle** for translated text. Type or paste the translation. The right-hand Text panel controls font, pixel size, color, bold, italic, underline and left/center/right alignment. Double-click an existing text box to return to it; every box keeps its own formatting until final raster export.
6. Choose **Кисти** for the Paint 3D-inspired page: marker, calligraphy, oil paint, watercolor, pixel pen, pencil, crayon, restore eraser, spray, fill and palette knife. Thickness **per brush**, opacity, selected tool and color persist. The pipette shows pixel color in a magnified cursor bubble and picks it on click.
7. A selection can be copied (Ctrl+C), cut (Ctrl+X) or pasted (Ctrl+V) as an **image**. The toolbar and right-click context menu have explicit copy/paste commands. Pasted images float above the page until confirmation: drag to move, drag the lower-right corner to scale, Enter/Ctrl+Enter to commit, Esc to cancel. When a text field has keyboard focus, ordinary Ctrl+V pastes text instead.
8. Save as lossless **PNG** (Ctrl+S). The source image is never overwritten.

## Shortcuts

- Alt+S: select; Alt+T: new text box; Alt+B: brushes; Alt+C: clean approved ink
- Ctrl+C / Ctrl+V / Ctrl+X: copy / paste / cut (text fields use ordinary text clipboard)
- Ctrl+Z / Ctrl+Y: undo / redo; Delete: delete selection or selected text object
- Enter / Ctrl+Enter: apply floating pasted image; Esc: cancel
- Mouse wheel: zoom under cursor; original pixels remain unchanged

## Important details

- Text outlines can be **light or dark**. The automatic contrast mask is a proposal, not semantic letter recognition; high-contrast illustration may also be detected. Check the pink mask and disable unwanted boxes before pressing Clean.
- A manually drawn cleanup rectangle (⋯ > Рамка очистки вручную) explicitly authorizes its entire interior. Automatic boxes authorize only segmented ink pixels.
- Fine details of Paint 3D brush rendering differ from Microsoft's proprietary application. This implementation provides independently functioning brush styles, brush controls and pixel-level eyedropper behavior.
- All editing works at source image resolution. Zooming is a display operation, with no destructive rescaling of the background. Editable text is composited only when exporting.
- Font, underline, alignment, brush size per tool, opacity, colors, selected brush and advanced cleanup settings persist via Qt QSettings.

## Build and Windows validation

Run python -m pytest -q tests for image-operation tests and python main.py --gui-self-test for interaction checks. GitHub Actions builds a Windows 64-bit PyInstaller/Inno Setup bundle, runs the full suite, launches the GUI and checks install/run/uninstall of the packaged application before publishing a preview release.
