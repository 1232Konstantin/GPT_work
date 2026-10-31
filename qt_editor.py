"""Paint-inspired manual manga lettering editor.

Work flow: rectangle -> clipboard image -> conservative ink cleanup -> editable,
bounded text box. No OCR, subprocesses, cloud APIs or automatic regions.
"""
from __future__ import annotations

from copy import deepcopy
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from PIL import Image
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QSettings, QThreadPool, QTimer
from PySide6.QtGui import (QAction, QColor, QFont, QPainter, QPen, QBrush,
                           QImage, QKeySequence, QTextDocument, QTextOption)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QSplitter, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsItem,
    QPushButton, QToolButton, QToolBar, QPlainTextEdit, QSpinBox,
    QFontComboBox, QColorDialog, QFileDialog, QMessageBox, QFrame)

from engine import open_image
from paint_core import clip_box, clear_letters, brush_segment, palette_smudge
from qt_common import pixmap_from_pil, Task


@dataclass
class TextLayer:
    box: tuple[int, int, int, int]
    text: str = ''
    family: str = 'Arial'
    size: int = 28
    color: str = '#202020'
    bold: bool = False
    italic: bool = False


def font_for(layer: TextLayer) -> QFont:
    f = QFont(layer.family)
    f.setPixelSize(layer.size)  # image pixels, not device-dependent points
    f.setBold(layer.bold)
    f.setItalic(layer.italic)
    return f


def qimage_from_pil(image: Image.Image) -> QImage:
    return pixmap_from_pil(image).toImage().convertToFormat(QImage.Format.Format_RGBA8888)


def qimage_to_pil(image: QImage) -> Image.Image:
    q = image.convertToFormat(QImage.Format.Format_RGBA8888)
    raw = bytes(q.constBits())
    return Image.frombytes('RGBA', (q.width(), q.height()), raw, 'raw', 'RGBA', q.bytesPerLine()).copy()


def composite_text(image: Image.Image, layers: list[TextLayer]) -> Image.Image:
    """Draw text with Qt's same font/layout engine used by the live scene.

    Graphics are clipped to each independent text rectangle. No scaling or
    lossy round-trip of background pixels happens during viewing or exporting.
    """
    q = qimage_from_pil(image)
    painter = QPainter(q)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    for layer in layers:
        x, y, w, h = clip_box(layer.box, image.size)
        if w < 2 or h < 2 or not layer.text:
            continue
        doc = QTextDocument()
        doc.setDocumentMargin(0)
        doc.setDefaultFont(font_for(layer))
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        doc.setDefaultTextOption(option)
        doc.setPlainText(layer.text)
        doc.setTextWidth(max(1, w - 6))
        painter.save()
        painter.setClipRect(QRectF(x, y, w, h))
        painter.translate(x + 3, y + 2)
        painter.setPen(QColor(layer.color))
        # QTextDocument uses its character format for foreground, not QPainter's pen.
        from PySide6.QtGui import QTextCursor, QTextCharFormat
        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.SelectionType.Document)
        charfmt = QTextCharFormat()
        charfmt.setForeground(QBrush(QColor(layer.color)))
        cursor.mergeCharFormat(charfmt)
        doc.drawContents(painter, QRectF(0, 0, max(1, w - 6), max(1, h - 4)))
        painter.restore()
    painter.end()
    out = qimage_to_pil(q)
    return out.convert('RGB') if image.mode == 'RGB' else out


class ImageCanvas(QGraphicsView):
    rectangle_selected = Signal(object)
    painted = Signal(object)       # start,end,mode,brush_size
    color_picked = Signal(object)
    text_box_changed = Signal(object)   # new x,y,w,h
    text_edited = Signal(str)
    text_layer_selected = Signal(int)
    paste_moved = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QBrush(QColor('#161b24')))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setMouseTracking(True)
        self.pixmap_item = None
        self.selected_item = None
        self.text_items = []
        self.text_edit_item = None
        self.text_frame = None
        self.paste_item = None
        self.current_image = None
        self.tool = 'new'  # keep compatibility with existing Windows smoke test
        self.selection = None
        self.fit = True
        self.stroke_start = None
        self.drag_start = None
        self.drag_mode = None
        self.drag_box = None
        self.temp_item = None
        self.brush_size = 18
        self._updating_text = False
        self._source_pixmap = None
        self._view_generation = 0
        self._rendered_size = None
        self._quality_jobs = set()
        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.timeout.connect(self._quality_render)
        self._text_layers: list[TextLayer] = []
        self._selected_text = -1

    def set_picture(self, image: Image.Image):
        self.current_image = image
        self._view_generation += 1
        self._source_pixmap = pixmap_from_pil(image)
        self._rendered_size = image.size
        if self.pixmap_item is None:
            self.pixmap_item = self.scene().addPixmap(self._source_pixmap)
            self.pixmap_item.setZValue(0)
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.scene().setSceneRect(QRectF(0, 0, image.width, image.height))
        else:
            self.pixmap_item.setPixmap(self._source_pixmap)
        if self.fit:
            self.fit_picture()
        self._draw_selection()
        self._queue_quality()

    def fit_picture(self):
        if not self.scene().sceneRect().isEmpty():
            self.fitInView(self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.fit = True
            self._queue_quality()

    def actual_size(self):
        self.resetTransform()
        self.fit = False
        self._queue_quality()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit and self.pixmap_item is not None:
            self.fit_picture()

    def wheelEvent(self, event):
        if self.pixmap_item is None:
            return
        delta = event.angleDelta().y()
        if delta:
            scale = 1.18 ** (delta / 120)
            if .05 < self.transform().m11() * scale < 18:
                self.scale(scale, scale)
                self.fit = False
                self._queue_quality()
        event.accept()

    def _queue_quality(self):
        if self.current_image is not None and self.pixmap_item is not None:
            self._view_generation += 1
            self._quality_timer.start(130)

    def _quality_render(self):
        image = self.current_image
        if image is None or self.pixmap_item is None:
            return
        factor = min(1.0, self.transform().m11() * self.viewport().devicePixelRatioF())
        target = image.size if factor >= .98 else (
            max(1, round(image.width * factor)), max(1, round(image.height * factor)))
        if target == self._rendered_size:
            return
        if target == image.size:
            self.pixmap_item.setPixmap(self._source_pixmap)
            self._rendered_size = target
            return
        generation = self._view_generation
        # Rendering on a worker avoids stalling brush movement and wheel zoom.
        work = Task(lambda _: image.resize(target, Image.Resampling.LANCZOS))
        self._quality_jobs.add(work)
        def ready(result):
            self._quality_jobs.discard(work)
            if self._view_generation != generation or image is not self.current_image:
                return
            pix = pixmap_from_pil(result)
            pix.setDevicePixelRatio(result.width / image.width)
            self.pixmap_item.setPixmap(pix)
            self._rendered_size = target
        work.signals.result.connect(ready)
        work.signals.error.connect(lambda _err: self._quality_jobs.discard(work))
        QThreadPool.globalInstance().start(work)

    def set_tool(self, tool: str):
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag
                         if tool == 'pan' else QGraphicsView.DragMode.NoDrag)
        if self.text_edit_item is not None:
            self.text_edit_item.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextEditorInteraction if tool == 'text'
                else Qt.TextInteractionFlag.NoTextInteraction)

    def _point(self, event):
        p = self.mapToScene(event.position().toPoint())
        bounds = self.scene().sceneRect()
        return QPointF(min(max(0, p.x()), bounds.width()), min(max(0, p.y()), bounds.height()))

    def set_selection(self, box):
        self.selection = box
        self._draw_selection()

    def _draw_selection(self):
        if self.selected_item is not None:
            self.scene().removeItem(self.selected_item)
            self.selected_item = None
        if self.selection is not None:
            x, y, w, h = self.selection
            self.selected_item = self.scene().addRect(QRectF(x, y, w, h),
                QPen(QColor('#6fcef5'), 1.5, Qt.PenStyle.DashLine))
            self.selected_item.setZValue(45)
            self.selected_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def set_text_layers(self, layers: list[TextLayer], selected: int):
        for frame in self.text_items:
            self.scene().removeItem(frame)
        self.text_items.clear()
        self.text_edit_item = None
        self.text_frame = None
        self._text_layers = layers
        self._selected_text = selected
        for i, layer in enumerate(layers):
            x, y, w, h = layer.box
            active = i == selected
            frame = self.scene().addRect(QRectF(0, 0, w, h),
                QPen(QColor('#55b9f7') if active else QColor(100, 175, 215, 120),
                     1.5 if active else .8, Qt.PenStyle.DashLine))
            frame.setPos(x, y)
            frame.setZValue(10)
            frame.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
            frame.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            content = QGraphicsTextItem(frame)
            content.setPos(3, 2)
            content.setFont(font_for(layer))
            content.setDefaultTextColor(QColor(layer.color))
            content.document().setDocumentMargin(0)
            content.setTextWidth(max(1, w-6))
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
            content.document().setDefaultTextOption(option)
            content.setPlainText(layer.text)
            content.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextEditorInteraction if active and self.tool=='text'
                else Qt.TextInteractionFlag.NoTextInteraction)
            content.setZValue(1)
            if active:
                self.text_frame = frame
                self.text_edit_item = content
                content.document().contentsChanged.connect(self._text_changed)
            self.text_items.append(frame)

    def _text_changed(self):
        if not self._updating_text and self.text_edit_item is not None:
            self.text_edited.emit(self.text_edit_item.toPlainText())

    def update_live_text(self, layer: TextLayer):
        if self.text_edit_item is None:
            return
        item = self.text_edit_item
        self._updating_text = True
        try:
            if item.toPlainText() != layer.text:
                item.setPlainText(layer.text)
            item.setFont(font_for(layer))
            item.setDefaultTextColor(QColor(layer.color))
            item.setTextWidth(max(1, layer.box[2] - 6))
        finally:
            self._updating_text = False
        self._source_pixmap = None
        self._view_generation = 0
        self._rendered_size = None
        self._quality_jobs = set()
        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.timeout.connect(self._quality_render)

    def add_paste(self, image: Image.Image, point: tuple[int, int]):
        self.cancel_paste()
        item = self.scene().addPixmap(pixmap_from_pil(image))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setPos(*point)
        item.setOpacity(.92)
        item.setZValue(60)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setSelected(True)
        self.paste_item = item
        self.set_tool('paste')

    def cancel_paste(self):
        if self.paste_item:
            self.scene().removeItem(self.paste_item)
            self.paste_item = None

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event); return
        if self.tool == 'pan' or self.tool == 'paste':
            super().mousePressEvent(event); return
        point = self._point(event)
        if self.tool in ('new','text'):
            if self.tool == 'text':
                for i in reversed(range(len(self._text_layers))):
                    if i == self._selected_text:
                        continue
                    x0, y0, w0, h0 = self._text_layers[i].box
                    if x0 <= point.x() <= x0+w0 and y0 <= point.y() <= y0+h0:
                        self.text_layer_selected.emit(i)
                        event.accept()
                        return
            if self.tool == 'text' and self.text_frame is not None:
                x, y, w, h = self._text_layers[self._selected_text].box
                margin = 11 / max(.3, self.transform().m11())
                if abs(point.x()-(x+w)) < margin and abs(point.y()-(y+h)) < margin:
                    self.drag_mode = 'resize_text'
                    self.drag_start = point
                    self.drag_box = (x,y,w,h)
                    event.accept(); return
                if x <= point.x() <= x+w and y <= point.y() <= y+h:
                    # Inside text: let QGraphicsTextItem handle caret/selection.
                    super().mousePressEvent(event); return
            self.drag_mode = 'select'
            self.drag_start = point
            if self.temp_item is not None:
                self.scene().removeItem(self.temp_item)
            self.temp_item = self.scene().addRect(QRectF(point,point),
               QPen(QColor('#51c8ff'), 2, Qt.PenStyle.DashLine))
            self.temp_item.setZValue(90)
            event.accept(); return
        if self.tool in ('brush','restore','smudge','picker'):
            if self.tool == 'picker':
                self.color_picked.emit((round(point.x()),round(point.y())))
            else:
                self.stroke_start = point
                self.painted.emit((point,point,self.tool,self.brush_size,True))
            event.accept(); return
        super().mousePressEvent(event)

    def mouseMoveEvent(self,event):
        point = self._point(event)
        if self.drag_mode == 'select' and self.drag_start is not None:
            self.temp_item.setRect(QRectF(self.drag_start, point).normalized())
            event.accept(); return
        if self.drag_mode == 'resize_text':
            x,y,w,h = self.drag_box
            self.text_box_changed.emit((x,y,max(15,round(point.x()-x)),
                                         max(15,round(point.y()-y))))
            event.accept(); return
        if self.stroke_start is not None:
            previous = self.stroke_start
            self.stroke_start = point
            self.painted.emit((previous,point,self.tool,self.brush_size,False))
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drag_mode == 'select':
                point = self._point(event)
                rect = QRectF(self.drag_start,point).normalized()
                self.drag_mode = None
                self.drag_start = None
                if self.temp_item:
                    self.scene().removeItem(self.temp_item)
                    self.temp_item = None
                if rect.width() >= 5 and rect.height() >= 5:
                    self.rectangle_selected.emit((round(rect.x()),round(rect.y()),
                         round(rect.width()),round(rect.height())))
                event.accept(); return
            if self.drag_mode == 'resize_text':
                self.drag_mode = None
                self.drag_box = None
                event.accept(); return
            if self.stroke_start is not None:
                self.stroke_start = None
                self.painted.emit((None,None,self.tool,self.brush_size,False))
                event.accept(); return
            if self.tool == 'paste':
                self.paste_moved.emit()
        super().mouseReleaseEvent(event)


class EditorWindow(QMainWindow):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)
        self.original = open_image(str(self.path))
        self.working = self.original.copy()
        self.layers: list[TextLayer] = []
        self.selected_text = -1
        self.selection = None
        self.history: list[tuple[Image.Image, list[TextLayer], object]] = []
        self.future = []
        self.settings = QSettings('MangaOffline','MangaOfflineTranslator')
        self.brush_color = QColor(str(self.settings.value('brushColor','#232323')))
        self.text_color = QColor(str(self.settings.value('textColor','#232323')))
        self.last_saved = None
        self.dirty = False
        self._in_sync = False
        self._stroke_in_progress = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._refresh_pixmap)
        self.setWindowTitle('Редактор — '+self.path.name)
        self.resize(1410,900)
        self.setMinimumSize(900,570)
        self._ui()
        self.canvas.set_picture(self.working)
        self.canvas.set_tool('new')
        self._sync()
        self.statusBar().showMessage('Выделите прямоугольник: фрагмент немедленно копируется как изображение.',9500)

    def _ui(self):
        main=QWidget();self.setCentralWidget(main)
        layout=QHBoxLayout(main);layout.setContentsMargins(0,0,0,0);layout.setSpacing(0)
        self.canvas=ImageCanvas(self)
        layout.addWidget(self.canvas,1)
        side=QFrame();side.setMinimumWidth(270);side.setMaximumWidth(330)
        sl=QVBoxLayout(side);sl.setContentsMargins(12,10,12,12);sl.setSpacing(8)
        self.hint=QLabel('1. Выделите надпись прямоугольником\n2. Вставьте фрагмент в переводчик\n3. Очистите буквы и вставьте перевод')
        self.hint.setWordWrap(True);sl.addWidget(self.hint)
        sl.addWidget(QLabel('Текст перевода (редактируется и на картинке):'))
        self.text_input=QPlainTextEdit()
        self.text_input.setPlaceholderText('Вставьте сюда перевод. Пробелы и переносы строк сохраняются.')
        self.text_input.setMaximumHeight(210)
        self.text_input.textChanged.connect(self._text_in_panel)
        sl.addWidget(self.text_input)
        self.insert_button=QPushButton('Вставить текст в зону')
        self.insert_button.clicked.connect(self.insert_text)
        sl.addWidget(self.insert_button)
        sl.addWidget(QLabel('Шрифт · настройки сохраняются'))
        self.family=QFontComboBox(); self.family.setCurrentFont(QFont(str(self.settings.value('fontFamily','Arial'))))
        self.family.currentFontChanged.connect(self._format_changed)
        sl.addWidget(self.family)
        fontrow=QHBoxLayout()
        self.font_size=QSpinBox();self.font_size.setRange(7,180)
        self.font_size.setValue(int(self.settings.value('fontSize',28)))
        self.font_size.valueChanged.connect(self._format_changed)
        fontrow.addWidget(QLabel('Размер'));fontrow.addWidget(self.font_size)
        self.bold_button=QToolButton();self.bold_button.setText('Ж');self.bold_button.setCheckable(True)
        self.bold_button.setChecked(str(self.settings.value('fontBold','false')).lower()=='true')
        self.bold_button.toggled.connect(self._format_changed)
        fontrow.addWidget(self.bold_button)
        self.italic_button=QToolButton();self.italic_button.setText('К');self.italic_button.setCheckable(True)
        self.italic_button.setChecked(str(self.settings.value('fontItalic','false')).lower()=='true')
        self.italic_button.toggled.connect(self._format_changed)
        fontrow.addWidget(self.italic_button)
        sl.addLayout(fontrow)
        self.color_button=QPushButton('Цвет текста…')
        self.color_button.clicked.connect(self.choose_text_color)
        sl.addWidget(self.color_button)
        sl.addWidget(QLabel('Рисование · цвет и диаметр кисти'))
        brushrow=QHBoxLayout()
        self.brush_color_button=QPushButton('Цвет кисти…')
        self.brush_color_button.clicked.connect(self.choose_brush_color)
        brushrow.addWidget(self.brush_color_button)
        self.brush_diameter=QSpinBox();self.brush_diameter.setRange(1,160)
        self.brush_diameter.setValue(int(self.settings.value('brushSize',18)))
        self.brush_diameter.valueChanged.connect(self._brush_size_changed)
        brushrow.addWidget(self.brush_diameter)
        sl.addLayout(brushrow)
        sl.addStretch(1)
        self.save_button=QPushButton('Сохранить результат в PNG…')
        self.save_button.clicked.connect(self.save_png)
        sl.addWidget(self.save_button)
        layout.addWidget(side)

        bar=QToolBar('Правка');bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea,bar)
        self.actions={}
        for tool,label,shortcut in (
            ('new','▭ Выделить','Alt+S'),('clear','Очистить буквы','Alt+C'),
            ('text','Текст','Alt+T'),('brush','Кисть','Alt+B'),('restore','Восстановить','Alt+E'),
            ('smudge','Мастихин','Alt+M'),('picker','Пипетка','Alt+I'),('pan','Рука',None),
            ('paste','Вставить фрагмент',None)):
            action=QAction(label,self);self.actions[tool]=action
            if shortcut:action.setShortcut(QKeySequence(shortcut))
            if tool=='clear':action.triggered.connect(self.clear_selection)
            elif tool=='paste':action.triggered.connect(self.paste_fragment)
            else:action.triggered.connect(lambda _,name=tool:self.set_tool(name))
            bar.addAction(action)
        bar.addSeparator()
        for label,callback,shortcut in (
            ('↶',self.undo,'Ctrl+Z'),('↷',self.redo,'Ctrl+Y'),
            ('−',lambda:self.canvas.scale(.8,.8),None),
            ('+',lambda:self.canvas.scale(1.25,1.25),None),
            ('Вписать',self.canvas.fit_picture,'Alt+F'),('100%',self.canvas.actual_size,'Alt+1'),
            ('PNG',self.save_png,'Ctrl+S')):
            action=QAction(label,self)
            if shortcut:action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            bar.addAction(action)
        for shortcut,callback in (
            ('Ctrl+C',self.copy_fragment),('Ctrl+V',self.paste_fragment),
            ('Ctrl+X',self.cut_fragment),('Delete',self.delete_active),
            ('Ctrl+Shift+Z',self.redo),('Escape',self.escape_tool),
            ('Ctrl+Return',self.commit_paste)):
            a=QAction(self);a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(callback);self.addAction(a)
        self.canvas.rectangle_selected.connect(self._selected)
        self.canvas.painted.connect(self._painted)
        self.canvas.color_picked.connect(self._pick)
        self.canvas.text_box_changed.connect(self._resize_text)
        self.canvas.text_edited.connect(self._text_in_canvas)
        self.canvas.text_layer_selected.connect(self._choose_text_layer)

    def _mark(self):
        self.dirty=True
        self.setWindowTitle('* Редактор — '+self.path.name)

    def _checkpoint(self):
        self.history.append((self.working.copy(),deepcopy(self.layers),self.selection))
        if len(self.history)>12:self.history.pop(0)
        self.future.clear();self._mark()

    def undo(self):
        if self.text_input.hasFocus():
            self.text_input.undo();return
        if self.canvas.text_edit_item is not None and self.canvas.text_edit_item.hasFocus():
            self.canvas.text_edit_item.document().undo();return
        if not self.history:return
        self.future.append((self.working.copy(),deepcopy(self.layers),self.selection))
        self.working,self.layers,self.selection=self.history.pop()
        self.selected_text=-1
        self.canvas.cancel_paste();self._sync()

    def redo(self):
        if self.text_input.hasFocus():
            self.text_input.redo();return
        if self.canvas.text_edit_item is not None and self.canvas.text_edit_item.hasFocus():
            self.canvas.text_edit_item.document().redo();return
        if not self.future:return
        self.history.append((self.working.copy(),deepcopy(self.layers),self.selection))
        self.working,self.layers,self.selection=self.future.pop()
        self.selected_text=-1
        self.canvas.cancel_paste();self._sync()

    def _refresh_pixmap(self):
        self.canvas.set_picture(self.working)

    def _sync(self):
        self.canvas.set_picture(self.working)
        self.canvas.set_selection(self.selection)
        self.canvas.set_text_layers(self.layers,self.selected_text)
        self._in_sync=True
        if 0 <= self.selected_text < len(self.layers):
            self.text_input.setPlainText(self.layers[self.selected_text].text)
        self._in_sync=False

    def _selected(self,box):
        self.commit_paste()
        self.selection=clip_box(box,self.working.size)
        self.canvas.set_selection(self.selection)
        self.copy_fragment()
        self.statusBar().showMessage('Фрагмент скопирован в буфер как картинка. Очистите буквы или вставьте перевод.',7500)

    def _focused_text(self):
        return self.text_input.hasFocus() or (self.canvas.text_edit_item is not None
           and self.canvas.text_edit_item.hasFocus())

    def copy_fragment(self):
        if self.text_input.hasFocus():
            self.text_input.copy();return
        if self.canvas.text_edit_item is not None and self.canvas.text_edit_item.hasFocus():
            text=self.canvas.text_edit_item.textCursor().selectedText()
            if text:QApplication.clipboard().setText(text)
            return
        if self.selection is None:return
        x,y,w,h=clip_box(self.selection,self.working.size)
        if w*h:
            QApplication.clipboard().setImage(qimage_from_pil(self.working.crop((x,y,x+w,y+h))))

    def cut_fragment(self):
        if self._focused_text():
            if self.text_input.hasFocus():self.text_input.cut()
            return
        if self.selection is None:return
        self.copy_fragment()
        x,y,w,h=clip_box(self.selection,self.working.size)
        self._checkpoint()
        # Cutting copies pixels then restores the original region rather than
        # creating a destructive solid-color hole.
        self.working.paste(self.original.crop((x,y,x+w,y+h)),(x,y))
        self._sync()

    def paste_fragment(self):
        if self.text_input.hasFocus():
            self.text_input.paste();return
        if self.canvas.text_edit_item is not None and self.canvas.text_edit_item.hasFocus():
            cursor = self.canvas.text_edit_item.textCursor()
            cursor.insertText(QApplication.clipboard().text())
            self.canvas.text_edit_item.setTextCursor(cursor)
            return
        image=QApplication.clipboard().image()
        if image.isNull():
            self.statusBar().showMessage('В буфере нет изображения.',4500);return
        pil=qimage_to_pil(image)
        if pil.width>max(5000,self.working.width*3) or pil.height>max(5000,self.working.height*3):
            QMessageBox.warning(self,'Вставка','Слишком большой фрагмент.');return
        if self.selection:
            x,y,_,_=self.selection
        else:
            x,y=round(self.working.width*.2),round(self.working.height*.2)
        self.canvas.add_paste(pil,(x,y))
        self.statusBar().showMessage('Перетащите фрагмент мышью. Enter — применить, Esc — отменить.',6500)

    def commit_paste(self):
        item=self.canvas.paste_item
        if item is None:return
        self._checkpoint()
        pil=qimage_to_pil(item.pixmap().toImage())
        x,y=round(item.pos().x()),round(item.pos().y())
        base=self.working.convert('RGBA')
        base.alpha_composite(pil,(x,y)) if x>=0 and y>=0 else base.paste(pil,(x,y),pil)
        self.working=base.convert('RGB') if self.working.mode=='RGB' else base
        self.canvas.cancel_paste()
        self.canvas.set_tool('new')
        self._sync()

    def escape_tool(self):
        if self.canvas.paste_item:
            self.canvas.cancel_paste()
        else:
            self.selection=None;self.canvas.set_selection(None)
        self.set_tool('new')

    def clear_selection(self):
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите надпись прямоугольником.',6000);return
        self.commit_paste()
        box=self.selection
        old=self.working.copy()
        new,count,_=clear_letters(old,box)
        if not count:
            self.statusBar().showMessage('Штрихи не найдены; используйте кисть восстановления/цвета.',8000);return
        self._checkpoint()
        self.working=new
        self._sync()
        self.statusBar().showMessage(f'Очистка: обработано {count} пикселей штрихов. Остальное не затронуто. Ctrl+Z — отменить.',9000)

    def set_tool(self,tool):
        if self.canvas.paste_item and tool!='paste':self.commit_paste()
        if tool=='text' and self.selected_text<0 and self.selection is not None:
            self.insert_text()
            return
        self.canvas.set_tool(tool)
        names={'new':'Прямоугольник: выделите область (автокопирование)',
               'brush':'Кисть: рисование поверх изображения','restore':'Восстановление пикселей из оригинала',
               'smudge':'Мастихин: протаскивание цвета','picker':'Пипетка: взять цвет','text':'Текст: правка прямо в рамке',
               'pan':'Перемещение изображения','paste':'Перемещение вставленного фрагмента'}
        self.statusBar().showMessage(names.get(tool,tool),7000)

    def _current_settings(self):
        return {'family':self.family.currentFont().family(),'size':self.font_size.value(),
                'bold':self.bold_button.isChecked(),'italic':self.italic_button.isChecked(),
                'color':self.text_color.name()}

    def insert_text(self):
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите прямоугольник для текста.',6000);return
        self.commit_paste()
        box=clip_box(self.selection,self.working.size)
        self._checkpoint()
        layer=TextLayer(box=box,text=self.text_input.toPlainText(),**self._current_settings())
        self.layers.append(layer)
        self.selected_text=len(self.layers)-1
        self.canvas.set_tool('text')
        self._sync()
        self.statusBar().showMessage('Текст редактируется прямо в голубой рамке. Потяните нижний правый угол для изменения размера.',8000)

    def _text_in_panel(self):
        if self._in_sync:return
        if 0 <= self.selected_text < len(self.layers):
            layer=self.layers[self.selected_text]
            layer.text=self.text_input.toPlainText()
            self.canvas.update_live_text(layer)
            self._mark()

    def _choose_text_layer(self,index):
        if not 0 <= index < len(self.layers):
            return
        self.selected_text=index
        self.selection=self.layers[index].box
        self.canvas.set_selection(self.selection)
        self.canvas.set_tool('text')
        self._sync()

    def _text_in_canvas(self,value):
        if self.selected_text<0 or self.selected_text>=len(self.layers):return
        self.layers[self.selected_text].text=value
        self._in_sync=True
        self.text_input.setPlainText(value)
        self._in_sync=False
        self._mark()

    def _resize_text(self,new_box):
        if self.selected_text<0 or self.selected_text>=len(self.layers):return
        self.layers[self.selected_text].box=clip_box(new_box,self.working.size)
        self.canvas.set_text_layers(self.layers,self.selected_text)
        self._mark()

    def _format_changed(self,*_):
        settings=self._current_settings()
        for key,val in [('fontFamily',settings['family']),('fontSize',settings['size']),
                        ('fontBold',settings['bold']),('fontItalic',settings['italic']),
                        ('textColor',settings['color'])]:self.settings.setValue(key,val)
        if 0<=self.selected_text<len(self.layers):
            layer=self.layers[self.selected_text]
            for key,val in settings.items():setattr(layer,key,val)
            self.canvas.update_live_text(layer)
            self._mark()

    def choose_text_color(self):
        c=QColorDialog.getColor(self.text_color,self,'Цвет вставляемого текста')
        if c.isValid():self.text_color=c;self._format_changed()

    def choose_brush_color(self):
        c=QColorDialog.getColor(self.brush_color,self,'Цвет кисти')
        if c.isValid():self.brush_color=c;self.settings.setValue('brushColor',c.name())

    def _brush_size_changed(self,value):
        self.canvas.brush_size=value
        self.settings.setValue('brushSize',value)

    def _painted(self,event):
        start,end,tool,size,first=event
        if start is None:
            self._stroke_in_progress=False
            self._refresh_pixmap()
            return
        if first or not self._stroke_in_progress:
            self._checkpoint();self._stroke_in_progress=True
        a=(start.x(),start.y());b=(end.x(),end.y())
        if tool=='smudge':
            self.working=palette_smudge(self.working,a,b,size)
        else:
            self.working=brush_segment(self.working,self.original,a,b,size,
                (*self.brush_color.getRgb()[:3],255),restore=(tool=='restore'))
        if not self._render_timer.isActive():self._render_timer.start(35)

    def _pick(self,coords):
        x,y=coords
        x=max(0,min(self.working.width-1,x));y=max(0,min(self.working.height-1,y))
        self.brush_color=QColor(*self.working.convert('RGB').getpixel((x,y)))
        self.settings.setValue('brushColor',self.brush_color.name())
        self.statusBar().showMessage('Цвет кисти: '+self.brush_color.name(),4500)

    def delete_active(self):
        if self._focused_text():
            if self.text_input.hasFocus():
                cursor=self.text_input.textCursor();cursor.deleteChar()
            elif self.canvas.text_edit_item is not None:
                cursor=self.canvas.text_edit_item.textCursor();cursor.deleteChar()
                self.canvas.text_edit_item.setTextCursor(cursor)
            return
        if self.canvas.paste_item:
            self.canvas.cancel_paste();return
        if 0<=self.selected_text<len(self.layers):
            self._checkpoint();self.layers.pop(self.selected_text)
            self.selected_text=-1;self._sync()
        else:
            self.selection=None;self.canvas.set_selection(None)

    def save_png(self):
        self.commit_paste()
        default=self.last_saved or self.path.with_name(self.path.stem+'_edited.png')
        target,_=QFileDialog.getSaveFileName(self,'Сохранить изображение',str(default),'PNG без потерь (*.png)')
        if not target:return
        file=Path(target)
        if file.suffix.lower()!='.png':file=file.with_suffix('.png')
        if file.resolve()==self.path.resolve():
            QMessageBox.warning(self,'Исходный файл','Оригинал нельзя перезаписывать.');return
        try:
            rendered=composite_text(self.working,self.layers)
            rendered.save(file,format='PNG')
            self.last_saved=file;self.dirty=False
            self.setWindowTitle('Редактор — '+self.path.name)
            self.statusBar().showMessage('Сохранено без потерь: '+str(file),7000)
        except (OSError,ValueError) as exc:
            QMessageBox.warning(self,'Ошибка сохранения',str(exc))

    def closeEvent(self,event):
        if self.dirty:
            result=QMessageBox.question(self,'Несохранённые изменения',
                'Закрыть редактор без сохранения результата?',
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if result!=QMessageBox.StandardButton.Yes:
                event.ignore();return
        super().closeEvent(event)