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
    QFontComboBox, QColorDialog, QFileDialog, QMessageBox, QFrame,
    QMenu, QStackedWidget, QSlider, QButtonGroup)

from engine import open_image
from paint_core import clip_box, brush_segment, palette_smudge
from cleanup_core import detect_glyph_boxes, remove_glyphs
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
    text_rectangle_requested = Signal(object)
    context_menu_requested = Signal(object)
    cleanup_box_toggled = Signal(int)
    cleanup_box_drawn = Signal(object)

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
        self.cleanup_items = []
        self.cleanup_boxes = []
        self.cleanup_enabled = []
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

    def set_cleanup_boxes(self, boxes: list[tuple[int,int,int,int]], enabled: list[bool]):
        for obj in self.cleanup_items:
            if obj.scene() == self.scene():
                self.scene().removeItem(obj)
        self.cleanup_items.clear()
        self.cleanup_boxes = list(boxes)
        self.cleanup_enabled = list(enabled)
        for (x,y,w,h), active in zip(self.cleanup_boxes, self.cleanup_enabled):
            outline=self.scene().addRect(QRectF(x,y,w,h),
                QPen(QColor('#39f0a9') if active else QColor('#999999'),
                     2.3 if active else 1.3,
                     Qt.PenStyle.SolidLine if active else Qt.PenStyle.DashLine))
            outline.setZValue(55)
            outline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.cleanup_items.append(outline)

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
        if self.tool == 'new' and self.cleanup_boxes:
            candidates=[]
            for idx,(x,y,w,h) in enumerate(self.cleanup_boxes):
                if x<=point.x()<=x+w and y<=point.y()<=y+h:
                    candidates.append((w*h,idx))
            if candidates:
                self.cleanup_box_toggled.emit(min(candidates)[1])
                event.accept()
                return
        if self.tool in ('new','text','text_new','cleanup_add'):
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

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._point(event)
            for i in range(len(self._text_layers)-1,-1,-1):
                x,y,w,h = self._text_layers[i].box
                if x <= point.x() <= x+w and y <= point.y() <= y+h:
                    self.drag_mode = None
                    if self.temp_item is not None:
                        self.scene().removeItem(self.temp_item)
                        self.temp_item = None
                    self.text_layer_selected.emit(i)
                    event.accept()
                    return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self,event):
        self.context_menu_requested.emit(event.globalPos())
        event.accept()

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
                    box=(round(rect.x()),round(rect.y()),round(rect.width()),round(rect.height()))
                    if self.tool == 'text_new':
                        self.text_rectangle_requested.emit(box)
                    elif self.tool == 'cleanup_add':
                        self.cleanup_box_drawn.emit(box)
                    else:
                        self.rectangle_selected.emit(box)
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
        self._cleanup_candidates=[]
        self._cleanup_enabled=[]
        self._cleanup_done_boxes=[]
        self._cleanup_done_selections=set()
        self._cleanup_preview_active=False
        self._cleanup_preview_image=None
        self._cleanup_preview_boxes=()
        self._cleanup_preview_base=None
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
        self.hint=QLabel('Выделение — копия в буфер. Очистка — только буквы. Текст — отдельной рамкой.')
        self.hint.setWordWrap(True)
        sl.addWidget(self.hint)
        self.sidebar_modes=QStackedWidget()
        sl.addWidget(self.sidebar_modes,1)
        self.selection_panel=QWidget()
        sel=QVBoxLayout(self.selection_panel)
        sel.setContentsMargins(0,4,0,0)
        sel.addWidget(QLabel('1. Выделите фрагмент изображения'))
        sel.addWidget(QLabel('2. Вставьте его в переводчик'))
        sel.addWidget(QLabel('3. Очистите буквы или исправьте кистью'))
        help_label=QLabel('Зелёные рамки — будут очищены, серые — исключены. Щелчком меняйте отметку.')
        help_label.setWordWrap(True)
        sel.addWidget(help_label)
        self.cleanup_stats=QLabel('Выделите надпись.')
        sel.addWidget(self.cleanup_stats)
        self.detect_button=QPushButton('Найти буквы')
        self.detect_button.clicked.connect(self.detect_cleanup)
        sel.addWidget(self.detect_button)
        self.manual_box_button=QPushButton('Добавить рамку вручную')
        self.manual_box_button.clicked.connect(lambda:self.set_tool('cleanup_add'))
        sel.addWidget(self.manual_box_button)
        self.preview_button=QPushButton('Предпросмотр результата')
        self.preview_button.clicked.connect(self.preview_cleanup)
        self.preview_button.setEnabled(False)
        sel.addWidget(self.preview_button)
        self.clear_button=QPushButton('Очистить отмеченные рамки')
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setEnabled(False)
        sel.addWidget(self.clear_button)
        self.selection_text_button=QPushButton('Создать отдельную рамку текста')
        self.selection_text_button.clicked.connect(self.insert_text)
        sel.addWidget(self.selection_text_button)
        sel.addStretch()
        self.sidebar_modes.addWidget(self.selection_panel)

        self.text_panel=QWidget()
        tl=QVBoxLayout(self.text_panel)
        tl.setContentsMargins(0,4,0,0)
        tl.addWidget(QLabel('Текст выбранного блока'))
        self.text_input=QPlainTextEdit()
        self.text_input.setPlaceholderText('Перевод. Пробелы и переносы строк сохраняются.')
        self.text_input.setMaximumHeight(220)
        self.text_input.textChanged.connect(self._text_in_panel)
        tl.addWidget(self.text_input)
        self.insert_button=QPushButton('Новая рамка текста · протяните мышью')
        self.insert_button.clicked.connect(self.insert_text)
        tl.addWidget(self.insert_button)
        tl.addWidget(QLabel('Шрифт выбранного блока'))
        self.family=QFontComboBox()
        self.family.setCurrentFont(QFont(str(self.settings.value('fontFamily','Arial'))))
        self.family.currentFontChanged.connect(self._format_changed)
        tl.addWidget(self.family)
        fontrow=QHBoxLayout()
        self.font_size=QSpinBox()
        self.font_size.setRange(7,180)
        self.font_size.setValue(int(self.settings.value('fontSize',28)))
        self.font_size.valueChanged.connect(self._format_changed)
        fontrow.addWidget(QLabel('Размер'))
        fontrow.addWidget(self.font_size)
        self.bold_button=QToolButton()
        self.bold_button.setText('Ж')
        self.bold_button.setCheckable(True)
        self.bold_button.setChecked(str(self.settings.value('fontBold','false')).lower()=='true')
        self.bold_button.toggled.connect(self._format_changed)
        fontrow.addWidget(self.bold_button)
        self.italic_button=QToolButton()
        self.italic_button.setText('К')
        self.italic_button.setCheckable(True)
        self.italic_button.setChecked(str(self.settings.value('fontItalic','false')).lower()=='true')
        self.italic_button.toggled.connect(self._format_changed)
        fontrow.addWidget(self.italic_button)
        tl.addLayout(fontrow)
        self.color_button=QPushButton('Цвет текста…')
        self.color_button.clicked.connect(self.choose_text_color)
        tl.addWidget(self.color_button)
        tl.addWidget(QLabel('Двойной щелчок по любому текстовому блоку возвращает его в режим правки.'))
        tl.addStretch()
        self.sidebar_modes.addWidget(self.text_panel)

        self.brush_panel=QWidget()
        bl=QVBoxLayout(self.brush_panel)
        bl.setContentsMargins(0,4,0,0)
        bl.addWidget(QLabel('Кисти — по образцу панели Paint 3D'))
        self.brush_tool_buttons={}
        tool_names=[
            ('brush','🖌 Кисть'),('restore','▤ Восстановление'),
            ('smudge','◉ Мастихин'),('picker','⊙ Пипетка')
        ]
        for mode,label in tool_names:
            button=QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False,t=mode:self.set_tool(t))
            self.brush_tool_buttons[mode]=button
            bl.addWidget(button)
        bl.addWidget(QLabel('Толщина кисти'))
        sizes=QHBoxLayout()
        self.brush_slider=QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1,160)
        self.brush_slider.setValue(int(self.settings.value('brushSize',18)))
        self.brush_diameter=QSpinBox()
        self.brush_diameter.setRange(1,160)
        self.brush_diameter.setValue(self.brush_slider.value())
        self.brush_slider.valueChanged.connect(self.brush_diameter.setValue)
        self.brush_diameter.valueChanged.connect(self.brush_slider.setValue)
        self.brush_diameter.valueChanged.connect(self._brush_size_changed)
        sizes.addWidget(self.brush_slider,1)
        sizes.addWidget(self.brush_diameter)
        bl.addLayout(sizes)
        bl.addWidget(QLabel('Быстрые размеры'))
        presets=QHBoxLayout()
        for size in (4,12,24,48):
            b=QToolButton()
            b.setText(str(size))
            b.clicked.connect(lambda checked=False,n=size:self.brush_diameter.setValue(n))
            presets.addWidget(b)
        bl.addLayout(presets)
        bl.addWidget(QLabel('Цвета'))
        swatches=QHBoxLayout()
        for css in ('#1b1b1b','#ffffff','#8e8e8e','#e84445','#2a87d2','#42a978'):
            b=QToolButton()
            b.setFixedSize(32,28)
            b.setToolTip(css)
            b.setStyleSheet('background:'+css+';border:1px solid #6b7b8b;border-radius:5px;')
            b.clicked.connect(lambda checked=False,color=css:self._set_brush_color(color))
            swatches.addWidget(b)
        bl.addLayout(swatches)
        self.brush_color_button=QPushButton('Другой цвет…')
        self.brush_color_button.clicked.connect(self.choose_brush_color)
        bl.addWidget(self.brush_color_button)
        bl.addWidget(QLabel('Изменения кистью видны сразу. Ctrl+Z отменяет последний штрих.'))
        bl.addStretch()
        self.sidebar_modes.addWidget(self.brush_panel)
        sl.addStretch(1)
        self.save_button=QPushButton('Сохранить результат в PNG…')
        self.save_button.clicked.connect(self.save_png)
        sl.addWidget(self.save_button)
        layout.addWidget(side)

        bar=QToolBar('Правка');bar.setMovable(False)
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea,bar)
        self.return_to_viewer=None
        self.back_action=QAction('← Просмотр',self)
        self.back_action.triggered.connect(self.back_to_viewer)
        bar.addAction(self.back_action)
        bar.addSeparator()
        self.actions={}
        for tool,label,shortcut in (
            ('new','▭ Выделить','Alt+S'),
            ('text','Текст','Alt+T'),('brush','Кисть','Alt+B'),('restore','Восстановить','Alt+E'),
            ('smudge','Мастихин','Alt+M'),('picker','Пипетка','Alt+I'),('pan','Рука',None),
            ('paste','Вставить фрагмент',None)):
            action=QAction(label,self);self.actions[tool]=action
            if shortcut:action.setShortcut(QKeySequence(shortcut))
            if tool!='paste':
                action.setCheckable(True)
            if tool=='paste':action.triggered.connect(lambda:self.paste_fragment(force_image=True))
            elif tool=='text':action.triggered.connect(self.insert_text)
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
        self.canvas.text_rectangle_requested.connect(self._create_text_box)
        self.canvas.context_menu_requested.connect(self._context_menu)
        self.canvas.cleanup_box_toggled.connect(self._toggle_cleanup_box)
        self.canvas.cleanup_box_drawn.connect(self._manual_cleanup_box)
        self.set_tool('new')

    def _mark(self):
        self.dirty=True
        self.setWindowTitle('* Редактор — '+self.path.name)

    def _checkpoint(self):
        self.history.append((self.working.copy(),deepcopy(self.layers),self.selection,
                             list(self._cleanup_done_boxes),set(self._cleanup_done_selections)))
        if len(self.history)>12:self.history.pop(0)
        self.future.clear();self._mark()

    def undo(self):
        if self.text_input.hasFocus():
            self.text_input.undo();return
        if self.canvas.text_edit_item is not None and self.canvas.text_edit_item.hasFocus():
            self.canvas.text_edit_item.document().undo();return
        if not self.history:return
        self.future.append((self.working.copy(),deepcopy(self.layers),self.selection,
                            list(self._cleanup_done_boxes),set(self._cleanup_done_selections)))
        (self.working,self.layers,self.selection,
         self._cleanup_done_boxes,self._cleanup_done_selections)=self.history.pop()
        self._cleanup_candidates=[];self._cleanup_enabled=[]
        self._cleanup_preview_active=False
        self.selected_text=-1
        self.canvas.cancel_paste();self._sync();self._refresh_cleanup_overlay()

    def redo(self):
        if self.text_input.hasFocus():
            self.text_input.redo();return
        if self.canvas.text_edit_item is not None and self.canvas.text_edit_item.hasFocus():
            self.canvas.text_edit_item.document().redo();return
        if not self.future:return
        self.history.append((self.working.copy(),deepcopy(self.layers),self.selection,
                             list(self._cleanup_done_boxes),set(self._cleanup_done_selections)))
        (self.working,self.layers,self.selection,
         self._cleanup_done_boxes,self._cleanup_done_selections)=self.future.pop()
        self._cleanup_candidates=[];self._cleanup_enabled=[]
        self.selected_text=-1
        self.canvas.cancel_paste();self._sync();self._refresh_cleanup_overlay()

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
        self._invalidate_cleanup_preview()
        self.commit_paste()
        self.selection=clip_box(box,self.working.size)
        self.canvas.set_selection(self.selection)
        self.copy_fragment()
        self.detect_cleanup()
        self.statusBar().showMessage('Фрагмент скопирован. Проверьте зелёные рамки букв перед очисткой.',8500)

    def _set_brush_color(self,color):
        self.brush_color=QColor(color)
        self.settings.setValue('brushColor',self.brush_color.name())

    def _context_menu(self,global_pos):
        self.canvas.setFocus()
        menu=QMenu(self)
        paste=menu.addAction('Вставить изображение из буфера  Ctrl+V')
        paste.setEnabled(QApplication.clipboard().mimeData().hasImage())
        paste.triggered.connect(lambda:self.paste_fragment(force_image=True))
        copy=menu.addAction('Копировать фрагмент  Ctrl+C')
        copy.setEnabled(self.selection is not None)
        copy.triggered.connect(self.copy_fragment)
        menu.addSeparator()
        menu.addAction('Выделить прямоугольник',lambda:self.set_tool('new'))
        menu.addAction('Новая рамка текста',self.insert_text)
        menu.addAction('Найти буквы',self.detect_cleanup)
        clear=menu.addAction('Очистить отмеченные рамки',self.clear_selection)
        clear.setEnabled(self.clear_button.isEnabled())
        menu.addSeparator()
        menu.addAction('Кисть',lambda:self.set_tool('brush'))
        menu.addAction('Мастихин',lambda:self.set_tool('smudge'))
        menu.addAction('Восстановить оригинал',lambda:self.set_tool('restore'))
        menu.exec(global_pos)

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

    def paste_fragment(self,force_image=False):
        if self.text_input.hasFocus() and not force_image:
            self.text_input.paste();return
        if (not force_image and self.canvas.text_edit_item is not None
                and self.canvas.text_edit_item.hasFocus()):
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

    def _refresh_cleanup_overlay(self):
        if not self._cleanup_preview_active:
            self.canvas.set_cleanup_boxes(self._cleanup_candidates,self._cleanup_enabled)
        enabled=sum(self._cleanup_enabled)
        self.cleanup_stats.setText(
            f'Найдено рамок: {len(self._cleanup_candidates)}. Отмечено: {enabled}.')
        self.clear_button.setEnabled(enabled>0)
        self.preview_button.setEnabled(enabled>0)
        self.preview_button.setText('Вернуться к оригиналу' if self._cleanup_preview_active
                                    else 'Предпросмотр результата')

    def preview_cleanup(self):
        if self._cleanup_preview_active:
            self._cleanup_preview_active=False
            self.canvas.set_picture(self.working)
            self._refresh_cleanup_overlay()
            return
        if not any(self._cleanup_enabled):
            return
        allowed=[box for box,active in zip(self._cleanup_candidates,self._cleanup_enabled)
                 if active and not any(self._boxes_overlap(box,done)
                                       for done in self._cleanup_done_boxes)]
        if not allowed:
            return
        boxes=tuple(allowed)
        # Keep exactly the image shown to the user. Apply must commit THIS
        # image, never start a second independently calculated cleanup.
        if (self._cleanup_preview_image is not None
                and self._cleanup_preview_base is self.working
                and self._cleanup_preview_boxes == boxes):
            preview=self._cleanup_preview_image
        else:
            preview,_=remove_glyphs(self.working,boxes)
            self._cleanup_preview_image=preview
            self._cleanup_preview_boxes=boxes
            self._cleanup_preview_base=self.working
        self._cleanup_preview_active=True
        self.canvas.set_cleanup_boxes([],[])
        self.canvas.set_picture(preview)
        self._refresh_cleanup_overlay()

    def _stop_cleanup_preview(self):
        if self._cleanup_preview_active:
            self._cleanup_preview_active=False
            self.canvas.set_picture(self.working)

    def _invalidate_cleanup_preview(self):
        self._stop_cleanup_preview()
        self._cleanup_preview_image=None
        self._cleanup_preview_boxes=()
        self._cleanup_preview_base=None

    def detect_cleanup(self):
        self._invalidate_cleanup_preview()
        if self.selection is None:
            self.cleanup_stats.setText('Сначала выделите надпись.')
            return
        if self.selection in self._cleanup_done_selections:
            self._cleanup_candidates=[]
            self._cleanup_enabled=[]
            self._refresh_cleanup_overlay()
            self.cleanup_stats.setText('Область уже очищена. Ctrl+Z — отменить.')
            return
        # Only show candidate rectangles; do not modify the image here.
        boxes=detect_glyph_boxes(self.working,self.selection)
        self._cleanup_candidates=boxes
        self._cleanup_enabled=[
            not any(self._boxes_overlap(box,done) for done in self._cleanup_done_boxes)
            for box in boxes]
        self._refresh_cleanup_overlay()
        if not boxes:
            self.statusBar().showMessage('Буквы не найдены. Добавьте рамки вручную.',6500)

    @staticmethod
    def _boxes_overlap(a,b):
        x,y,w,h=a
        xx,yy,ww,hh=b
        return x<xx+ww and x+w>xx and y<yy+hh and y+h>yy

    def _toggle_cleanup_box(self,index):
        self._invalidate_cleanup_preview()
        if not 0<=index<len(self._cleanup_enabled):return
        self._cleanup_enabled[index]=not self._cleanup_enabled[index]
        self._refresh_cleanup_overlay()

    def _manual_cleanup_box(self,box):
        self._invalidate_cleanup_preview()
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите надпись.',6000)
            self.set_tool('new')
            return
        x,y,w,h=box
        sx,sy,sw,sh=self.selection
        left=max(x,sx)
        top=max(y,sy)
        right=min(x+w,sx+sw)
        bottom=min(y+h,sy+sh)
        if right-left>=3 and bottom-top>=3:
            self._cleanup_candidates.append((left,top,right-left,bottom-top))
            self._cleanup_enabled.append(True)
            self._refresh_cleanup_overlay()
        self.set_tool('new')

    def clear_selection(self):
        # The PREVIEW is the transaction's exact proposed result. If a user
        # saw a cleaned image, clicking Apply must retain that same image.
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите прямоугольную область.',6000)
            return
        if self.selection in self._cleanup_done_selections:
            self.statusBar().showMessage('Уже очищено. Повторное нажатие ничего не изменяет.',6500)
            return
        self.commit_paste()
        approved=tuple(box for box,active in zip(self._cleanup_candidates,self._cleanup_enabled)
                       if active and not any(self._boxes_overlap(box,done)
                                            for done in self._cleanup_done_boxes))
        if not approved:
            self.statusBar().showMessage('Нет отмеченных рамок для очистки.',6500)
            return
        if (self._cleanup_preview_image is not None
                and self._cleanup_preview_base is self.working
                and self._cleanup_preview_boxes == approved):
            result=self._cleanup_preview_image.copy()
            count=sum(w*h for x,y,w,h in approved)
        else:
            result,count=remove_glyphs(self.working,approved)
        if not count or result.tobytes()==self.working.tobytes():
            self.statusBar().showMessage('Очистка не дала изменений. Попробуйте другую рамку.',6500)
            return
        self._checkpoint()
        self.working=result
        self._cleanup_done_boxes.extend(approved)
        self._cleanup_done_selections.add(self.selection)
        self._cleanup_preview_active=False
        self._cleanup_preview_image=None
        self._cleanup_preview_base=None
        self._cleanup_preview_boxes=()
        self._cleanup_candidates=[]
        self._cleanup_enabled=[]
        self._sync()
        self._refresh_cleanup_overlay()
        self.statusBar().showMessage(
            f'Применён результат предпросмотра ({len(approved)} рамок). Ctrl+Z — отменить.',9000)

    def set_tool(self,tool):
        if self.canvas.paste_item and tool!='paste':self.commit_paste()
        self.canvas.set_tool(tool)
        section=2 if tool in ('brush','restore','smudge','picker') else (1 if tool in ('text','text_new') else 0)
        self.sidebar_modes.setCurrentIndex(section)
        active='text' if tool in ('text','text_new') else tool
        for name,action in self.actions.items():
            if action.isCheckable():action.setChecked(name==active)
        for name,button in self.brush_tool_buttons.items():
            button.setChecked(name==active)
        names={'cleanup_add':'Нарисуйте рамку пропущенной буквы',
               'new':'Выделение сразу копирует картинку',
               'text_new':'Протяните отдельную рамку текста',
               'brush':'Кисть','restore':'Восстановление из оригинала',
               'smudge':'Мастихин','picker':'Пипетка',
               'text':'Двойной щелчок по другой текстовой рамке вернёт к её правке',
               'pan':'Перемещение изображения','paste':'Перемещение вставленного фрагмента'}
        self.statusBar().showMessage(names.get(tool,tool),7000)

    def _current_settings(self):
        return {'family':self.family.currentFont().family(),'size':self.font_size.value(),
                'bold':self.bold_button.isChecked(),'italic':self.italic_button.isChecked(),
                'color':self.text_color.name()}

    def insert_text(self):
        """Text placement does not reuse the image selection rectangle."""
        self.commit_paste()
        if 0 <= self.selected_text < len(self.layers):
            self.selected_text=-1
            self._in_sync=True
            self.text_input.clear()
            self._in_sync=False
        self.set_tool('text_new')
        self.statusBar().showMessage('Протяните мышью новую рамку текста независимо от зоны очистки.',8500)

    def _create_text_box(self,box):
        box=clip_box(box,self.working.size)
        if box[2]<8 or box[3]<8:return
        self._checkpoint()
        layer=TextLayer(box=box,text=self.text_input.toPlainText(),**self._current_settings())
        self.layers.append(layer)
        self.selected_text=len(self.layers)-1
        self.set_tool('text')
        self._sync()
        if self.canvas.text_edit_item is not None:
            self.canvas.text_edit_item.setFocus()
        self.statusBar().showMessage('Двойной щелчок по тексту вернёт к редактированию.',8500)

    def _text_in_panel(self):
        if self._in_sync:return
        if 0 <= self.selected_text < len(self.layers):
            layer=self.layers[self.selected_text]
            layer.text=self.text_input.toPlainText()
            self.canvas.update_live_text(layer)
            self._mark()

    def _load_layer_settings(self,layer):
        self._in_sync=True
        controls=(self.family,self.font_size,self.bold_button,self.italic_button)
        for c in controls:c.blockSignals(True)
        try:
            self.family.setCurrentFont(QFont(layer.family))
            self.font_size.setValue(layer.size)
            self.bold_button.setChecked(layer.bold)
            self.italic_button.setChecked(layer.italic)
            self.text_color=QColor(layer.color)
        finally:
            for c in controls:c.blockSignals(False)
            self._in_sync=False
        self.color_button.setStyleSheet('border: 2px solid '+layer.color)

    def _choose_text_layer(self,index):
        if not 0 <= index < len(self.layers):return
        self.selected_text=index
        self._load_layer_settings(self.layers[index])
        self.set_tool('text')
        self._sync()
        if self.canvas.text_edit_item is not None:
            self.canvas.text_edit_item.setFocus()

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
        if self._in_sync:return
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

    def back_to_viewer(self):
        if callable(self.return_to_viewer):self.return_to_viewer()
        else:self.close()

    def closeEvent(self,event):
        if self.dirty:
            result=QMessageBox.question(self,'Несохранённые изменения',
                'Закрыть редактор без сохранения результата?',
                QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if result!=QMessageBox.StandardButton.Yes:
                event.ignore();return
        super().closeEvent(event)