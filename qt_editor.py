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
import numpy as np
import cv2
from PySide6.QtCore import Qt, QRectF, QPointF, QSize, Signal, QSettings, QThreadPool, QTimer, QEvent
from PySide6.QtGui import (QAction, QColor, QFont, QPainter, QPen, QBrush,
                           QImage, QPixmap, QKeySequence, QTextDocument, QTextOption, QCursor)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QSplitter, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsItem,
    QPushButton, QToolButton, QToolBar, QPlainTextEdit, QSpinBox,
    QFontComboBox, QColorDialog, QFileDialog, QMessageBox, QFrame,
    QMenu, QStackedWidget, QSlider, QButtonGroup, QComboBox,
    QGridLayout, QDialog, QDialogButtonBox, QFormLayout, QScrollArea)

from engine import open_image
from paint_core import clip_box, brush_segment, palette_smudge
from contrast_cleanup import segment_ink, repair_background, mask_components, confirmed_mask
from qt_common import pixmap_from_pil, Task
from qt_brush_icons import brush_icon, picker_cursor_pixmap
from qt_toolbar_icons import toolbar_icon
from paint3d_tools import draw_stroke, TOOLS as PAINT_TOOLS

VISIBLE_PAINT_TOOLS = ('pencil', 'fill', 'spray', 'blur', 'restore')


@dataclass
class TextLayer:
    box: tuple[int, int, int, int]
    text: str = ''
    family: str = 'Arial'
    size: int = 28
    color: str = '#202020'
    bold: bool = False
    italic: bool = False
    underline: bool = False
    alignment: str = 'left'


def font_for(layer: TextLayer) -> QFont:
    f = QFont(layer.family)
    f.setPixelSize(layer.size)  # image pixels, not device-dependent points
    f.setBold(layer.bold)
    f.setItalic(layer.italic)
    f.setUnderline(layer.underline)
    return f


def text_option(layer: TextLayer) -> QTextOption:
    opt=QTextOption()
    opt.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    opt.setAlignment({'left':Qt.AlignmentFlag.AlignLeft,
                      'center':Qt.AlignmentFlag.AlignHCenter,
                      'right':Qt.AlignmentFlag.AlignRight}.get(
                          layer.alignment,Qt.AlignmentFlag.AlignLeft))
    return opt


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
        doc.setDefaultTextOption(text_option(layer))
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


class EditableTextItem(QGraphicsTextItem):
    """Enter inserts a newline; Ctrl+Enter commits the caption."""
    apply_requested = Signal()

    def keyPressEvent(self, event):
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.apply_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class ImageCanvas(QGraphicsView):
    rectangle_selected = Signal(object)
    painted = Signal(object)       # start,end,mode,brush_size
    color_picked = Signal(object)
    text_box_changed = Signal(object)   # new x,y,w,h
    text_geometry_started = Signal()
    text_geometry_finished = Signal()
    zoom_changed = Signal(float)
    text_edited = Signal(str)
    text_commit_requested = Signal()
    text_layer_selected = Signal(int)
    paste_moved = Signal()
    paste_confirmed = Signal()
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
        self.cleanup_mask_item = None
        self.cleanup_boxes = []
        self.cleanup_enabled = []
        self.text_items = []
        self.text_content_items = []  # retain Python-owned QGraphicsTextItem subclasses
        self.text_handles = []
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
        self.paste_resize_start = None
        self.paste_resize_scale = 1.0
        self.picker_preview_item = self.scene().addPixmap(QPixmap())
        self.picker_preview_item.setZValue(100)
        self.picker_preview_item.setFlag(
            QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.picker_preview_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.picker_preview_item.hide()
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
        self._picker_rgb = image if image.mode == 'RGB' else image.convert('RGB')
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
            self.zoom_changed.emit(self.transform().m11())

    def actual_size(self):
        self.resetTransform()
        self.fit = False
        self._queue_quality()
        self.zoom_changed.emit(1.0)

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
                self.zoom_changed.emit(self.transform().m11())
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
        if tool in VISIBLE_PAINT_TOOLS:
            hotspots = {'pencil': (17,31), 'fill': (6,25),
                        'spray': (9,9), 'blur': (16,16), 'restore': (9,24)}
            self.viewport().setCursor(QCursor(
                brush_icon(tool).pixmap(32,32), *hotspots[tool]))
        else:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        if tool != 'picker':
            self.picker_preview_item.hide()
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

    def set_cleanup_mask(self, mask: np.ndarray | None, box):
        # Overlay only *actual writable pixels*. Its own item has no mouse
        # interaction; rectangle toggles still go to ImageCanvas.mousePressEvent.
        if self.cleanup_mask_item is not None:
            self.scene().removeItem(self.cleanup_mask_item)
            self.cleanup_mask_item = None
        if mask is None or box is None:
            return
        x,y,w,h=clip_box(box,(mask.shape[1],mask.shape[0]))
        if w<=0 or h<=0:
            return
        roi=np.asarray(mask[y:y+h,x:x+w])>0
        if not np.any(roi):
            return
        rgba=np.zeros((h,w,4),dtype=np.uint8)
        rgba[roi]=[255,55,95,125]
        item=self.scene().addPixmap(pixmap_from_pil(Image.fromarray(rgba,'RGBA')))
        item.setPos(x,y)
        item.setZValue(50)
        item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.cleanup_mask_item=item

    def _position_text_handles(self, box):
        if not self.text_handles:
            return
        x,y,w,h=box
        points=((x,y),(x+w/2,y),(x+w,y),
                (x,y+h/2),(x+w,y+h/2),
                (x,y+h),(x+w/2,y+h),(x+w,y+h))
        for handle,(hx,hy) in zip(self.text_handles,points):
            handle.setPos(hx-3.5,hy-3.5)

    def set_text_geometry_live(self,box):
        if self.text_frame is None or self.text_edit_item is None:
            return
        x,y,w,h=box
        self.text_frame.setPos(x,y)
        self.text_frame.setRect(QRectF(0,0,w,h))
        self.text_edit_item.setTextWidth(max(1,w-6))
        self._position_text_handles(box)

    def set_text_layers(self, layers: list[TextLayer], selected: int):
        # Detach the outgoing editor before removing scene items: Qt may
        # otherwise report a late document change against the new layer.
        if self.text_edit_item is not None:
            try:
                self.text_edit_item.document().contentsChanged.disconnect(
                    self._text_changed)
            except (TypeError, RuntimeError):
                pass
        for item in self.text_handles:
            if item.scene()==self.scene():
                self.scene().removeItem(item)
        self.text_handles=[]
        for frame in self.text_items:
            self.scene().removeItem(frame)
        self.text_items.clear()
        self.text_content_items.clear()
        self.text_edit_item = None
        self.text_frame = None
        self._text_layers = layers
        self._selected_text = selected
        for i, layer in enumerate(layers):
            x, y, w, h = layer.box
            active = i == selected
            frame = self.scene().addRect(QRectF(0, 0, w, h),
                QPen(QColor('#55b9f7'),1.5,Qt.PenStyle.DashLine) if active
                else QPen(Qt.PenStyle.NoPen))
            frame.setPos(x, y)
            frame.setZValue(10)
            frame.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
            frame.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            content = EditableTextItem(frame)
            content.setPos(3, 2)
            content.setFont(font_for(layer))
            content.setDefaultTextColor(QColor(layer.color))
            content.document().setDocumentMargin(0)
            content.document().setUndoRedoEnabled(False)
            content.setTextWidth(max(1, w-6))
            content.document().setDefaultTextOption(text_option(layer))
            content.setPlainText(layer.text)
            content.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextEditorInteraction if active and self.tool=='text'
                else Qt.TextInteractionFlag.NoTextInteraction)
            content.setZValue(1)
            if active:
                self.text_frame = frame
                self.text_edit_item = content
                content.document().contentsChanged.connect(self._text_changed)
                content.apply_requested.connect(self.text_commit_requested)
            self.text_items.append(frame)
            self.text_content_items.append(content)
        if 0 <= selected < len(layers):
            for _ in range(8):
                handle=self.scene().addRect(QRectF(0,0,7,7),
                    QPen(QColor('#80cdff'),1.3),QBrush(QColor('#1b3449')))
                handle.setZValue(25)
                handle.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.text_handles.append(handle)
            self._position_text_handles(layers[selected].box)

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
            item.document().setDefaultTextOption(text_option(layer))
        finally:
            self._updating_text = False

    def add_paste(self, image: Image.Image, point: tuple[int, int]):
        self.cancel_paste()
        item = self.scene().addPixmap(pixmap_from_pil(image))
        item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        item.setPos(*point)
        item.setOpacity(1.0)
        item.setZValue(60)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setSelected(True)
        self.paste_item = item
        self.paste_source = image.copy()
        self.set_tool('paste')

    def cancel_paste(self):
        if self.paste_item:
            self.scene().removeItem(self.paste_item)
            self.paste_item = None
        self.paste_source = None

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event); return
        if self.tool == 'pan':
            super().mousePressEvent(event); return
        if self.tool == 'paste':
            if self.paste_item is not None:
                rect=self.paste_item.sceneBoundingRect()
                pt=self.mapToScene(event.position().toPoint())
                tolerance=12.0/max(.1,self.transform().m11())
                if abs(pt.x()-rect.right())<=tolerance and abs(pt.y()-rect.bottom())<=tolerance:
                    self.drag_mode='resize_paste'
                    self.paste_resize_start=pt
                    self.paste_resize_scale=self.paste_item.scale()
                    event.accept()
                    return
            super().mousePressEvent(event)
            return
        point = self._point(event)
        if self.tool == 'cleanup' and self.cleanup_boxes:
            candidates=[]
            for idx,(x,y,w,h) in enumerate(self.cleanup_boxes):
                if x<=point.x()<=x+w and y<=point.y()<=y+h:
                    candidates.append((w*h,idx))
            if candidates:
                self.cleanup_box_toggled.emit(min(candidates)[1])
                event.accept()
                return
        if self.tool in ('new','cleanup','text','text_new','cleanup_add'):
            if self.tool == 'text' and self.text_frame is not None:
                x,y,w,h=self._text_layers[self._selected_text].box
                margin=min(11/max(.2,self.transform().m11()),max(3,min(w,h)/5))
                px,py=point.x(),point.y()
                inside=(x-margin<=px<=x+w+margin and y-margin<=py<=y+h+margin)
                if inside:
                    left=abs(px-x)<=margin
                    right=abs(px-(x+w))<=margin
                    top=abs(py-y)<=margin
                    bottom=abs(py-(y+h))<=margin
                    # Top border is a Paint3D-like grab bar; Alt+drag
                    # anywhere inside is an additional move gesture.
                    move=((top and not left and not right)
                          or (event.modifiers() & Qt.KeyboardModifier.AltModifier
                              and x<=px<=x+w and y<=py<=y+h))
                    if move or left or right or bottom or top:
                        self.drag_mode='move_text' if move else 'resize_text'
                        self.drag_edges=('l' if left else '')+('r' if right else '')+(
                            't' if top else '')+('b' if bottom else '')
                        self.drag_start=point
                        self.drag_box=(x,y,w,h)
                        self.text_geometry_started.emit()
                        event.accept()
                        return
                if x<=px<=x+w and y<=py<=y+h:
                    super().mousePressEvent(event)
                    return
            if self.tool=='text':
                # Active resize handles win over underlying overlapping layers.
                for i in reversed(range(len(self._text_layers))):
                    if i==self._selected_text:
                        continue
                    x0,y0,w0,h0=self._text_layers[i].box
                    if x0<=point.x()<=x0+w0 and y0<=point.y()<=y0+h0:
                        self.text_layer_selected.emit(i)
                        event.accept()
                        return
            self.drag_mode = 'select'
            self.drag_start = point
            if self.temp_item is not None:
                self.scene().removeItem(self.temp_item)
            self.temp_item = self.scene().addRect(QRectF(point,point),
               QPen(QColor('#51c8ff'), 2, Qt.PenStyle.DashLine))
            self.temp_item.setZValue(90)
            event.accept(); return
        if self.tool in (*PAINT_TOOLS,'picker'):
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

    def leaveEvent(self,event):
        if self.tool=='picker':
            self.picker_preview_item.hide()
        super().leaveEvent(event)

    def keyPressEvent(self,event):
        if self.paste_item is not None and event.key() in (
                Qt.Key.Key_Return,Qt.Key.Key_Enter):
            self.paste_confirmed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self,event):
        self.context_menu_requested.emit(event.globalPos())
        event.accept()

    def mouseMoveEvent(self,event):
        point = self._point(event)
        if self.tool=='picker' and self.current_image is not None:
            x=max(0,min(self.current_image.width-1,int(point.x())))
            y=max(0,min(self.current_image.height-1,int(point.y())))
            c=QColor(*self._picker_rgb.getpixel((x,y)))
            self.picker_preview_item.setPixmap(picker_cursor_pixmap(c))
            self.picker_preview_item.setOffset(-49,-83)
            self.picker_preview_item.setPos(point)
            self.picker_preview_item.show()
        if self.drag_mode == 'resize_paste' and self.paste_item is not None:
            base=self.paste_source
            if base is not None:
                p=self.mapToScene(event.position().toPoint())
                x0,y0=self.paste_item.pos().x(),self.paste_item.pos().y()
                scale=max(0.05,min(12.0,
                    max((p.x()-x0)/max(1,base.width),
                        (p.y()-y0)/max(1,base.height))))
                self.paste_item.setScale(scale)
            event.accept()
            return
        if self.drag_mode == 'select' and self.drag_start is not None:
            self.temp_item.setRect(QRectF(self.drag_start, point).normalized())
            event.accept(); return
        if self.drag_mode in ('move_text','resize_text'):
            x,y,w,h=self.drag_box
            dx=int(round(point.x()-self.drag_start.x()))
            dy=int(round(point.y()-self.drag_start.y()))
            max_w=int(self.scene().sceneRect().width())
            max_h=int(self.scene().sceneRect().height())
            if self.drag_mode=='move_text':
                x=max(0,min(max(0,max_w-w),x+dx))
                y=max(0,min(max(0,max_h-h),y+dy))
            else:
                l=x+(dx if 'l' in self.drag_edges else 0)
                r=x+w+(dx if 'r' in self.drag_edges else 0)
                t=y+(dy if 't' in self.drag_edges else 0)
                b=y+h+(dy if 'b' in self.drag_edges else 0)
                l=max(0,min(l,r-18));r=min(max_w,max(r,l+18))
                t=max(0,min(t,b-18));b=min(max_h,max(b,t+18))
                x,y,w,h=l,t,r-l,b-t
            self.text_box_changed.emit((int(x),int(y),int(w),int(h)))
            event.accept();return
        if self.stroke_start is not None:
            previous = self.stroke_start
            self.stroke_start = point
            self.painted.emit((previous,point,self.tool,self.brush_size,False))
            event.accept(); return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drag_mode == 'resize_paste':
                self.drag_mode=None
                event.accept()
                return
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
                    if self.tool in ('text_new','text'):
                        self.text_rectangle_requested.emit(box)
                    elif self.tool == 'cleanup_add':
                        self.cleanup_box_drawn.emit(box)
                    else:
                        self.rectangle_selected.emit(box)
                event.accept(); return
            if self.drag_mode in ('resize_text','move_text'):
                self.drag_mode=None
                self.drag_box=None
                self.drag_start=None
                self.text_geometry_finished.emit()
                event.accept()
                return
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
        self.history = []  # domain-specific transactions
        self.future = []
        self.settings = QSettings('MangaOffline','MangaOfflineTranslator')
        self.brush_color = QColor(str(self.settings.value('brushColor','#232323')))
        self.text_color = QColor(str(self.settings.value('textColor','#232323')))
        self.last_saved = None
        self.dirty = False
        self._in_sync = False
        self._stroke_in_progress = False
        self._stroke_before = None
        self._text_before = None
        self._cleanup_candidates=[]
        self._cleanup_enabled=[]
        self._cleanup_done_boxes=[]
        self._cleanup_done_selections=set()
        self._cleanup_preview_active=False
        self._cleanup_preview_image=None
        self._cleanup_preview_boxes=()
        self._cleanup_preview_base=None
        self._cleanup_preview_mask=None
        self._cleanup_preview_method=None
        self._cleanup_base_mask=None
        self._cleanup_region=None
        self._cleanup_labels=None
        self._cleanup_label_ids=[]
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._refresh_pixmap)
        self.setWindowTitle('Редактор — '+self.path.name)
        self.resize(1410,900)
        self.setMinimumSize(900,570)
        self._ui()
        self.canvas.set_picture(self.working)
        self.canvas.set_tool('cleanup')
        self._sync()
        self.statusBar().showMessage('Выделите прямоугольник: фрагмент немедленно копируется как изображение.',9500)

    def _zoom_combo_activated(self,index):
        self._apply_zoom_text(self.zoom_combo.itemText(index))

    def _zoom_combo_typed(self):
        self._apply_zoom_text(self.zoom_combo.currentText())

    def _apply_zoom_text(self,value):
        name=value.strip().replace(' ','')
        if name.lower()=='вписать':
            self.canvas.fit_picture()
            return
        try:
            factor=float(name.rstrip('%').replace(',','.'))/100
        except ValueError:
            self._update_zoom_combo(self.canvas.transform().m11())
            return
        if not .05<=factor<=18:
            self._update_zoom_combo(self.canvas.transform().m11())
            return
        self.canvas.resetTransform()
        self.canvas.scale(factor,factor)
        self.canvas.fit=False
        self.canvas._queue_quality()
        self._update_zoom_combo(factor)

    def _update_zoom_combo(self,factor):
        if not hasattr(self,'zoom_combo'):
            return
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText('Вписать' if self.canvas.fit
                                       else f'{factor*100:.0f}%')
        self.zoom_combo.blockSignals(False)

    def _saved_bool(self,key,default=False):
        return str(self.settings.value(key,default)).lower() in ('true','1','yes')

    def show_advanced_cleanup(self):
        self.advanced_cleanup.exec()

    def _update_color_previews(self):
        if not hasattr(self,'color_button'):
            return
        text_color=self.text_color.name()
        self.color_button.setStyleSheet(
            'background:'+text_color+';border:3px solid #54728e;')
        brush=self.brush_color.name()
        darker=self.brush_color.darker(185).name()
        self.brush_color_preview.setStyleSheet(
            'background:qlineargradient(x1:0,y1:0,x2:0,y2:1,'
            'stop:0 '+brush+',stop:1 '+darker+');'
            'border:1px solid #a9b2c5;')

    def _opacity_changed(self,value):
        self.settings.setValue('brushOpacity',int(value))

    def _ui(self):
        main=QWidget()
        self.setCentralWidget(main)
        layout=QHBoxLayout(main)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.canvas=ImageCanvas(self)
        layout.addWidget(self.canvas,1)

        side=QFrame(self)
        side.setObjectName('editorSide')
        side.setMinimumWidth(265)
        side.setMaximumWidth(310)
        side.setStyleSheet("""
        QFrame#editorSide, QWidget#cleanupPane, QWidget#textPane, QWidget#brushPane {
            background:#1b2430;color:#e7edf6;}
        QFrame#editorSide QLabel {background:transparent;color:#d0dcec;}
        QFrame#editorSide QPushButton,QFrame#editorSide QToolButton {
            background:#263545;color:#e7edf6;border:1px solid #3a4d60;
            border-radius:6px;padding:6px;font-size:12px;}
        QFrame#editorSide QPushButton:hover,QFrame#editorSide QToolButton:hover {
            background:#34485d;border-color:#6e9bc3;}
        QFrame#editorSide QPushButton:disabled {
            color:#7d8a9d;background:#202a36;border-color:#303f50;}
        QFrame#editorSide QToolButton:checked {
            background:#2965a0;color:#ffffff;border:1px solid #69b4ed;}
        QFrame#editorSide QComboBox,QFrame#editorSide QFontComboBox,
        QFrame#editorSide QSpinBox,QFrame#editorSide QPlainTextEdit {
            background:#151e29;color:#f0f4f9;
            border:1px solid #40556c;border-radius:5px;padding:6px;}
        QFrame#editorSide QAbstractItemView {
            background:#17212c;color:#f0f4f9;
            selection-background-color:#326da5;
            selection-color:#ffffff;}
        QFrame#editorSide QSlider::groove:horizontal {
            background:#425267;height:3px;}
        QFrame#editorSide QSlider::handle:horizontal {
            background:#65b4f0;width:11px;height:21px;margin:-10px 0;border-radius:4px;}
        """)
        sl=QVBoxLayout(side)
        sl.setContentsMargins(11,12,11,11)
        sl.setSpacing(6)
        self.sidebar_modes=QStackedWidget(side)
        sl.addWidget(self.sidebar_modes,1)
        self.save_button=QPushButton('Сохранить PNG…')
        self.save_button.clicked.connect(self.save_png)
        sl.addWidget(self.save_button)
        layout.addWidget(side)

        # CLEANUP — no debug information, instruction paragraphs or preview.
        self.selection_panel=QWidget()
        self.selection_panel.setObjectName('cleanupPane')
        self.selection_panel.setStyleSheet('QWidget#cleanupPane {background:#1b2430;}')
        sel=QVBoxLayout(self.selection_panel)
        sel.setContentsMargins(2,2,2,4)
        sel.setSpacing(9)
        head=QLabel('Очистка текста')
        head.setStyleSheet('color:#8dcaff;font-size:16px;font-weight:500;')
        sel.addWidget(head)
        sel.addWidget(QLabel('Тип текста'))
        self.cleanup_mode=QComboBox()
        self.cleanup_mode.addItem('Тёмный', 'dark')
        self.cleanup_mode.addItem('Светлый', 'light')
        self.cleanup_mode.addItem('С обводкой', 'outline')
        self.cleanup_mode.addItem('С тёмной обводкой', 'outline_dark')
        self.cleanup_mode.addItem('Ровный фон', 'flat')
        idx=self.cleanup_mode.findData(str(self.settings.value('cleanupMode','outline')))
        self.cleanup_mode.setCurrentIndex(max(0,idx))
        self.cleanup_mode.currentIndexChanged.connect(self._cleanup_parameters_changed)
        sel.addWidget(self.cleanup_mode)
        self.clear_button=QPushButton('Очистить')
        self.clear_button.setMinimumHeight(40)
        self.clear_button.clicked.connect(self.clear_selection)
        self.clear_button.setEnabled(False)
        sel.addWidget(self.clear_button)
        self.cleanup_paste_button=QPushButton('Вставить из буфера')
        self.cleanup_paste_button.clicked.connect(lambda:self.paste_fragment(force_image=True))
        sel.addWidget(self.cleanup_paste_button)
        clear_sel=QPushButton('Снять выделение   Esc')
        clear_sel.clicked.connect(self.cancel_selection)
        sel.addWidget(clear_sel)
        # The controls retained below are not displayed in the release UI.
        self.cleanup_stats=QLabel('')
        self.cleanup_stats.hide()
        self.detect_button=QPushButton('Найти объекты')
        self.detect_button.hide()
        self.detect_button.clicked.connect(self.detect_cleanup)
        self.preview_button=QPushButton('Предпросмотр')
        self.preview_button.hide()
        self.preview_button.clicked.connect(self.preview_cleanup)
        self.manual_box_button=QPushButton('Рамка')
        self.manual_box_button.hide()
        self.manual_box_button.clicked.connect(lambda:self.set_tool('cleanup_add'))
        sel.addStretch(1)
        self.sidebar_modes.addWidget(self.selection_panel)

        # Advanced settings are only in the toolbar menu. They are persistent.
        self.advanced_cleanup=QDialog(self)
        self.advanced_cleanup.setWindowTitle('Расширенные настройки очистки')
        form=QFormLayout(self.advanced_cleanup)
        self.cleanup_method=QComboBox(self.advanced_cleanup)
        self.cleanup_method.addItem('Telea — текстура и градиенты','telea')
        self.cleanup_method.addItem('Линейный градиент','plane')
        self.cleanup_method.addItem('Квадратичный градиент','quadratic')
        method_idx=self.cleanup_method.findData(str(self.settings.value('cleanupMethod','telea')))
        self.cleanup_method.setCurrentIndex(max(0,method_idx))
        self.cleanup_method.currentIndexChanged.connect(self._cleanup_repair_changed)
        form.addRow('Восстановление фона',self.cleanup_method)
        self.cleanup_grow=QSpinBox(self.advanced_cleanup)
        self.cleanup_grow.setRange(0,3)
        self.cleanup_grow.setValue(int(self.settings.value('cleanupGrow',2)))
        self.cleanup_grow.valueChanged.connect(self._cleanup_parameters_changed)
        form.addRow('Расширение маски, пикс.',self.cleanup_grow)
        close=QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.advanced_cleanup.reject)
        form.addRow(close)

        # TEXT — similar in grouping, size and alignment controls to Paint 3D.
        self.text_panel=QWidget()
        self.text_panel.setObjectName('textPane')
        self.text_panel.setStyleSheet('QWidget#textPane {background:#1b2430;}')
        tl=QVBoxLayout(self.text_panel)
        tl.setContentsMargins(2,2,2,4)
        tl.setSpacing(8)
        title=QLabel('Текст')
        title.setStyleSheet('color:#8dcaff;font-size:16px;font-weight:500;')
        tl.addWidget(title)
        self.family=QFontComboBox()
        self.family.setCurrentFont(QFont(str(self.settings.value('fontFamily','Segoe UI'))))
        self.family.currentFontChanged.connect(self._format_changed)
        self.family.view().setStyleSheet(
            'QAbstractItemView {background:#17212c;color:#f0f4f9;'
            'selection-background-color:#326da5;selection-color:white;}')
        tl.addWidget(self.family)
        row=QHBoxLayout()
        self.font_size=QSpinBox()
        self.font_size.setRange(7,180)
        self.font_size.setValue(int(self.settings.value('fontSize',48)))
        self.font_size.valueChanged.connect(self._format_changed)
        row.addWidget(self.font_size,1)
        self.text_color=QColor(str(self.settings.value('textColor','#101010')))
        self.color_button=QToolButton()
        self.color_button.setToolTip('Цвет текста')
        self.color_button.setFixedSize(40,38)
        self.color_button.clicked.connect(self.choose_text_color)
        row.addWidget(self.color_button)
        tl.addLayout(row)
        style_row=QHBoxLayout()
        self.bold_button=QToolButton()
        self.bold_button.setText('Ж')
        self.bold_button.setToolTip('Полужирный')
        self.bold_button.setCheckable(True)
        self.bold_button.setChecked(self._saved_bool('fontBold'))
        self.bold_button.toggled.connect(self._format_changed)
        style_row.addWidget(self.bold_button)
        self.italic_button=QToolButton()
        self.italic_button.setText('К')
        self.italic_button.setToolTip('Курсив')
        italic_font=QFont('Segoe UI',15)
        italic_font.setItalic(True)
        self.italic_button.setFont(italic_font)
        self.italic_button.setCheckable(True)
        self.italic_button.setChecked(self._saved_bool('fontItalic'))
        self.italic_button.toggled.connect(self._format_changed)
        style_row.addWidget(self.italic_button)
        self.underline_button=QToolButton()
        self.underline_button.setText('Ч̲')
        self.underline_button.setToolTip('Подчёркнутый')
        self.underline_button.setCheckable(True)
        self.underline_button.setChecked(self._saved_bool('fontUnderline'))
        self.underline_button.toggled.connect(self._format_changed)
        style_row.addWidget(self.underline_button)
        tl.addLayout(style_row)
        self.align_buttons={}
        self.align_group=QButtonGroup(self)
        self.align_group.setExclusive(True)
        align_row=QHBoxLayout()
        for key,icon_text in (('left','≡'),('center','☷'),('right','≣')):
            b=QToolButton()
            b.setText(icon_text)
            b.setCheckable(True)
            b.setToolTip({'left':'По левому краю','center':'По центру',
                          'right':'По правому краю'}[key])
            b.setMinimumHeight(37)
            b.clicked.connect(self._format_changed)
            self.align_group.addButton(b)
            self.align_buttons[key]=b
            align_row.addWidget(b)
        self.align_buttons.get(str(self.settings.value('fontAlign','left')),
                               self.align_buttons['left']).setChecked(True)
        tl.addLayout(align_row)
        self.text_input=QPlainTextEdit()
        self.text_input.setUndoRedoEnabled(False)
        self.text_input.installEventFilter(self)
        self.text_input.setPlaceholderText('Введите или вставьте перевод')
        self.text_input.textChanged.connect(self._text_in_panel)
        self.text_input.setMaximumHeight(230)
        tl.addWidget(self.text_input)
        # Text rectangle is drawn directly by mouse.
        tl.addStretch(1)
        self.sidebar_modes.addWidget(self.text_panel)

        # BRUSHES — Paint 3D-style, 5 tools per row, sliders and color grid.
        self.brush_panel=QScrollArea()
        self.brush_panel.setFrameShape(QFrame.Shape.NoFrame)
        self.brush_panel.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.brush_panel.setWidgetResizable(True)
        brush_inner=QWidget()
        brush_inner.setObjectName('brushPane')
        brush_inner.setStyleSheet('QWidget#brushPane {background:#1b2430;}')
        self.brush_panel.setWidget(brush_inner)
        bl=QVBoxLayout(brush_inner)
        bl.setContentsMargins(2,2,2,4)
        bl.setSpacing(8)
        self.brush_title=QLabel('Карандаш')
        self.brush_title.setStyleSheet('color:#8dcaff;font-size:16px;font-weight:500;')
        bl.addWidget(self.brush_title)
        tools_grid=QGridLayout()
        tools_grid.setHorizontalSpacing(5)
        tools_grid.setVerticalSpacing(4)
        self.brush_tool_buttons={}
        brush_names=(
            ('pencil','Карандаш'),('fill','Заливка'),
            ('spray','Баллончик'),('blur','Размывайка'),
            ('restore','Ластик — восстановление оригинала'))
        for index,(name,label) in enumerate(brush_names):
            b=QToolButton()
            b.setIcon(brush_icon(name))
            b.setIconSize(QSize(37,37))
            b.setFixedSize(47,51)
            b.setCheckable(True)
            b.setToolTip(label)
            b.clicked.connect(lambda checked=False,tool=name:self.set_tool(tool))
            self.brush_tool_buttons[name]=b
            tools_grid.addWidget(b,index//5,index%5)
        bl.addLayout(tools_grid)
        self.brush_labels=dict(brush_names)
        self.brush_labels['picker']='Пипетка'

        thickness_head=QHBoxLayout()
        thickness_head.addWidget(QLabel('Толщина'))
        thickness_head.addStretch()
        self.brush_diameter=QSpinBox()
        self.brush_diameter.setSuffix(' пкс')
        self.brush_diameter.setRange(1,160)
        saved_tool=str(self.settings.value('lastBrush','pencil'))
        if saved_tool not in VISIBLE_PAINT_TOOLS:saved_tool='pencil'
        self.last_brush=saved_tool
        self.brush_diameter.setValue(int(self.settings.value('brushSize/'+saved_tool,5)))
        self.brush_diameter.valueChanged.connect(self._brush_size_changed)
        thickness_head.addWidget(self.brush_diameter)
        bl.addLayout(thickness_head)
        self.brush_slider=QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1,160)
        self.brush_slider.setValue(self.brush_diameter.value())
        self.brush_slider.valueChanged.connect(self.brush_diameter.setValue)
        self.brush_diameter.valueChanged.connect(self.brush_slider.setValue)
        bl.addWidget(self.brush_slider)

        opacity_head=QHBoxLayout()
        opacity_head.addWidget(QLabel('Непрозрачность'))
        opacity_head.addStretch()
        self.brush_opacity=QSpinBox()
        self.brush_opacity.setRange(0,100)
        self.brush_opacity.setSuffix('%')
        self.brush_opacity.setValue(int(self.settings.value('brushOpacity',100)))
        self.brush_opacity.valueChanged.connect(self._opacity_changed)
        opacity_head.addWidget(self.brush_opacity)
        bl.addLayout(opacity_head)
        self.opacity_slider=QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0,100)
        self.opacity_slider.setValue(self.brush_opacity.value())
        self.opacity_slider.valueChanged.connect(self.brush_opacity.setValue)
        self.brush_opacity.valueChanged.connect(self.opacity_slider.setValue)
        bl.addWidget(self.opacity_slider)
        bl.addStretch(1)

        self.other_color=QPushButton('Другое  ▾')
        self.other_color.clicked.connect(self.choose_brush_color)
        bl.addWidget(self.other_color)
        color_row=QHBoxLayout()
        self.brush_color=QColor(str(self.settings.value('brushColor','#2538ba')))
        self.brush_color_preview=QPushButton()
        self.brush_color_preview.setFixedHeight(47)
        self.brush_color_preview.clicked.connect(self.choose_brush_color)
        color_row.addWidget(self.brush_color_preview,1)
        self.picker_button=QToolButton()
        self.picker_button.setIcon(brush_icon('picker'))
        self.picker_button.setIconSize(QSize(30,30))
        self.picker_button.setFixedSize(47,47)
        self.picker_button.setToolTip('Пипетка: наведите на пиксель и нажмите')
        self.picker_button.clicked.connect(lambda:self.set_tool('picker'))
        self.brush_tool_buttons['picker']=self.picker_button
        color_row.addWidget(self.picker_button)
        bl.addLayout(color_row)
        colors=(
            '#ffffff','#d0d0d0','#9d9d9d','#555555','#171717','#910018',
            '#f1161d','#ff7b20','#ffbd00','#ffe9a0','#fff500','#b6ff00',
            '#04c759','#80f5e8','#00c8f9','#473dc2','#ba39ad','#ffb5d4',
            '#af7c55','#30304d','#0b76bb','#006d92','#8250a6','#bc88b2')
        swatch_grid=QGridLayout()
        swatch_grid.setSpacing(0)
        for i,color in enumerate(colors):
            b=QToolButton()
            b.setFixedSize(39,34)
            b.setStyleSheet('background:'+color+';border:0;border-radius:0;')
            b.setToolTip(color)
            b.clicked.connect(lambda checked=False,c=color:self._set_brush_color(c))
            swatch_grid.addWidget(b,i//6,i%6)
        bl.addLayout(swatch_grid)
        self.add_color_button=QPushButton('＋  Другой цвет')
        self.add_color_button.clicked.connect(self.choose_brush_color)
        bl.addWidget(self.add_color_button)
        self.sidebar_modes.addWidget(self.brush_panel)
        self._update_color_previews()

        # One coherent toolbar: modes, clipboard, history and visible zoom.
        bar=QToolBar('Редактор')
        bar.setObjectName('editorToolbar')
        bar.setMovable(False)
        bar.setIconSize(QSize(24,24))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea,bar)
        self.return_to_viewer=None
        self.back_action=QAction('← Просмотр',self)
        self.back_action.triggered.connect(self.back_to_viewer)
        bar.addAction(self.back_action)
        bar.addSeparator()

        self.actions={}
        for key,label,shortcut,ico in (
            ('cleanup','Очистка','Alt+C',toolbar_icon('cleanup')),
            ('text','Текст','Alt+T',toolbar_icon('text')),
            ('brushes','Кисти','Alt+B',brush_icon('pencil')),
            ('pan','Рука',None,toolbar_icon('pan'))):
            a=QAction(ico,label,self)
            a.setCheckable(True)
            if shortcut:a.setShortcut(QKeySequence(shortcut))
            if key=='brushes':
                a.triggered.connect(lambda:self.set_tool(self.last_brush))
            else:
                a.triggered.connect(lambda checked=False,t=key:self.set_tool(t))
            a.setToolTip(label+((' · '+shortcut) if shortcut else ''))
            bar.addAction(a)
            self.actions[key]=a
        bar.addSeparator()

        self.copy_action=QAction(toolbar_icon('copy'),'Копировать',self)
        self.copy_action.setToolTip('Копировать выделенный фрагмент · Ctrl+C')
        self.copy_action.triggered.connect(lambda:self.copy_fragment(force_image=True))
        bar.addAction(self.copy_action)
        self.paste_action=QAction(toolbar_icon('paste'),'Вставить',self)
        self.paste_action.setToolTip('Вставить изображение из буфера · Ctrl+V')
        self.paste_action.triggered.connect(lambda:self.paste_fragment(force_image=True))
        bar.addAction(self.paste_action)
        bar.addSeparator()

        self.undo_action=QAction(toolbar_icon('undo'),'Отменить',self)
        self.undo_action.setToolTip('Отменить · Ctrl+Z')
        self.undo_action.triggered.connect(self.undo)
        bar.addAction(self.undo_action)
        self.redo_action=QAction(toolbar_icon('redo'),'Повторить',self)
        self.redo_action.setToolTip('Повторить · Ctrl+Y')
        self.redo_action.triggered.connect(self.redo)
        bar.addAction(self.redo_action)
        for action in (self.copy_action,self.paste_action,self.undo_action,self.redo_action):
            button=bar.widgetForAction(action)
            if isinstance(button,QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                button.setMinimumWidth(38)
        bar.addSeparator()

        self.zoom_combo=QComboBox()
        self.zoom_combo.setObjectName('editorZoom')
        self.zoom_combo.setEditable(True)
        self.zoom_combo.setFixedWidth(95)
        self.zoom_combo.addItems(('Вписать','25%','50%','75%','100%',
                                   '125%','150%','200%','400%'))
        self.zoom_combo.setCurrentText('Вписать')
        self.zoom_combo.activated.connect(self._zoom_combo_activated)
        self.zoom_combo.lineEdit().editingFinished.connect(self._zoom_combo_typed)
        bar.addWidget(self.zoom_combo)
        self.actual_action=QAction('100%',self)
        self.actual_action.setToolTip('Исходный масштаб 100%')
        self.actual_action.triggered.connect(self.canvas.actual_size)
        bar.addAction(self.actual_action)

        self.more_button=QToolButton()
        self.more_button.setText('⋯')
        self.more_button.setToolTip('Дополнительно')
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu=QMenu(self.more_button)
        menu.addAction('Расширенные настройки очистки…',self.show_advanced_cleanup)
        self.more_button.setMenu(menu)
        bar.addWidget(self.more_button)
        self.canvas.zoom_changed.connect(self._update_zoom_combo)
        self._update_zoom_combo(self.canvas.transform().m11())

        for shortcut,callback in (
            ('Ctrl+C',self.copy_fragment),
            ('Ctrl+V',self.paste_fragment),
            ('Ctrl+Z',self._undo_shortcut),
            ('Ctrl+Y',self._redo_shortcut),
            ('Ctrl+X',self.cut_fragment),('Delete',self.delete_active),
            ('Ctrl+Shift+Z',self._redo_shortcut),('Escape',self.escape_tool),
            ('Ctrl+Return',self.commit_paste)):
            action=QAction(self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(callback)
            self.addAction(action)
        self.canvas.rectangle_selected.connect(self._selected)
        self.canvas.painted.connect(self._painted)
        self.canvas.color_picked.connect(self._pick)
        self.canvas.text_box_changed.connect(self._resize_text)
        self.canvas.text_geometry_started.connect(self._begin_text_geometry)
        self.canvas.text_geometry_finished.connect(self._finish_text_geometry)
        self.canvas.text_edited.connect(self._text_in_canvas)
        self.canvas.text_commit_requested.connect(self.commit_text)
        self.canvas.text_layer_selected.connect(self._choose_text_layer)
        self.canvas.text_rectangle_requested.connect(self._create_text_box)
        self.canvas.context_menu_requested.connect(self._context_menu)
        self.canvas.cleanup_box_toggled.connect(self._toggle_cleanup_box)
        self.canvas.cleanup_box_drawn.connect(self._manual_cleanup_box)
        self.canvas.paste_confirmed.connect(self.commit_paste)
        self.set_tool('cleanup')
        self._update_history_actions()

    def _mark(self):
        self.dirty=True
        self.setWindowTitle('* Редактор — '+self.path.name)

    def _capture_domain(self, domain):
        if domain == 'text':
            return (deepcopy(self.layers), self.selected_text)
        if domain == 'image':
            return (self.working.copy(), list(self._cleanup_done_boxes),
                    set(self._cleanup_done_selections))
        raise ValueError(domain)

    def _restore_domain(self, domain, state):
        if domain == 'text':
            self.layers = deepcopy(state[0])
            # Undo restores committed artwork, not an open editing frame.
            self.selected_text = -1
        elif domain == 'image':
            self.working = state[0].copy()
            self._cleanup_done_boxes = list(state[1])
            self._cleanup_done_selections = set(state[2])
        else:
            raise ValueError(domain)
        self._text_before = None
        self._reset_cleanup(clear_selection=True)
        if self.selected_text == -1 and domain == 'text':
            self._in_sync = True
            self.text_input.clear()
            self._in_sync = False
        self._sync()

    def _record_transaction(self, domain, before):
        self.history.append((domain, before))
        if len(self.history)>12:self.history.pop(0)
        self.future.clear()
        self._mark()
        self._update_history_actions()

    def _checkpoint(self, domain='image'):
        self._record_transaction(domain,self._capture_domain(domain))

    def eventFilter(self,obj,event):
        if (obj is self.text_input and event.type()==QEvent.Type.KeyPress
                and event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter)
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.commit_text()
            return True
        return super().eventFilter(obj,event)

    def commit_text(self):
        """A complete caption is one undoable transaction, never one per key."""
        if self._text_before is None and self.selected_text < 0:
            return
        before = self._text_before
        if before is None:
            before = self._capture_domain('text')
        # Empty new rectangles are transient GUI state, not image changes.
        if (0 <= self.selected_text < len(self.layers)
                and not self.layers[self.selected_text].text
                and len(self.layers)>len(before[0])):
            self.layers.pop(self.selected_text)
        changed = self.layers != before[0]
        self.selected_text = -1
        self._text_before = None
        self._in_sync = True
        self.text_input.clear()
        self._in_sync = False
        self._sync()
        if changed:
            self._record_transaction('text',before)
        self.canvas.setFocus()


    def _undo_shortcut(self):
        self.undo()

    def _redo_shortcut(self):
        self.redo()

    def undo(self):
        if self._text_before is not None:
            self.commit_text()
        if self.canvas.paste_item is not None:
            self.canvas.cancel_paste()
            self.set_tool('cleanup')
            return
        if not self.history:return
        domain, previous = self.history.pop()
        self.future.append((domain, self._capture_domain(domain)))
        self._restore_domain(domain, previous)
        self._update_history_actions()

    def redo(self):
        if self._text_before is not None:
            self.commit_text()
        if not self.future:return
        domain, next_state = self.future.pop()
        self.history.append((domain, self._capture_domain(domain)))
        self._restore_domain(domain, next_state)
        self._update_history_actions()

    def _update_history_actions(self):
        if hasattr(self,'undo_action'):
            self.undo_action.setEnabled(bool(self.history) or self.canvas.paste_item is not None)
        if hasattr(self,'redo_action'):
            self.redo_action.setEnabled(bool(self.future))

    def _refresh_pixmap(self):
        self.canvas.set_picture(self.working)

    def _sync(self):
        self.canvas.set_picture(self.working)
        self.canvas.set_selection(self.selection)
        self._in_sync=True
        self.canvas.set_text_layers(self.layers,self.selected_text)
        if 0 <= self.selected_text < len(self.layers):
            layer=self.layers[self.selected_text]
            self._load_layer_settings(layer)
            if self.text_input.toPlainText()!=layer.text:
                self.text_input.setPlainText(layer.text)
        self._in_sync=False

    def _selected(self,box):
        self.commit_paste()
        self._reset_cleanup(clear_selection=True)
        self.selection=clip_box(box,self.working.size)
        self.canvas.set_selection(self.selection)
        if self.canvas.tool=='cleanup':
            self.copy_fragment(force_image=True)
            self.detect_cleanup()
        else:
            self.statusBar().showMessage(
                'Фрагмент выделен. Ctrl+C — копировать; Ctrl+V — вставить.',6000)

    def _reset_cleanup(self,clear_selection=False):
        self._invalidate_cleanup_preview()
        self._cleanup_candidates=[]
        self._cleanup_enabled=[]
        self._cleanup_label_ids=[]
        self._cleanup_labels=None
        self._cleanup_base_mask=None
        self._cleanup_region=None
        if clear_selection:
            self.selection=None
            self.canvas.set_selection(None)
        self.canvas.set_cleanup_boxes([],[])
        self.canvas.set_cleanup_mask(None,None)
        self.clear_button.setEnabled(False)

    def cancel_selection(self):
        if self._text_before is not None:
            self.commit_text()
        if self.canvas.paste_item is not None:
            self.canvas.cancel_paste()
        self._reset_cleanup(clear_selection=True)
        self.selected_text=-1
        self.canvas.set_text_layers(self.layers,-1)
        self.set_tool('cleanup')
        self.statusBar().showMessage('Выделение отменено.',3500)

    def _set_brush_color(self,color):
        self.brush_color=QColor(color)
        self.settings.setValue('brushColor',self.brush_color.name())
        self._update_color_previews()

    def _context_menu(self,global_pos):
        self._make_context_menu(global_pos).exec(global_pos)

    def _make_context_menu(self,global_pos):
        self.canvas.setFocus()
        loc=self.canvas.viewport().mapFromGlobal(global_pos)
        scene=self.canvas.mapToScene(loc)
        target=(round(scene.x()),round(scene.y()))
        menu=QMenu(self)
        if self.canvas.paste_item is not None:
            menu.addAction('Применить вставку',self.commit_paste)
            menu.addAction('Отменить вставку',self.escape_tool)
        else:
            if self.selection is not None:
                menu.addAction('Копировать',lambda:self.copy_fragment(force_image=True))
                menu.addAction('Вырезать',self.cut_fragment)
            insert=menu.addAction('Вставить',
                lambda:self.paste_fragment(force_image=True,at=target))
            insert.setEnabled(QApplication.clipboard().mimeData().hasImage())
            if self.selection is not None:
                menu.addSeparator()
                menu.addAction('Снять выделение',self.cancel_selection)
        return menu

    def _focused_text(self):
        return self.text_input.hasFocus() or (self.canvas.text_edit_item is not None
           and self.canvas.text_edit_item.hasFocus())

    def copy_fragment(self,force_image=False):
        if not force_image and self.text_input.hasFocus():
            self.text_input.copy()
            return
        if (not force_image and self.canvas.text_edit_item is not None
                and self.canvas.text_edit_item.hasFocus()):
            selection=self.canvas.text_edit_item.textCursor().selectedText()
            if selection:
                QApplication.clipboard().setText(selection)
                return
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите фрагмент.',4500)
            return
        x,y,w,h=clip_box(self.selection,self.working.size)
        if w*h:
            # Include already entered translated text in copied raster.
            picture=(composite_text(self.working,self.layers)
                     if self.layers else self.working)
            crop=picture.crop((x,y,x+w,y+h))
            QApplication.clipboard().setImage(qimage_from_pil(crop))
            self.statusBar().showMessage('Фрагмент скопирован как изображение.',3500)

    def _erase_image_selection(self):
        if self.selection is None:
            return
        x,y,w,h=clip_box(self.selection,self.working.size)
        if w*h==0:
            return
        src=np.asarray(self.working.convert('RGB'))
        pad=min(22,max(4,min(w,h)//3))
        x0=max(0,x-pad);y0=max(0,y-pad)
        x1=min(src.shape[1],x+w+pad);y1=min(src.shape[0],y+h+pad)
        context=src[y0:y1,x0:x1]
        valid=np.ones(context.shape[:2],dtype=bool)
        valid[y-y0:y-y0+h,x-x0:x-x0+w]=False
        samples=context[valid]
        color=tuple(np.median(samples,axis=0).astype(np.uint8)) if len(samples) else (255,255,255)
        self._checkpoint()
        new=self.working.copy()
        new.paste(Image.new('RGB',(w,h),color),(x,y))
        self.working=new
        self._invalidate_cleanup_preview()
        self._cleanup_candidates=[]
        self._cleanup_enabled=[]
        self._cleanup_base_mask=None
        self._cleanup_labels=None
        self._sync()
        self._refresh_cleanup_overlay()

    def cut_fragment(self):
        if self._focused_text():
            if self.text_input.hasFocus():
                self.text_input.cut()
            elif self.canvas.text_edit_item is not None:
                cur=self.canvas.text_edit_item.textCursor()
                cur.removeSelectedText()
            return
        if self.selection is None:
            return
        self.copy_fragment(force_image=True)
        self._erase_image_selection()

    def paste_fragment(self,force_image=False,at=None):
        clipboard=QApplication.clipboard()
        if not force_image and self.text_input.hasFocus():
            self.text_input.paste()
            return
        if (not force_image and self.canvas.text_edit_item is not None
                and self.canvas.text_edit_item.hasFocus()):
            cursor=self.canvas.text_edit_item.textCursor()
            cursor.insertText(clipboard.text())
            self.canvas.text_edit_item.setTextCursor(cursor)
            return
        q=clipboard.image()
        if q.isNull():
            self.statusBar().showMessage('В буфере обмена нет изображения.',4000)
            return
        pil=qimage_to_pil(q)
        if pil.width>max(5000,self.working.width*3) or pil.height>max(5000,self.working.height*3):
            QMessageBox.warning(self,'Вставка','Фрагмент слишком большой.')
            return
        if at is None:
            center=self.canvas.mapToScene(self.canvas.viewport().rect().center())
            x,y=round(center.x()-pil.width/2),round(center.y()-pil.height/2)
        else:
            x,y=at
        x=max(0,min(self.working.width-15,x))
        y=max(0,min(self.working.height-15,y))
        self.canvas.add_paste(pil,(x,y))
        self.set_tool('paste')
        self.statusBar().showMessage(
            'Перетащите фрагмент. Нижний правый угол — размер; Enter — применить, Esc — отменить.',9000)

    def commit_paste(self):
        item=self.canvas.paste_item
        if item is None:
            return
        source=self.canvas.paste_source.copy()
        scale=item.scale()
        if abs(scale-1)>1e-6:
            source=source.resize((max(1,round(source.width*scale)),
                                  max(1,round(source.height*scale))),
                                 Image.Resampling.LANCZOS)
        x,y=round(item.pos().x()),round(item.pos().y())
        self._checkpoint()
        base=self.working.convert('RGBA')
        pasted=Image.new('RGBA',base.size,(0,0,0,0))
        pasted.paste(source.convert('RGBA'),(x,y))
        self.working=Image.alpha_composite(base,pasted)
        if self.original.mode=='RGB':
            self.working=self.working.convert('RGB')
        self.canvas.cancel_paste()
        self._sync()
        self.set_tool('cleanup')
        self._update_history_actions()

    def escape_tool(self):
        if self.canvas.paste_item:
            self.canvas.cancel_paste()
        self.cancel_selection()

    def _approved_cleanup_mask(self):
        if self._cleanup_base_mask is None or self._cleanup_labels is None:
            return np.zeros((self.working.height,self.working.width),dtype=np.uint8)
        return confirmed_mask(self._cleanup_base_mask,self._cleanup_region,
                              self._cleanup_labels,self._cleanup_label_ids,
                              self._cleanup_candidates,self._cleanup_enabled)

    def _refresh_cleanup_overlay(self):
        enabled=sum(self._cleanup_enabled)
        self.cleanup_stats.setText(
            f'Найдено объектов: {len(self._cleanup_candidates)}. Отмечено: {enabled}.')
        self.clear_button.setEnabled(enabled>0)
        self.preview_button.setEnabled(enabled>0)
        self.preview_button.setText(
            'Вернуться к исходнику' if self._cleanup_preview_active
            else 'Предпросмотр очистки')
        if self._cleanup_preview_active or self.canvas.tool!='cleanup':
            self.canvas.set_cleanup_boxes([],[])
            self.canvas.set_cleanup_mask(None,None)
        else:
            self.canvas.set_cleanup_boxes(self._cleanup_candidates,self._cleanup_enabled)
            self.canvas.set_cleanup_mask(self._approved_cleanup_mask(),self.selection)

    def _cleanup_parameters_changed(self,*_):
        self.settings.setValue('cleanupMode',self.cleanup_mode.currentData())
        self.settings.setValue('cleanupGrow',self.cleanup_grow.value())
        if self.selection is not None and self.selection not in self._cleanup_done_selections:
            self.detect_cleanup()

    def _cleanup_repair_changed(self,*_):
        self.settings.setValue('cleanupMethod',self.cleanup_method.currentData())
        self._invalidate_cleanup_preview()
        self._refresh_cleanup_overlay()

    def _stop_cleanup_preview(self):
        if self._cleanup_preview_active:
            self._cleanup_preview_active=False
            self.canvas.set_picture(self.working)

    def _invalidate_cleanup_preview(self):
        self._stop_cleanup_preview()
        self._cleanup_preview_image=None
        self._cleanup_preview_base=None
        self._cleanup_preview_mask=None
        self._cleanup_preview_method=None

    def detect_cleanup(self):
        self._invalidate_cleanup_preview()
        if self.selection is None:
            self.cleanup_stats.setText('Сначала выделите надпись.')
            return
        if self.selection in self._cleanup_done_selections:
            self._cleanup_candidates=[]
            self._cleanup_enabled=[]
            self._cleanup_base_mask=None
            self._cleanup_labels=None
            self._cleanup_label_ids=[]
            self._refresh_cleanup_overlay()
            self.cleanup_stats.setText('Область уже очищена. Ctrl+Z — отменить.')
            return
        # No OCR: contrast segmentation on a background-dominated rectangle.
        # Each connected component is a candidate for explicit user approval.
        seg=segment_ink(self.working,self.selection,
                        mode=self.cleanup_mode.currentData(),
                        grow=self.cleanup_grow.value())
        self._cleanup_base_mask=seg.mask
        (self._cleanup_region,self._cleanup_candidates,
         self._cleanup_label_ids,self._cleanup_labels)=mask_components(
            seg.mask,self.selection)
        self._cleanup_enabled=[
            not any(self._boxes_overlap(box,done) for done in self._cleanup_done_boxes)
            for box in self._cleanup_candidates]
        self._refresh_cleanup_overlay()
        if not self._cleanup_candidates:
            self.statusBar().showMessage(
                'Контрастные объекты не найдены. Можно добавить рамку вручную.',6500)

    @staticmethod
    def _boxes_overlap(a,b):
        x,y,w,h=a
        xx,yy,ww,hh=b
        return x<xx+ww and x+w>xx and y<yy+hh and y+h>yy

    def _toggle_cleanup_box(self,index):
        self._invalidate_cleanup_preview()
        if not 0<=index<len(self._cleanup_enabled):
            return
        self._cleanup_enabled[index]=not self._cleanup_enabled[index]
        self._refresh_cleanup_overlay()

    def _manual_cleanup_box(self,box):
        self._invalidate_cleanup_preview()
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите надпись.',6000)
            self.set_tool('cleanup')
            return
        x,y,w,h=box
        sx,sy,sw,sh=self.selection
        left=max(x,sx);top=max(y,sy)
        right=min(x+w,sx+sw);bottom=min(y+h,sy+sh)
        if right-left>=3 and bottom-top>=3 and self._cleanup_labels is not None:
            # Explicitly drawn boxes may cover all pixels inside their bounds.
            # Suggested automatic boxes only cover their actual ink pixels.
            self._cleanup_candidates.append((left,top,right-left,bottom-top))
            self._cleanup_label_ids.append(None)
            self._cleanup_enabled.append(True)
            self._refresh_cleanup_overlay()
        self.set_tool('cleanup')

    def preview_cleanup(self):
        if self._cleanup_preview_active:
            self._stop_cleanup_preview()
            self._refresh_cleanup_overlay()
            return
        mask=self._approved_cleanup_mask()
        if not np.any(mask):
            return
        method=self.cleanup_method.currentData()
        if (self._cleanup_preview_image is not None
                and self._cleanup_preview_base is self.working
                and self._cleanup_preview_method==method
                and np.array_equal(mask,self._cleanup_preview_mask)):
            preview=self._cleanup_preview_image
        else:
            preview=repair_background(self.working,mask,method)
            self._cleanup_preview_image=preview
            self._cleanup_preview_base=self.working
            self._cleanup_preview_mask=mask.copy()
            self._cleanup_preview_method=method
        self._cleanup_preview_active=True
        self.canvas.set_cleanup_boxes([],[])
        self.canvas.set_cleanup_mask(None,None)
        self.canvas.set_picture(preview)
        self._refresh_cleanup_overlay()

    def clear_selection(self):
        if self.selection is None:
            self.statusBar().showMessage('Сначала выделите надпись.',6000)
            return
        if self.selection in self._cleanup_done_selections:
            self.statusBar().showMessage(
                'Эта область уже очищена. Повторное применение заблокировано.',6500)
            return
        mask=self._approved_cleanup_mask()
        if not np.any(mask):
            self.statusBar().showMessage('Нет отмеченных пикселей для удаления.',6500)
            return
        method=self.cleanup_method.currentData()
        if (self._cleanup_preview_image is not None
                and self._cleanup_preview_base is self.working
                and self._cleanup_preview_method==method
                and np.array_equal(mask,self._cleanup_preview_mask)):
            result=self._cleanup_preview_image.copy()
        else:
            result=repair_background(self.working,mask,method)
        if result.tobytes()==self.working.tobytes():
            self.statusBar().showMessage('Очистка не изменила изображение.',6500)
            return
        # Invariant: only the approved pixel mask can differ from the source.
        before=np.asarray(self.working.convert('RGBA'))
        after=np.asarray(result.convert('RGBA'))
        if np.any(before[mask==0]!=after[mask==0]):
            raise RuntimeError('Cleanup attempted to change protected image pixels')
        self._checkpoint()
        self.working=result
        self._cleanup_last_applied_mask=mask.copy()
        approved=[box for box,active in zip(self._cleanup_candidates,self._cleanup_enabled)
                  if active]
        self._cleanup_done_boxes.extend(approved)
        self._cleanup_done_selections.add(self.selection)
        self._invalidate_cleanup_preview()
        self._cleanup_candidates=[]
        self._cleanup_enabled=[]
        self._cleanup_label_ids=[]
        self._cleanup_base_mask=None
        self._cleanup_labels=None
        self._reset_cleanup(clear_selection=True)
        self._sync()
        self.set_tool('cleanup')
        self.statusBar().showMessage(
            f'Очищено {np.count_nonzero(mask)} пикселей. Ctrl+Z — отменить.',7500)

    def set_tool(self,tool):
        if tool=='brushes':
            tool=self.last_brush
        if self.canvas.paste_item and tool!='paste':
            self.commit_paste()
        if tool in VISIBLE_PAINT_TOOLS:
            self.last_brush=tool
            self.settings.setValue('lastBrush',tool)
            self.brush_title.setText(self.brush_labels.get(tool,'Кисти'))
            size=int(self.settings.value('brushSize/'+tool,5))
            if size!=self.brush_diameter.value():
                self.brush_diameter.blockSignals(True)
                self.brush_diameter.setValue(size)
                self.brush_diameter.blockSignals(False)
                self.brush_slider.setValue(size)
            self.canvas.brush_size=size
        previous=self.canvas.tool
        if previous=='text' and tool!='text':
            self.commit_text()
        # Each mode owns its own selection state; never retain active cleanup
        # candidates when switching to text, paste or ordinary copy.
        if tool not in ('cleanup','cleanup_add') and previous in ('cleanup','cleanup_add'):
            self._reset_cleanup(clear_selection=(tool not in ('new',)))
        elif tool in ('text','text_new','paste') and self.selection is not None:
            self._reset_cleanup(clear_selection=True)
        self.canvas.set_tool(tool)
        brushes=tool in (*PAINT_TOOLS,'picker')
        section=(2 if brushes else 1 if tool in ('text','text_new') else 0)
        self.sidebar_modes.setCurrentIndex(section)
        mode='brushes' if brushes else ('text' if tool in ('text','text_new') else
                                       'cleanup' if tool in ('cleanup_add','new','paste') else tool)
        for name,action in self.actions.items():
            if action.isCheckable():
                action.setChecked(name==mode)
        for name,button in self.brush_tool_buttons.items():
            button.setChecked(name==tool)
        if tool in ('cleanup','cleanup_add'):
            if tool=='cleanup' and self.selection and not self._cleanup_candidates:
                self.detect_cleanup()
            else:
                self._refresh_cleanup_overlay()
        else:
            self.canvas.set_cleanup_boxes([],[])
            self.canvas.set_cleanup_mask(None,None)
        if tool!='text':
            self.canvas.setFocus()
        names={
            'cleanup':'Протяните рамку текста, исключите лишнее щелчком по объекту, затем «Очистить»',
            'cleanup_add':'Протяните дополнительную рамку очистки',
            'new':'Выделите фрагмент, затем Ctrl+C — копировать',
            'text_new':'Протяните рамку текста мышью',
            'text':'Перемещение за верхний край; растяжение за края и углы',
            'pan':'Перемещение страницы',
            'paste':'Перемещение / растяжение вставленного фрагмента; Enter — применить',
            'picker':'Наведите на пиксель, щёлкните для выбора цвета'
        }
        self.statusBar().showMessage(names.get(tool,self.brush_labels.get(tool,'')),6000)

    def _current_settings(self):
        align=next((key for key,b in self.align_buttons.items() if b.isChecked()),'left')
        return {'family':self.family.currentFont().family(),
                'size':self.font_size.value(),
                'bold':self.bold_button.isChecked(),
                'italic':self.italic_button.isChecked(),
                'underline':self.underline_button.isChecked(),
                'alignment':align,
                'color':self.text_color.name()}

    def insert_text(self):
        """Text placement does not reuse the image selection rectangle."""
        self.commit_paste()
        self.commit_text()
        if 0 <= self.selected_text < len(self.layers):
            self.selected_text=-1
            self._in_sync=True
            self.text_input.clear()
            self._in_sync=False
        self.set_tool('text')
        self.statusBar().showMessage('Протяните мышью новую рамку текста независимо от зоны очистки.',8500)

    def _create_text_box(self,box):
        box=clip_box(box,self.working.size)
        if box[2]<8 or box[3]<8:return
        # Finish the previous caption before opening a fresh, empty box.
        self.commit_text()
        self._text_before = self._capture_domain('text')
        layer=TextLayer(box=box,text='',**self._current_settings())
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
            value=self.text_input.toPlainText()
            if layer.text==value:return
            if self._text_before is None:
                self._text_before=self._capture_domain('text')
            layer.text=value
            self.canvas.update_live_text(layer)

    def _load_layer_settings(self,layer):
        old_sync=self._in_sync
        self._in_sync=True
        controls=(self.family,self.font_size,self.bold_button,
                  self.italic_button,self.underline_button)
        for control in controls:control.blockSignals(True)
        try:
            self.family.setCurrentFont(QFont(layer.family))
            self.font_size.setValue(layer.size)
            self.bold_button.setChecked(layer.bold)
            self.italic_button.setChecked(layer.italic)
            self.underline_button.setChecked(layer.underline)
            self.align_buttons.get(layer.alignment,self.align_buttons['left']).setChecked(True)
            self.text_color=QColor(layer.color)
            self._update_color_previews()
        finally:
            for control in controls:control.blockSignals(False)
            self._in_sync=old_sync

    def _choose_text_layer(self,index):
        if not 0 <= index < len(self.layers):return
        if index != self.selected_text:
            self.commit_text()
            self._text_before=self._capture_domain('text')
        elif self._text_before is None:
            self._text_before=self._capture_domain('text')
        self.selected_text=index
        self._load_layer_settings(self.layers[index])
        self.set_tool('text')
        self._sync()
        if self.canvas.text_edit_item is not None:
            self.canvas.text_edit_item.setFocus()

    def _text_in_canvas(self,value):
        if self.selected_text<0 or self.selected_text>=len(self.layers):return
        if self.layers[self.selected_text].text==value:return
        if self._text_before is None:
            self._text_before=self._capture_domain('text')
        self.layers[self.selected_text].text=value
        self._in_sync=True
        self.text_input.setPlainText(value)
        self._in_sync=False

    def _begin_text_geometry(self):
        if 0 <= self.selected_text < len(self.layers):
            if self._text_before is None:
                self._text_before=self._capture_domain('text')

    def _finish_text_geometry(self):
        if 0 <= self.selected_text < len(self.layers):
            self.canvas.set_text_layers(self.layers,self.selected_text)
            self.canvas.setFocus()

    def _resize_text(self,new_box):
        if self.selected_text<0 or self.selected_text>=len(self.layers):
            return
        box=clip_box(new_box,self.working.size)
        self.layers[self.selected_text].box=box
        self.canvas.set_text_geometry_live(box)

    def _format_changed(self,*_):
        if self._in_sync:return
        settings=self._current_settings()
        for key,val in (('fontFamily',settings['family']),('fontSize',settings['size']),
                        ('fontBold',settings['bold']),('fontItalic',settings['italic']),
                        ('fontUnderline',settings['underline']),
                        ('fontAlign',settings['alignment']),('textColor',settings['color'])):
            self.settings.setValue(key,val)
        self._update_color_previews()
        if 0<=self.selected_text<len(self.layers):
            layer=self.layers[self.selected_text]
            if any(getattr(layer,key)!=val for key,val in settings.items()):
                if self._text_before is None:
                    self._text_before=self._capture_domain('text')
                for key,val in settings.items():setattr(layer,key,val)
                self.canvas.update_live_text(layer)

    def choose_text_color(self):
        chosen=QColorDialog.getColor(self.text_color,self,'Цвет текста')
        if chosen.isValid():
            self.text_color=chosen
            self._format_changed()

    def choose_brush_color(self):
        chosen=QColorDialog.getColor(self.brush_color,self,'Цвет кисти')
        if chosen.isValid():
            self._set_brush_color(chosen.name())

    def _brush_size_changed(self,value):
        self.canvas.brush_size=value
        name=self.last_brush if self.last_brush in VISIBLE_PAINT_TOOLS else 'pencil'
        self.settings.setValue('brushSize/'+name,int(value))

    def _painted(self,event):
        start,end,tool,size,first=event
        if start is None:
            if self._stroke_in_progress and self._stroke_before is not None:
                before=self._stroke_before
                if self.working.tobytes()!=before[0].tobytes():
                    self._record_transaction('image',before)
            self._stroke_before=None
            self._stroke_in_progress=False
            self._refresh_pixmap()
            return
        if first or not self._stroke_in_progress:
            self._stroke_before=self._capture_domain('image')
            self._stroke_in_progress=True
        a=(start.x(),start.y())
        b=(end.x(),end.y())
        # Filling is a click operation; dragging must not flood other areas.
        if tool!='fill' or first:
            self.working=draw_stroke(
                self.working,self.original,a,b,size,
                self.brush_color.getRgb()[:3],tool,
                self.brush_opacity.value())
        if not self._render_timer.isActive():
            self._render_timer.start(35)

    def _pick(self,coords):
        x,y=coords
        x=max(0,min(self.working.width-1,x))
        y=max(0,min(self.working.height-1,y))
        pixel=self.working.convert('RGB').getpixel((x,y))
        self._set_brush_color(QColor(*pixel).name())
        self.set_tool(self.last_brush)
        self.statusBar().showMessage('Цвет: '+self.brush_color.name().upper(),6500)

    def delete_active(self):
        if self._focused_text():
            if self.text_input.hasFocus():
                cursor=self.text_input.textCursor()
                cursor.deleteChar()
            elif self.canvas.text_edit_item is not None:
                cursor=self.canvas.text_edit_item.textCursor()
                cursor.deleteChar()
                self.canvas.text_edit_item.setTextCursor(cursor)
            return
        if self.canvas.paste_item:
            self.canvas.cancel_paste()
            self.set_tool('cleanup')
            return
        if 0<=self.selected_text<len(self.layers) and self.canvas.tool=='text':
            before = self._text_before or self._capture_domain('text')
            self.layers.pop(self.selected_text)
            self.selected_text=-1
            self._text_before=None
            self._in_sync=True
            self.text_input.clear()
            self._in_sync=False
            self._sync()
            if self.layers!=before[0]:
                self._record_transaction('text',before)
            return
        if self.selection:
            self._erase_image_selection()

    def save_png(self):
        self.commit_paste()
        self.commit_text()
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