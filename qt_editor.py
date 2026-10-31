"""User-directed single-page translation studio: selection -> local OCR -> boundary -> preview."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
from PIL import Image
from PySide6.QtCore import Qt, Signal, QPointF, QRectF, QThreadPool
from PySide6.QtGui import QAction, QColor, QPen, QBrush, QPainterPath, QFont, QKeySequence, QPainter
from PySide6.QtWidgets import (QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QListWidget, QListWidgetItem, QPushButton, QComboBox, QCheckBox,
    QPlainTextEdit, QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene,
    QFrame, QSpinBox, QProgressBar, QColorDialog, QApplication, QScrollArea,
    QGraphicsItem, QButtonGroup)
from engine import (Page, Region, open_image, clip_xywh, clean_image,
    add_translations, numbered_source, numbered_translations, parse_numbered,
    to_json_pages, from_json_pages, save_image, default_font_path)
from manual_regions import suggest_boundary, inner_text_box, recognize_selection
from qt_common import pixmap_from_pil, Task


def boundary_rect(r):
    x,y,w,h=r.box
    return [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]


class Canvas(QGraphicsView):
    rectangle_drawn=Signal(object)
    clicked=Signal(object)
    polygon_changed=Signal(object)
    vertex_started=Signal()
    vertex_inserted=Signal(object)
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self)); self.setBackgroundBrush(QBrush(QColor('#10141a')))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.tool='new'; self.fit=True; self.drag=None; self.temp=None
        self.overlays=[]; self.points=[]; self.active_id=None; self.handle=-1;self.segments=[]
    def set_picture(self,image):
        z=self.transform()
        self.scene().clear(); self.overlays=[]; self.points=[]; self.temp=None;self.segments=[]
        self.scene().addPixmap(pixmap_from_pil(image))
        self.scene().setSceneRect(0,0,image.width,image.height)
        if self.fit: self.fit_picture()
        else: self.setTransform(z)
    def fit_picture(self):
        if self.scene().sceneRect().isEmpty(): return
        self.fitInView(self.scene().sceneRect(),Qt.AspectRatioMode.KeepAspectRatio)
        self.fit=True
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self.fit: self.fit_picture()
    def wheelEvent(self,event):
        self.fit=False
        self.scale(1.15 if event.angleDelta().y()>0 else 1/1.15,
                   1.15 if event.angleDelta().y()>0 else 1/1.15)
    def set_tool(self,tool):
        self.tool=tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool=='pan' else QGraphicsView.DragMode.NoDrag)
    def show_overlays(self,regions,selected,show=True):
        for item in self.overlays:
            if item.scene()==self.scene():self.scene().removeItem(item)
        self.overlays=[]; self.points=[]; self.segments=[]; self.active_id=selected
        if not show:return
        for r in regions:
            isactive=r.id==selected
            # Dark-red polygon: translation boundary. Coral rectangles: OCR glyphs.
            coords=r.polygon if len(r.polygon)>=3 else boundary_rect(r)
            path=QPainterPath(QPointF(*coords[0]))
            for p in coords[1:]:path.lineTo(*p)
            path.closeSubpath()
            outline=self.scene().addPath(path,QPen(QColor('#f06c70' if isactive else '#b65057'),
                 2.7 if isactive else 1.3,Qt.PenStyle.SolidLine))
            outline.setZValue(8);outline.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.overlays.append(outline)
            if isactive:
                self.segments=[(i,QPointF(*coords[i]),QPointF(*coords[(i+1)%len(coords)])) for i in range(len(coords))]
                for index,(x,y) in enumerate(coords):
                    radius=5 / max(.4,self.transform().m11())
                    dot=self.scene().addEllipse(x-radius,y-radius,radius*2,radius*2,
                        QPen(QColor('#e94c5b'),1),QBrush(QColor('#ffffff')))
                    dot.setZValue(22);dot.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self.overlays.append(dot);self.points.append((index,QPointF(x,y)))
                for x,y,w,h in r.glyph_boxes:
                    rect=self.scene().addRect(x,y,w,h,QPen(QColor('#ffe270'),1.4))
                    rect.setZValue(13);rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                    self.overlays.append(rect)
                x,y,w,h=r.ink
                mark=self.scene().addRect(x,y,w,h,QPen(QColor('#63c9f0'),1,Qt.PenStyle.DashLine))
                mark.setZValue(7);mark.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.overlays.append(mark)
            x,y,_,_=r.ink
            label=self.scene().addText(str(r.id));label.setDefaultTextColor(QColor('#ffda9d'))
            label.setPos(x,max(0,y-25));label.setZValue(18)
            label.setAcceptedMouseButtons(Qt.MouseButton.NoButton);self.overlays.append(label)
    def mousePressEvent(self,event):
        if event.button()==Qt.MouseButton.LeftButton and self.tool in ('new','ink','rect'):
            self.drag=self.mapToScene(event.position().toPoint())
            self.temp=self.scene().addRect(QRectF(self.drag,self.drag),QPen(QColor('#69caf0'),2,Qt.PenStyle.DashLine))
            event.accept();return
        if event.button()==Qt.MouseButton.LeftButton and self.tool=='boundary':
            pos=self.mapToScene(event.position().toPoint())
            radius=12/max(.4,self.transform().m11())
            candidate=[(i,(pt-pos).manhattanLength()) for i,pt in self.points if (pt-pos).manhattanLength()<radius]
            if candidate:
                self.handle=min(candidate,key=lambda t:t[1])[0]
                self.vertex_started.emit()
                event.accept();return
            # Clicking directly on a red edge adds a new draggable vertex.
            threshold=15/max(.4,self.transform().m11())
            for i,a,b in self.segments:
                ab=b-a; ap=pos-a
                length=ab.x()*ab.x()+ab.y()*ab.y()
                if length<.001:continue
                t=max(0,min(1,(ap.x()*ab.x()+ap.y()*ab.y())/length))
                nearest=a+ab*t
                if (pos-nearest).manhattanLength()<threshold:
                    self.vertex_inserted.emit((i+1,(round(pos.x()),round(pos.y()))))
                    self.handle=i+1
                    event.accept();return
        if event.button()==Qt.MouseButton.LeftButton and self.tool=='select':
            self.clicked.emit(self.mapToScene(event.position().toPoint()))
            event.accept();return
        super().mousePressEvent(event)
    def mouseMoveEvent(self,event):
        if self.drag is not None and self.temp is not None:
            self.temp.setRect(QRectF(self.drag,self.mapToScene(event.position().toPoint())).normalized())
            event.accept();return
        if self.handle>=0:
            pos=self.mapToScene(event.position().toPoint())
            self.polygon_changed.emit((self.handle,(round(pos.x()),round(pos.y())),False))
            event.accept();return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self,event):
        if self.drag is not None:
            rect=QRectF(self.drag,self.mapToScene(event.position().toPoint())).normalized()
            self.drag=None
            if self.temp is not None:
                self.scene().removeItem(self.temp);self.temp=None
            if rect.width()>5 and rect.height()>5:
                self.rectangle_drawn.emit((round(rect.x()),round(rect.y()),round(rect.width()),round(rect.height())))
            event.accept();return
        if self.handle>=0:
            pos=self.mapToScene(event.position().toPoint())
            self.polygon_changed.emit((self.handle,(round(pos.x()),round(pos.y())),True))
            self.handle=-1;event.accept();return
        super().mouseReleaseEvent(event)


class ExchangeDialog(QDialog):
    def __init__(self,page,apply,parent=None):
        super().__init__(parent);self.page=page;self.apply_callback=apply
        self.setWindowTitle('Перевести сразу несколько надписей');self.resize(1000,570)
        root=QVBoxLayout(self)
        label=QLabel('Скопируйте текст слева, переведите удобным способом и вставьте справа. Сохраните номера [0001], [0002]…')
        label.setWordWrap(True);root.addWidget(label)
        row=QHBoxLayout();root.addLayout(row,1)
        for text,attr,editable in [('Оригинальный японский текст','source',False),('Перевод · русский или английский','target',True)]:
            column=QVBoxLayout();column.addWidget(QLabel(text));field=QPlainTextEdit();field.setReadOnly(not editable)
            field.setPlainText(numbered_translations([page]) if editable else numbered_source([page]))
            column.addWidget(field);row.addLayout(column,1);setattr(self,attr,field)
        controls=QHBoxLayout();root.addLayout(controls)
        a=QPushButton('Скопировать оригинал');a.clicked.connect(lambda:QApplication.clipboard().setText(self.source.toPlainText()))
        b=QPushButton('Вставить из буфера');b.clicked.connect(self.target.paste)
        c=QPushButton('Применить перевод');c.setProperty('primary',True);c.clicked.connect(self.apply)
        controls.addWidget(a);controls.addWidget(b);controls.addStretch();controls.addWidget(c)
    def apply(self):
        try:
            values=parse_numbered(self.target.toPlainText());valid={r.id for r in self.page.regions if r.action=='translate'}
            if set(values)-valid: raise ValueError('Неизвестный номер надписи')
            self.apply_callback(values);self.accept()
        except ValueError as e:QMessageBox.warning(self,'Ошибка нумерации',str(e))


class EditorWindow(QMainWindow):
    def __init__(self,image_path:Path):
        super().__init__();self.path=Path(image_path).resolve();self.page=Page(str(self.path))
        self.original=open_image(str(self.path));self.selected_id=None;self.project_path=None
        self.dirty=False;self._loading=False;self.pending=set();self.preview_number=0
        self.ocr_number=0;self.history=[];self.redo_stack=[];self.font_path=default_font_path()
        self._build_ui();self.canvas.set_picture(self.original);self.refresh_list()
        self.statusBar().showMessage('1. Выделите надпись мышью → 2. Исправьте контур → 3. Введите перевод.')
    def _build_ui(self):
        self.resize(1470,920);self.setWindowTitle('Перевод страницы — '+self.path.name)
        outer=QWidget();self.setCentralWidget(outer);layout=QVBoxLayout(outer)
        layout.setContentsMargins(8,8,8,5);layout.setSpacing(7)
        top=QHBoxLayout();layout.addLayout(top)
        title=QLabel('Перевод страницы');title.setStyleSheet('font-size:19px;font-weight:650')
        top.addWidget(title);top.addStretch()
        def button(text,func,primary=False):
            b=QPushButton(text);b.clicked.connect(func)
            if primary:b.setProperty('primary',True)
            top.addWidget(b);return b
        button('↶ Отменить',self.undo);button('↷ Повторить',self.redo)
        button('Сохранить проект',self.save_project)
        button('Открыть проект',self.load_project)
        button('Экспорт очищенного',lambda:self.export('clean'))
        button('Экспорт перевода',lambda:self.export('translated'),True)
        split=QSplitter(Qt.Orientation.Horizontal);layout.addWidget(split,1)
        visual=QWidget();vl=QVBoxLayout(visual);vl.setContentsMargins(0,0,0,0)
        instructions=QLabel('Выделите текст прямоугольником · красная линия — зона перевода · жёлтые рамки — распознанные слова')
        instructions.setProperty('muted',True);vl.addWidget(instructions)
        self.canvas=Canvas();self.canvas.rectangle_drawn.connect(self.draw_rect)
        self.canvas.clicked.connect(self.hit_test);self.canvas.polygon_changed.connect(self.move_vertex)
        self.canvas.vertex_started.connect(self.checkpoint)
        self.canvas.vertex_inserted.connect(self.insert_vertex)
        vl.addWidget(self.canvas,1)
        bottom=QHBoxLayout();vl.addLayout(bottom)
        self.tool=QComboBox()
        for label,key in [('＋ Выделить надпись','new'),('Выбрать','select'),('Изменять красный контур','boundary'),
                          ('Задать прямоугольный контур','rect'),('Изменить область очистки','ink'),('Перемещать','pan')]:
            self.tool.addItem(label,key)
        self.tool.currentIndexChanged.connect(lambda _:self.canvas.set_tool(self.tool.currentData()))
        bottom.addWidget(self.tool)
        self.mode=QComboBox();self.mode.addItems(['Оригинал','После очистки','С переводом'])
        self.mode.currentIndexChanged.connect(self.preview);bottom.addWidget(self.mode)
        fit=QPushButton('Вписать в окно');fit.clicked.connect(self.canvas.fit_picture);bottom.addWidget(fit)
        self.show_boxes=QCheckBox('Показать границы');self.show_boxes.setChecked(True)
        self.show_boxes.toggled.connect(self.paint);bottom.addWidget(self.show_boxes)
        split.addWidget(visual)
        sidebar=QWidget();sidebar.setMinimumWidth(360);sidebar.setMaximumWidth(480)
        side=QVBoxLayout(sidebar);side.setContentsMargins(6,0,3,0);side.setSpacing(7)
        side.addWidget(QLabel('НАДПИСИ НА СТРАНИЦЕ'))
        self.regions=QListWidget();self.regions.setMaximumHeight(155)
        self.regions.currentRowChanged.connect(self.region_chosen);side.addWidget(self.regions)
        a=QHBoxLayout();side.addLayout(a)
        delbtn=QPushButton('Удалить область');delbtn.clicked.connect(self.remove_region);a.addWidget(delbtn)
        resetbtn=QPushButton('Контур → прямоугольник');resetbtn.clicked.connect(self.reset_boundary);a.addWidget(resetbtn)
        side.addWidget(QLabel('Оригинальная надпись (OCR, можно исправить)'))
        self.source=QPlainTextEdit();self.source.setMaximumHeight(90);side.addWidget(self.source)
        opts=QHBoxLayout();side.addLayout(opts)
        self.ocr_dir=QComboBox()
        for title,key in [('Японский: авто','both'),('Вертикальный','vertical'),('Горизонтальный','horizontal')]:self.ocr_dir.addItem(title,key)
        opts.addWidget(self.ocr_dir)
        reocr=QPushButton('Повторить OCR выделения');reocr.clicked.connect(self.recognize_selected);opts.addWidget(reocr)
        side.addWidget(QLabel('Что сделать с надписью'))
        self.action=QComboBox()
        for label,value in [('Заменить переводом','translate'),('Только скрыть','hide'),('Оставить как есть','keep')]:
            self.action.addItem(label,value)
        side.addWidget(self.action)
        side.addWidget(QLabel('Новый текст (русский / английский)'))
        self.target=QPlainTextEdit();self.target.setMaximumHeight(120);side.addWidget(self.target)
        fontrow=QHBoxLayout();side.addLayout(fontrow)
        fontrow.addWidget(QLabel('Размер:'))
        self.font=QSpinBox();self.font.setRange(0,180);self.font.setSpecialValueText('Авто');fontrow.addWidget(self.font)
        self.bold=QCheckBox('Ж');fontrow.addWidget(self.bold)
        self.italic=QCheckBox('Курсив');fontrow.addWidget(self.italic)
        self.font_color=''
        self.color_btn=QPushButton('Цвет текста');self.color_btn.clicked.connect(self.choose_color)
        fontrow.addWidget(self.color_btn)
        self.strength=QSpinBox();self.strength.setRange(0,100);self.strength.setValue(60)
        settings=QHBoxLayout();side.addLayout(settings)
        settings.addWidget(QLabel('Очистка, %'));settings.addWidget(self.strength)
        settings.addWidget(QLabel('0 = авто размер шрифта'))
        apply=QPushButton('Применить изменения · Ctrl+Enter');apply.setProperty('primary',True)
        apply.clicked.connect(self.commit);side.addWidget(apply)
        batch=QPushButton('Скопировать / вставить переводы по номерам');batch.clicked.connect(self.exchange)
        side.addWidget(batch)
        self.progress=QProgressBar();self.progress.setRange(0,0);self.progress.hide();side.addWidget(self.progress)
        side.addStretch(1)
        split.addWidget(sidebar);split.setSizes([1040,420])
        for shortcut,func in [('Ctrl+S',self.save_project),('Ctrl+Z',self.undo),('Ctrl+Y',self.redo),
                              ('Ctrl+Shift+Z',self.redo),('Delete',self.delete_shortcut),
                              ('Ctrl+Return',self.commit),('Ctrl+T',self.exchange),('Ctrl+C',self.copy_selected),
                              ('Ctrl+V',self.paste_selected)]:
            act=QAction(self);act.setShortcut(QKeySequence(shortcut));act.triggered.connect(func);self.addAction(act)
    def _field_focus(self):
        focus=QApplication.focusWidget()
        return isinstance(focus,QPlainTextEdit)
    def copy_selected(self):
        if self._field_focus():
            QApplication.focusWidget().copy();return
        r=self.selected();
        if r:QApplication.clipboard().setText(r.source)
    def paste_selected(self):
        if self._field_focus():
            QApplication.focusWidget().paste();return
        r=self.selected()
        if r:self.target.setPlainText(QApplication.clipboard().text());self.commit()
    def delete_shortcut(self):
        if self._field_focus():return
        self.remove_region()
    def task(self,fn,done):
        task=Task(fn);self.pending.add(task)
        def ready(result):self.pending.discard(task);done(result)
        def failed(message):
            self.pending.discard(task);self.progress.hide();QMessageBox.warning(self,'Ошибка обработки',message)
        task.signals.result.connect(ready);task.signals.error.connect(failed)
        QThreadPool.globalInstance().start(task)
    def selected(self):return next((r for r in self.page.regions if r.id==self.selected_id),None)
    def checkpoint(self):
        self.history.append(deepcopy(self.page.regions))
        if len(self.history)>70:self.history.pop(0)
        self.redo_stack.clear();self.dirty=True;self.setWindowTitle('* Перевод — '+self.path.name)
    def undo(self):
        if self._field_focus() and QApplication.focusWidget().document().isUndoAvailable():
            QApplication.focusWidget().undo();return
        if not self.history:return
        self.redo_stack.append(deepcopy(self.page.regions));self.page.regions=self.history.pop()
        self.selected_id=None;self.refresh_list();self.preview()
    def redo(self):
        if not self.redo_stack:return
        self.history.append(deepcopy(self.page.regions));self.page.regions=self.redo_stack.pop()
        self.selected_id=None;self.refresh_list();self.preview()
    def refresh_list(self):
        self.regions.blockSignals(True);self.regions.clear()
        for r in self.page.regions:
            label={'translate':'Перевести','hide':'Скрыть','keep':'Оставить'}[r.action]
            self.regions.addItem(QListWidgetItem(f'#{r.id}  · {label}   {r.source[:22] or "(распознавание…)"}'))
        row=next((i for i,r in enumerate(self.page.regions) if r.id==self.selected_id),-1)
        self.regions.setCurrentRow(row);self.regions.blockSignals(False)
        self.fill_form();self.paint()
    def fill_form(self):
        self._loading=True;r=self.selected()
        self.source.setPlainText(r.source if r else '')
        self.target.setPlainText(r.translated if r else '')
        for i in range(self.action.count()):
            if self.action.itemData(i)==(r.action if r else 'translate'):self.action.setCurrentIndex(i)
        self.font.setValue(r.font_size if r else 0)
        self.bold.setChecked(bool(r and r.bold));self.italic.setChecked(bool(r and r.italic))
        self.font_color=r.color if r else ''
        self.color_btn.setStyleSheet('border: 2px solid '+r.color if r and r.color else '')
        self._loading=False
    def region_chosen(self,row):
        # Auto-save editing when changing selections, but not programmatic refreshes.
        if not self._loading:self.store_form()
        self.selected_id=self.page.regions[row].id if 0<=row<len(self.page.regions) else None
        self.fill_form();self.paint()
    def store_form(self):
        r=self.selected()
        if not r or self._loading:return
        new=(self.source.toPlainText(),self.target.toPlainText(),self.action.currentData(),
             self.font.value(),self.font_color,self.bold.isChecked(),self.italic.isChecked())
        old=(r.source,r.translated,r.action,r.font_size,r.color,r.bold,r.italic)
        if new!=old:
            self.checkpoint();(r.source,r.translated,r.action,r.font_size,r.color,r.bold,r.italic)=new
            r.confirmed=True
    def commit(self):self.store_form();self.refresh_list();self.preview()
    def choose_color(self):
        c=QColorDialog.getColor(QColor(self.font_color or '#222222'),self,'Цвет текста')
        if c.isValid():self.font_color=c.name();self.color_btn.setStyleSheet('border: 2px solid '+c.name())
    def hit_test(self,p):
        for r in reversed(self.page.regions):
            x,y,w,h=r.box
            if x<=p.x()<=x+w and y<=p.y()<=y+h:
                for i,item in enumerate(self.page.regions):
                    if item.id==r.id:self.regions.setCurrentRow(i);return
    def paint(self):self.canvas.show_overlays(self.page.regions,self.selected_id,self.show_boxes.isChecked())
    def choose_tool(self,key):
        for i in range(self.tool.count()):
            if self.tool.itemData(i)==key:self.tool.setCurrentIndex(i);return
    def draw_rect(self,bounds):
        rect=clip_xywh(bounds,*self.original.size)
        if rect[2]<6 or rect[3]<6:return
        self.store_form();mode=self.tool.currentData()
        if mode=='new':
            self.checkpoint();rid=max([r.id for r in self.page.regions],default=0)+1
            r=Region(rid,rect,rect,orientation='manual');self.page.regions.append(r)
            self.selected_id=rid;self.refresh_list();self.progress.show()
            original=self.original.copy();direction=self.ocr_dir.currentData()
            def work(_):return suggest_boundary(original,rect),recognize_selection(original,rect,direction)
            def ready(result):
                self.progress.hide()
                current=next((r for r in self.page.regions if r.id==rid),None)
                if not current:return
                polygon,ocr=result
                current.polygon=polygon
                if polygon:
                    current.box=inner_text_box(polygon,rect,self.original.size)
                    current.balloon=True
                current.source=ocr.text;current.confidence=ocr.confidence
                current.orientation=ocr.orientation;current.glyph_boxes=ocr.glyph_boxes
                self.refresh_list();self.preview()
                self.statusBar().showMessage('OCR — подсказка: исправьте при необходимости. Красные точки перетаскиваются в режиме «контур».',10000)
            self.task(work,ready)
        else:
            r=self.selected()
            if not r:return
            self.checkpoint()
            if mode=='ink':r.ink=rect;r.glyph_boxes=[]
            elif mode=='rect':r.box=rect;r.polygon=boundary_rect(r)
            else:return
            self.refresh_list();self.preview()
        self.choose_tool('boundary' if mode=='new' else 'select')
    def move_vertex(self,event):
        r=self.selected()
        if not r:return
        index,coords,finished=event
        if not r.polygon:r.polygon=boundary_rect(r)
        if index>=len(r.polygon):return
        x,y=coords;r.polygon[index]=(max(0,min(x,self.original.width)),max(0,min(y,self.original.height)))
        r.box=inner_text_box(r.polygon,r.box,self.original.size)
        self.paint()
        if finished:self.preview()
    def insert_vertex(self,event):
        r=self.selected()
        if not r:return
        self.checkpoint()
        index,point=event
        if not r.polygon:r.polygon=boundary_rect(r)
        r.polygon.insert(index,point)
        self.paint()
    def reset_boundary(self):
        r=self.selected()
        if not r:return
        self.checkpoint();r.polygon=[];r.box=r.ink;r.balloon=False
        self.paint();self.preview();self.choose_tool('rect')
        self.statusBar().showMessage('Контур удалён. Теперь выделите новый прямоугольник на изображении.',7000)
    def remove_region(self):
        r=self.selected()
        if not r:return
        self.checkpoint();self.page.regions.remove(r);self.selected_id=None
        self.refresh_list();self.preview()
    def recognize_selected(self):
        self.store_form();r=self.selected()
        if not r:return
        self.ocr_number+=1;gen=self.ocr_number;rid=r.id;ink=r.ink
        self.progress.show();direction=self.ocr_dir.currentData();image=self.original.copy()
        def ready(result):
            self.progress.hide()
            current=next((r for r in self.page.regions if r.id==rid),None)
            if gen!=self.ocr_number or not current:return
            self.checkpoint();current.source=result.text
            current.glyph_boxes=result.glyph_boxes;current.confidence=result.confidence
            current.orientation=result.orientation
            self.refresh_list();self.preview()
        self.task(lambda _:recognize_selection(image,ink,direction),ready)
    def process(self,translated=False,regions=None):
        regions=deepcopy(regions or self.page.regions)
        cleaned=clean_image(self.original,regions,self.strength.value())
        return add_translations(cleaned,regions,self.font_path) if translated else cleaned
    def preview(self):
        if not hasattr(self,'mode'):return
        self.preview_number+=1;version=self.preview_number
        if self.mode.currentIndex()==0:self.canvas.set_picture(self.original);self.paint();return
        regions=deepcopy(self.page.regions);strength=self.strength.value();which=self.mode.currentIndex()
        original=self.original.copy();font=self.font_path
        def ready(image):
            if version!=self.preview_number:return
            self.canvas.set_picture(image);self.paint()
        self.task(lambda _:add_translations(clean_image(original,regions,strength),regions,font)
                  if which==2 else clean_image(original,regions,strength),ready)
    def exchange(self):
        self.store_form()
        def apply(values):
            self.checkpoint()
            for r in self.page.regions:
                if r.id in values:r.translated=values[r.id]
            self.refresh_list();self.preview()
        ExchangeDialog(self.page,apply,self).exec()
    def export(self,mode):
        self.store_form()
        suggestion=self.path.with_name(self.path.stem+('_clean' if mode=='clean' else '_translated')+'.png')
        name,fl=QFileDialog.getSaveFileName(self,'Экспорт',str(suggestion),'PNG без потерь (*.png);;JPEG (*.jpg)')
        if not name:return
        path=Path(name).with_suffix('.jpg' if 'JPEG' in fl and not Path(name).suffix else Path(name).suffix or '.png')
        if path.resolve()==self.path:QMessageBox.warning(self,'Оригинал','Исходный файл нельзя перезаписывать.');return
        try:save_image(self.process(mode=='translated'),str(path));self.statusBar().showMessage('Сохранено: '+str(path),7000)
        except Exception as e:QMessageBox.warning(self,'Экспорт',str(e))
    def save_project(self):
        self.store_form();p=self.project_path or self.path.with_suffix('.manga.json')
        name,_=QFileDialog.getSaveFileName(self,'Сохранить проект',str(p),'Проект манги (*.json)')
        if not name:return False
        try:
            Path(name).write_text(json.dumps({'version':2,'pages':to_json_pages([self.page]),
                'strength':self.strength.value()},ensure_ascii=False,indent=2),encoding='utf8')
            self.project_path=Path(name);self.dirty=False;self.setWindowTitle('Перевод — '+self.path.name);return True
        except OSError as e:QMessageBox.warning(self,'Сохранение',str(e));return False
    def load_project(self):
        self.store_form()
        if self.dirty and QMessageBox.question(self,'Несохранённый проект',
            'Продолжить без сохранения текущих правок?', QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)!=QMessageBox.StandardButton.Yes:return
        file,_=QFileDialog.getOpenFileName(self,'Открыть проект',str(self.path.parent),'Проекты манги (*.json)')
        if not file:return
        try:
            data=json.loads(Path(file).read_text(encoding='utf8'))
            pages=from_json_pages(data['pages'])
            if len(pages)!=1 or Path(pages[0].path).resolve()!=self.path:
                raise ValueError('Проект предназначен для другого изображения.')
            self.page=pages[0];self.project_path=Path(file);self.selected_id=None
            self.history.clear();self.redo_stack.clear();self.dirty=False
            self.strength.setValue(data.get('strength',60))
            self.refresh_list();self.preview()
            self.setWindowTitle('Перевод — '+self.path.name)
        except (OSError,ValueError,KeyError,TypeError) as e:
            QMessageBox.warning(self,'Не удалось открыть проект',str(e))
    def closeEvent(self,event):
        self.store_form()
        if not self.dirty:return super().closeEvent(event)
        answer=QMessageBox.question(self,'Несохранённый проект','Закрыть без сохранения проекта?',
              QMessageBox.StandardButton.Yes|QMessageBox.StandardButton.No,QMessageBox.StandardButton.No)
        if answer==QMessageBox.StandardButton.Yes:event.accept()
        else:event.ignore()