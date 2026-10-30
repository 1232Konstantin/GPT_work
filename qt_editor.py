"""Qt one-page translation editor. OCR regions require explicit user confirmation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
from PySide6.QtCore import Qt, Signal, QThreadPool, QRectF, QPointF
from PySide6.QtGui import QAction, QPen, QColor, QBrush
from PySide6.QtWidgets import (QMainWindow, QWidget, QDialog, QVBoxLayout, QHBoxLayout,
    QFormLayout, QSplitter, QToolBar, QLabel, QGroupBox, QListWidget, QListWidgetItem,
    QPushButton, QComboBox, QCheckBox, QPlainTextEdit, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem, QFrame,
    QSlider, QSpinBox, QProgressBar)

from engine import (Page, Region, open_image, detect_regions, clip_xywh,
    clean_image, add_translations, numbered_source, numbered_translations,
    parse_numbered, to_json_pages, from_json_pages, save_image, default_font_path)
from qt_common import pixmap_from_pil, Task


class RegionCanvas(QGraphicsView):
    rect_drawn = Signal(object)
    point_clicked = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QBrush(QColor('#0c121b')))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.fit = True
        self.pen_items = []
        self.tool = 'select'
        self.drag_start: QPointF | None = None
        self.temp = None

    def set_picture(self, picture: Image.Image):
        self.scene().clear()
        self.scene().addPixmap(pixmap_from_pil(picture))
        self.scene().setSceneRect(0, 0, picture.width, picture.height)
        self.pen_items.clear()
        if self.fit:
            self.fit_picture()

    def fit_picture(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit:
            self.fit_picture()

    def wheelEvent(self, event):
        self.fit = False
        factor = 1.18 if event.angleDelta().y() > 0 else 1/1.18
        self.scale(factor, factor)

    def set_tool(self, tool):
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if tool == 'pan'
                         else QGraphicsView.DragMode.NoDrag)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.tool in ('new', 'ink', 'box'):
            self.drag_start = self.mapToScene(event.position().toPoint())
            pen = QPen(QColor('#62d5e9'), 2, Qt.PenStyle.DashLine)
            self.temp = self.scene().addRect(QRectF(self.drag_start, self.drag_start), pen)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.tool == 'select':
            self.point_clicked.emit(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start is not None and self.temp:
            self.temp.setRect(QRectF(self.drag_start, self.mapToScene(event.position().toPoint())).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_start is not None:
            rect = QRectF(self.drag_start, self.mapToScene(event.position().toPoint())).normalized()
            self.drag_start = None
            if self.temp:
                self.scene().removeItem(self.temp)
                self.temp = None
            if rect.width() >= 4 and rect.height() >= 4:
                self.rect_drawn.emit((round(rect.x()), round(rect.y()),
                                      round(rect.width()), round(rect.height())))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def show_regions(self, regions, selected_id, enabled=True):
        for item in self.pen_items:
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self.pen_items.clear()
        if not enabled:
            return
        for r in regions:
            chosen = r.id == selected_id
            for box, color, style in ((r.ink, '#ff8585', Qt.PenStyle.SolidLine),
                                      (r.box, '#64dfb0', Qt.PenStyle.DashLine)):
                pen = QPen(QColor(color if r.confirmed else '#fbc66a'), 2.5 if chosen else 1.5, style)
                item = self.scene().addRect(QRectF(*box), pen)
                item.setZValue(10)
                item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                self.pen_items.append(item)
            x, y, _, _ = r.ink
            label = self.scene().addText(f'{r.id} {"✓" if r.confirmed else "!"}')
            label.setDefaultTextColor(QColor('#a8f3e0' if r.confirmed else '#ffd28c'))
            label.setPos(x, max(0, y-25))
            label.setZValue(11)
            label.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
            self.pen_items.append(label)


class TranslationDialog(QDialog):
    def __init__(self, page: Page, on_apply, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Обмен переводами · номера сохраняются')
        self.resize(1080, 660)
        self.page = page
        self.on_apply = on_apply
        layout = QVBoxLayout(self)
        hint = QLabel('Скопируйте исходные строки, переведите любым способом и вставьте справа. '
                      'Номера вида [0001] должны сохраниться.')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        texts = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel('Японский оригинал'))
        self.source = QPlainTextEdit(numbered_source([page]))
        self.source.setReadOnly(True)
        left.addWidget(self.source)
        right = QVBoxLayout()
        right.addWidget(QLabel('Русский / английский перевод'))
        self.target = QPlainTextEdit(numbered_translations([page]))
        right.addWidget(self.target)
        texts.addLayout(left, 1)
        texts.addLayout(right, 1)
        layout.addLayout(texts, 1)
        buttons = QHBoxLayout()
        copy = QPushButton('Копировать оригинал')
        copy.clicked.connect(lambda: self.source.selectAll() or self.source.copy())
        paste = QPushButton('Вставить перевод')
        paste.clicked.connect(self.target.paste)
        save = QPushButton('Применить переводы')
        save.setProperty('primary', True)
        save.clicked.connect(self.apply)
        buttons.addWidget(copy)
        buttons.addWidget(paste)
        buttons.addStretch()
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def apply(self):
        try:
            parsed = parse_numbered(self.target.toPlainText())
            expected = {r.id for r in self.page.regions if r.action == 'translate'}
            unknown = set(parsed) - expected
            if unknown:
                raise ValueError('Неизвестные номера: ' + ', '.join(map(str, sorted(unknown))))
            missing = expected - set(parsed)
            if missing and QMessageBox.question(self, 'Неполный перевод',
                f'Отсутствуют {len(missing)} переводов. Сохранить прежний текст для них?') != QMessageBox.StandardButton.Yes:
                return
            self.on_apply(parsed)
            self.accept()
        except ValueError as exc:
            QMessageBox.warning(self, 'Ошибка нумерации', str(exc))


class EditorWindow(QMainWindow):
    def __init__(self, image_path: Path):
        super().__init__()
        self.path = Path(image_path).resolve()
        self.page = Page(str(self.path))
        self.original = open_image(str(self.path))
        self.selected_id = None
        self.dirty = False
        self.project_path: Path | None = None
        self.preview_generation = 0
        self.ocr_generation = 0
        self.pending: set[Task] = set()
        self.busy = False
        self.font_path = default_font_path()
        self._loading_form = False
        self._build_ui()
        self.canvas.set_picture(self.original)
        self.refresh_regions()
        self._title()
        self.statusBar().showMessage('Распознайте текст, затем проверьте каждую область.')

    def task(self, function, callback, error=None):
        job = Task(function)
        self.pending.add(job)
        def done(result):
            self.pending.discard(job)
            callback(result)
        def failed(message):
            self.pending.discard(job)
            (error or self._error)(message)
        job.signals.result.connect(done)
        job.signals.error.connect(failed)
        QThreadPool.globalInstance().start(job)

    def _error(self, text):
        self.busy = False
        self.progress.setVisible(False)
        QMessageBox.warning(self, 'Не удалось выполнить операцию', text)
        self.statusBar().showMessage(text, 10000)

    def _title(self):
        mark = ' *' if self.dirty else ''
        self.setWindowTitle(f'Редактор перевода  |  {self.path.name}{mark}')

    def mark_dirty(self):
        self.dirty = True
        self._title()

    def _act(self, bar, title, callback, shortcut=None):
        action = QAction(title, self)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        bar.addAction(action)
        return action

    def _build_ui(self):
        self.resize(1480, 920)
        toolbar = QToolBar('Редактор')
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self._act(toolbar, '◎  Распознать OCR', self.recognize)
        self._act(toolbar, '≡  Строки перевода', self.exchange, 'Ctrl+T')
        toolbar.addSeparator()
        self._act(toolbar, '◧  Сохранить очищенное', lambda: self.export('clean'))
        self._act(toolbar, '▣  Сохранить перевод', lambda: self.export('translated'))
        toolbar.addSeparator()
        self._act(toolbar, 'Сохранить проект', self.save_project, 'Ctrl+S')
        self._act(toolbar, 'Открыть проект', self.load_project)

        main = QWidget()
        self.setCentralWidget(main)
        outer = QVBoxLayout(main)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)
        central = QWidget()
        cv = QVBoxLayout(central)
        cv.setContentsMargins(0, 0, 7, 0)
        top = QHBoxLayout()
        title = QLabel(self.path.name)
        title.setStyleSheet('font-size:17px; font-weight:650;')
        top.addWidget(title, 1)
        self.mode = QComboBox()
        self.mode.addItems(['Оригинал', 'Очищено', 'Перевод'])
        self.mode.currentIndexChanged.connect(self.refresh_preview)
        top.addWidget(self.mode)
        fit = QPushButton('Вписать')
        fit.clicked.connect(self.canvas_fit)
        top.addWidget(fit)
        self.bounds = QCheckBox('Области')
        self.bounds.setChecked(True)
        self.bounds.toggled.connect(self.paint_overlays)
        top.addWidget(self.bounds)
        cv.addLayout(top)
        self.canvas = RegionCanvas()
        self.canvas.rect_drawn.connect(self.draw_rect)
        self.canvas.point_clicked.connect(self.hit_test)
        self.canvas.setFrameShape(QFrame.Shape.NoFrame)
        cv.addWidget(self.canvas, 1)
        help_label = QLabel('Красный: область стирания · Зелёный: зона перевода · Жёлтый: не подтверждено · Колёсико: масштаб')
        help_label.setProperty('muted', True)
        cv.addWidget(help_label)
        split.addWidget(central)

        side = QWidget()
        side.setMinimumWidth(330)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(5, 0, 0, 0)
        sv.setSpacing(8)
        heading = QLabel('ОБЛАСТИ ТЕКСТА')
        heading.setStyleSheet('font-weight:700; letter-spacing:2px; color:#97a9bf;')
        sv.addWidget(heading)
        self.regions = QListWidget()
        self.regions.setMaximumHeight(220)
        self.regions.currentRowChanged.connect(self.region_chosen)
        sv.addWidget(self.regions)
        tools = QGroupBox('Инструмент на изображении')
        tl = QVBoxLayout(tools)
        self.tool = QComboBox()
        self.tool.addItem('Выбрать область', 'select')
        self.tool.addItem('Добавить область', 'new')
        self.tool.addItem('Изменить стирание (красная)', 'ink')
        self.tool.addItem('Изменить размещение (зелёная)', 'box')
        self.tool.addItem('Перемещать изображение', 'pan')
        self.tool.currentIndexChanged.connect(lambda _: self.canvas.set_tool(self.tool.currentData()))
        tl.addWidget(self.tool)
        sv.addWidget(tools)
        fields = QGroupBox('Содержимое выделенной области')
        form = QVBoxLayout(fields)
        self.action = QComboBox()
        self.action.addItem('Перевести', 'translate')
        self.action.addItem('Скрыть', 'hide')
        self.action.addItem('Оставить', 'keep')
        form.addWidget(QLabel('Что сделать с надписью'))
        form.addWidget(self.action)
        self.balloon = QCheckBox('Внутри облачка (автоподбор шрифта)')
        form.addWidget(self.balloon)
        form.addWidget(QLabel('Распознанный текст (можно исправить)'))
        self.source = QPlainTextEdit()
        self.source.setMaximumHeight(84)
        form.addWidget(self.source)
        form.addWidget(QLabel('Перевод'))
        self.translated = QPlainTextEdit()
        self.translated.setMaximumHeight(110)
        form.addWidget(self.translated)
        self.apply = QPushButton('✓  Применить и подтвердить')
        self.apply.setProperty('primary', True)
        self.apply.clicked.connect(self.confirm_region)
        form.addWidget(self.apply)
        delete = QPushButton('Удалить область')
        delete.clicked.connect(self.remove_region)
        form.addWidget(delete)
        sv.addWidget(fields)
        settings = QGroupBox('Параметры обработки')
        fl = QFormLayout(settings)
        self.ocr_dir = QComboBox()
        self.ocr_dir.addItem('Японский — оба направления', 'both')
        self.ocr_dir.addItem('Вертикальный', 'vertical')
        self.ocr_dir.addItem('Горизонтальный', 'horizontal')
        fl.addRow('OCR', self.ocr_dir)
        self.strength = QSpinBox()
        self.strength.setRange(0, 100)
        self.strength.setValue(60)
        self.strength.valueChanged.connect(self.refresh_preview)
        fl.addRow('Очистка, %', self.strength)
        self.font_percent = QSpinBox()
        self.font_percent.setRange(50, 160)
        self.font_percent.setSingleStep(5)
        self.font_percent.setValue(100)
        self.font_percent.valueChanged.connect(self.refresh_preview)
        fl.addRow('Шрифт, %', self.font_percent)
        sv.addWidget(settings)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        sv.addWidget(self.progress)
        sv.addStretch(1)
        split.addWidget(side)
        split.setSizes([1040, 390])

    def canvas_fit(self):
        self.canvas.fit_picture()

    def _region(self):
        return next((r for r in self.page.regions if r.id == self.selected_id), None)

    def store_draft(self):
        r = self._region()
        if r is None or self._loading_form:
            return
        data = (self.source.toPlainText(), self.translated.toPlainText(),
                self.action.currentData(), self.balloon.isChecked())
        if data != (r.source, r.translated, r.action, r.balloon):
            r.source, r.translated, r.action, r.balloon = data
            r.confirmed = False
            self.mark_dirty()

    def region_chosen(self, row):
        self.store_draft()
        self.selected_id = self.page.regions[row].id if 0 <= row < len(self.page.regions) else None
        r = self._region()
        self._loading_form = True
        self.source.setPlainText(r.source if r else '')
        self.translated.setPlainText(r.translated if r else '')
        self.action.setCurrentIndex(next((i for i in range(self.action.count())
            if self.action.itemData(i) == (r.action if r else 'translate')), 0))
        self.balloon.setChecked(r.balloon if r else False)
        self._loading_form = False
        self.paint_overlays()

    def select_region(self, rid):
        for idx, r in enumerate(self.page.regions):
            if r.id == rid:
                self.regions.setCurrentRow(idx)
                return

    def refresh_regions(self):
        self.regions.blockSignals(True)
        self.regions.clear()
        for r in self.page.regions:
            prefix = '✓' if r.confirmed else '!'
            type_ = {'translate':'Перевести','hide':'Скрыть','keep':'Оставить'}.get(r.action, '?')
            item = QListWidgetItem(f'{prefix}   {r.id:04d}  · {type_}  · {r.source[:24]}')
            item.setForeground(QColor('#a4ebd1' if r.confirmed else '#ffd292'))
            self.regions.addItem(item)
        index = next((n for n, r in enumerate(self.page.regions) if r.id == self.selected_id), -1)
        self.regions.setCurrentRow(index)
        self.regions.blockSignals(False)
        self.paint_overlays()

    def paint_overlays(self):
        self.canvas.show_regions(self.page.regions, self.selected_id, self.bounds.isChecked())

    def hit_test(self, point):
        x, y = point.x(), point.y()
        candidates = [r for r in self.page.regions if any(
            rx <= x <= rx+rw and ry <= y <= ry+rh for rx, ry, rw, rh in (r.ink, r.box))]
        if candidates:
            self.select_region(min(candidates, key=lambda r: r.box[2]*r.box[3]).id)

    def draw_rect(self, bounds):
        tool = self.tool.currentData()
        rect = clip_xywh(bounds, self.original.width, self.original.height)
        if rect[2] < 4 or rect[3] < 4:
            return
        self.store_draft()
        if tool == 'new':
            rid = max((r.id for r in self.page.regions), default=0)+1
            self.page.regions.append(Region(rid, rect, rect, orientation='manual'))
            self.selected_id = None
        else:
            r = self._region()
            if not r:
                QMessageBox.information(self, 'Выбор области', 'Сначала выберите область в списке.')
                return
            if tool == 'ink':
                r.ink = rect
            elif tool == 'box':
                r.box = rect
            else:
                return
            r.confirmed = False
        self.mark_dirty()
        self.refresh_regions()
        self.region_chosen(next((i for i, r in enumerate(self.page.regions)
                                 if r.id == (rid if tool == 'new' else self.selected_id)), -1))
        self.tool.setCurrentIndex(0)
        self.refresh_preview()

    def confirm_region(self):
        r = self._region()
        if not r:
            return
        self.store_draft()
        r.confirmed = True
        self.mark_dirty()
        self.refresh_regions()
        self.refresh_preview()

    def remove_region(self):
        if not self._region():
            return
        self.page.regions = [r for r in self.page.regions if r.id != self.selected_id]
        self.selected_id = None
        self.mark_dirty()
        self.refresh_regions()
        self.region_chosen(-1)
        self.refresh_preview()

    def recognize(self):
        if self.busy:
            return
        self.store_draft()
        if self.page.regions and QMessageBox.question(self, 'Повторное распознавание',
            'Заменить все области? Введённые переводы будут потеряны.') != QMessageBox.StandardButton.Yes:
            return
        self.busy = True
        self.ocr_generation += 1
        gen = self.ocr_generation
        self.progress.setVisible(True)
        self.statusBar().showMessage('Распознавание японского текста локальным Tesseract…')
        mode = self.ocr_dir.currentData()
        def ready(regions):
            if gen != self.ocr_generation:
                return
            self.busy = False
            self.progress.setVisible(False)
            for rid, region in enumerate(regions, 1):
                region.id = rid
                region.confirmed = False
            self.page.regions = regions
            self.selected_id = None  # Old form must never overwrite fresh OCR results.
            self.mark_dirty()
            self.refresh_regions()
            self.region_chosen(0 if regions else -1)
            self.refresh_preview()
            self.statusBar().showMessage(f'OCR: найдено {len(regions)} областей. Проверьте каждую.', 10000)
        self.task(lambda _: detect_regions(self.original, mode), ready)

    def processed(self, mode, regions=None, strength=None, font_percent=None):
        regions = regions if regions is not None else self.page.regions
        cleaned = clean_image(self.original, regions, strength if strength is not None else self.strength.value())
        if mode == 'translated':
            return add_translations(cleaned, regions, self.font_path,
                font_percent if font_percent is not None else self.font_percent.value())
        return cleaned

    def refresh_preview(self):
        if not hasattr(self, 'canvas'):
            return
        self.store_draft()
        idx = self.mode.currentIndex()
        if idx == 0:
            self.preview_generation += 1
            self.canvas.set_picture(self.original)
            self.paint_overlays()
            return
        self.preview_generation += 1
        gen = self.preview_generation
        snap = deepcopy(self.page.regions)
        strength = self.strength.value()
        font_percent = self.font_percent.value()
        mode = 'clean' if idx == 1 else 'translated'
        self.statusBar().showMessage('Подготовка предпросмотра…')
        def ready(image):
            if gen != self.preview_generation:
                return
            self.canvas.set_picture(image)
            self.paint_overlays()
            self.statusBar().showMessage('Предпросмотр готов', 4000)
        self.task(lambda _: self.processed(mode, snap, strength, font_percent), ready,
                  error=lambda msg: self.statusBar().showMessage(msg, 9000))

    def exchange(self):
        self.store_draft()
        if not any(r.action == 'translate' for r in self.page.regions):
            QMessageBox.information(self, 'Перевод', 'Нет областей с действием «Перевести».')
            return
        def apply(values):
            selected = self.selected_id
            for r in self.page.regions:
                if r.id in values:
                    r.translated = values[r.id]
            self.mark_dirty()
            self.selected_id = None  # Avoid overwriting inserted translations from stale text edits.
            self.refresh_regions()
            self.region_chosen(next((i for i, r in enumerate(self.page.regions)
                                     if r.id == selected), -1))
            self.refresh_preview()
        TranslationDialog(self.page, apply, self).exec()

    def unconfirmed(self):
        self.store_draft()
        missing = [r.id for r in self.page.regions if not r.confirmed]
        if missing:
            QMessageBox.warning(self, 'Проверка областей',
                'Сначала подтвердите все области: ' + ', '.join(f'[{n:04d}]' for n in missing))
            return True
        return False

    def export(self, mode):
        if self.busy or self.unconfirmed():
            return
        suffix = '_clean' if mode == 'clean' else '_translated'
        default = str(self.path.with_name(self.path.stem + suffix + '.png'))
        name, selected_filter = QFileDialog.getSaveFileName(self, 'Экспорт изображения', default,
            'PNG без потерь (*.png);;JPEG высокого качества (*.jpg)')
        if not name:
            return
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix('.jpg' if 'JPEG' in selected_filter else '.png')
        if path.resolve() == self.path:
            QMessageBox.warning(self, 'Оригинал', 'Нельзя перезаписывать исходное изображение.')
            return
        try:
            save_image(self.processed(mode), str(path))
            self.statusBar().showMessage(f'Сохранено: {path}', 9000)
        except Exception as exc:
            self._error(str(exc))

    def save_project(self):
        self.store_draft()
        default = str(self.project_path or self.path.with_suffix('.manga.json'))
        name, _ = QFileDialog.getSaveFileName(self, 'Сохранить проект', default, 'Manga JSON (*.json)')
        if not name:
            return False
        path = Path(name)
        try:
            path.write_text(json.dumps({'version': 1, 'pages': to_json_pages([self.page]),
                'strength': self.strength.value(), 'font_percent': self.font_percent.value()},
                ensure_ascii=False, indent=2), encoding='utf-8')
            self.project_path = path
            self.dirty = False
            self._title()
            self.statusBar().showMessage('Проект сохранён', 4500)
            return True
        except OSError as exc:
            self._error(str(exc))
            return False

    def load_project(self):
        if not self.may_discard():
            return
        name, _ = QFileDialog.getOpenFileName(self, 'Открыть проект', '', 'Manga JSON (*.json)')
        if not name:
            return
        try:
            data = json.loads(Path(name).read_text(encoding='utf-8'))
            pages = from_json_pages(data['pages'])
            if len(pages) != 1 or Path(pages[0].path).resolve() != self.path:
                raise ValueError('Проект должен содержать эту же страницу.')
            self.page = pages[0]
            self.selected_id = None  # Old form must never overwrite the loaded project.
            self.strength.setValue(data.get('strength', 60))
            self.font_percent.setValue(data.get('font_percent', 100))
            self.project_path = Path(name)
            self.dirty = False
            self._title()
            self.refresh_regions()
            self.region_chosen(0 if self.page.regions else -1)
            self.refresh_preview()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._error(str(exc))

    def may_discard(self):
        self.store_draft()
        if not self.dirty:
            return True
        result = QMessageBox.question(self, 'Несохранённые изменения',
            'В проекте есть несохранённые изменения. Закрыть без сохранения?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        return result == QMessageBox.StandardButton.Yes

    def closeEvent(self, event):
        if self.may_discard():
            self.preview_generation += 1
            self.ocr_generation += 1
            event.accept()
        else:
            event.ignore()