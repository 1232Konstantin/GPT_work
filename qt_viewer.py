"""Modern Qt image viewer, folder browser and exact duplicate search."""
from __future__ import annotations

from pathlib import Path
from collections import OrderedDict
import threading

from PIL import Image, ImageOps
from PySide6.QtCore import Qt, QSize, QThreadPool, QTimer
from PySide6.QtGui import QAction, QColor, QBrush, QIcon, QPainter
from PySide6.QtWidgets import (QMainWindow, QWidget, QSplitter, QVBoxLayout,
    QHBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QListWidget,
    QListWidgetItem, QLabel, QToolBar, QCheckBox, QComboBox, QPushButton,
    QFileDialog, QMessageBox, QDialog, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QProgressBar, QScrollArea, QFrame)

from engine import open_image
from viewer_core import discover_images, find_duplicates, sort_images
from qt_common import Task, pixmap_from_pil


class PictureView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.setBackgroundBrush(QBrush(QColor('#0e131b')))
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.is_fit = True
        self.angle = 0

    def set_picture(self, image: Image.Image):
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pixmap_from_pil(image))
        self.angle = 0
        self.is_fit = True
        self.fit_picture()

    def fit_picture(self):
        if self.pixmap_item:
            self.resetTransform()
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self.is_fit = True

    def actual_size(self):
        if self.pixmap_item:
            self.resetTransform()
            self.is_fit = False
            self.centerOn(self.pixmap_item)

    def zoom(self, amount):
        if self.pixmap_item:
            self.scale(amount, amount)
            self.is_fit = False

    def rotate(self, degrees):
        if self.pixmap_item:
            self.angle = (self.angle + degrees) % 360
            # Rotate around image center, not its upper-left corner.
            rect = self.pixmap_item.boundingRect()
            self.pixmap_item.setTransformOriginPoint(rect.center())
            self.pixmap_item.setRotation(self.angle)
            self.scene().setSceneRect(self.scene().itemsBoundingRect())
            if self.is_fit:
                self.fit_picture()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.is_fit:
            self.fit_picture()

    def wheelEvent(self, event):
        self.zoom(1.18 if event.angleDelta().y() > 0 else 1 / 1.18)


class DuplicateDialog(QDialog):
    def __init__(self, scan, jump_to, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Точные дубликаты — группы совпадений')
        self.resize(870, 530)
        self.jump_to = jump_to
        outer = QVBoxLayout(self)
        outer.addWidget(QLabel(f'Групп: {len(scan.groups)} · Проверено файлов: {scan.checked} · Ошибок: {len(scan.errors)}'))
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Совпадающие изображения', 'Тип совпадения'])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for n, group in enumerate(scan.groups, 1):
            kind = 'Идентичные файлы' if group.kind == 'file' else 'Идентичные пиксели'
            parent_item = QTreeWidgetItem([f'Группа {n} · {len(group.paths)} файла · {group.width}×{group.height}', kind])
            self.tree.addTopLevelItem(parent_item)
            for path in group.paths:
                child = QTreeWidgetItem([str(path), ''])
                child.setData(0, Qt.ItemDataRole.UserRole, path)
                parent_item.addChild(child)
            parent_item.setExpanded(True)
        self.tree.itemDoubleClicked.connect(self._open)
        outer.addWidget(self.tree)
        outer.addWidget(QLabel('Двойной щелчок по файлу — открыть в просмотрщике. Удаление не производится.'))

    def _open(self, item, _column):
        path = item.data(0, Qt.ItemDataRole.UserRole)
        if path:
            self.jump_to(path)
            self.accept()


class ViewerWindow(QMainWindow):
    COLOR_GROUPS = ('#356479', '#6b5439', '#65507b', '#48684f', '#67535d', '#4b5e80')

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Manga Offline Translator — просмотр изображений')
        self.resize(1450, 910)
        self.setMinimumSize(850, 560)
        self.threadpool = QThreadPool.globalInstance()
        self.pending: set[Task] = set()
        self.folder: Path | None = None
        self.paths: list[Path] = []
        self.visible: list[Path] = []
        self.current: Path | None = None
        self.counter = 0
        self.folder_generation = 0
        self.image_generation = 0
        self.thumb_generation = 0
        self.duplicate_generation = 0
        self.dup_scan = None
        self.dup_map: dict[Path, int] = {}
        self.cancel_event = threading.Event()
        self.scanning = False
        self.editor_windows = []
        self.thumbnail_cache: OrderedDict[Path, Image.Image] = OrderedDict()
        self._build_ui()
        self.statusBar().showMessage('Откройте папку или изображение. Интернет не требуется.')

    def run_task(self, fn, done, *, failed=None, progress=None):
        job = Task(fn)
        self.pending.add(job)
        def cleanup():
            self.pending.discard(job)
        job.signals.result.connect(done)
        job.signals.result.connect(lambda _: cleanup())
        job.signals.error.connect(failed or self._error)
        job.signals.error.connect(lambda _: cleanup())
        if progress:
            job.signals.progress.connect(progress)
        self.threadpool.start(job)

    def _error(self, message):
        self.statusBar().showMessage(message, 9000)
        QMessageBox.warning(self, 'Не удалось выполнить операцию', message)

    def _action(self, toolbar, text, func, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(func)
        if shortcut:
            action.setShortcut(shortcut)
        toolbar.addAction(action)
        return action

    def _build_ui(self):
        toolbar = QToolBar('Инструменты')
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(Qt.ToolBarArea.BottomToolBarArea, toolbar)
        self._action(toolbar, '▣  Папка', self.choose_folder, 'Ctrl+O')
        self._action(toolbar, '＋  Файл', self.choose_file)
        self._action(toolbar, '☷  Файлы', self.toggle_library, 'Ctrl+L')
        self._action(toolbar, 'Дубликаты', self.find_duplicates, 'Ctrl+D')
        toolbar.addSeparator()
        self._action(toolbar, '←', lambda: self.move(-1), 'Left')
        self._action(toolbar, '→', lambda: self.move(1), 'Right')
        self._action(toolbar, 'Вписать', self.picture_fit, 'F')
        self._action(toolbar, '1:1', self.canvas_actual, '1')
        self._action(toolbar, '↶ 90°', lambda: self.canvas.rotate(-90), 'Shift+R')
        self._action(toolbar, '↷ 90°', lambda: self.canvas.rotate(90), 'R')
        self._action(toolbar, '⛶  Экран', self.toggle_fullscreen, 'F11')
        toolbar.addSeparator()
        self.edit_action = self._action(toolbar, '✎  Перевести страницу', self.open_editor, 'Ctrl+E')
        self.edit_action.setEnabled(False)

        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)
        left = QWidget()
        self.library_panel = left
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        heading = QLabel('БИБЛИОТЕКА')
        heading.setStyleSheet('font-weight:700; letter-spacing:2px; color:#97a9bf;')
        lv.addWidget(heading)
        self.path_label = QLabel('Папка не выбрана')
        self.path_label.setProperty('muted', True)
        self.path_label.setWordWrap(True)
        lv.addWidget(self.path_label)
        filters = QHBoxLayout()
        self.recursive = QCheckBox('Подпапки')
        self.recursive.toggled.connect(self.rescan_folder)
        filters.addWidget(self.recursive)
        filters.addStretch()
        self.sorter = QComboBox()
        self.sorter.addItems(['По имени', 'По дате', 'По размеру'])
        self.sorter.currentIndexChanged.connect(self.sort_current)
        filters.addWidget(self.sorter)
        lv.addLayout(filters)
        self.list = QListWidget()
        self.list.setIconSize(QSize(50, 50))
        self.list.setSpacing(3)
        self.list.currentRowChanged.connect(self._select_row)
        lv.addWidget(self.list, 1)
        self.dup_button = QPushButton('⌕  Найти точные дубликаты')
        self.dup_button.clicked.connect(self.find_duplicates)
        lv.addWidget(self.dup_button)
        dup_controls = QHBoxLayout()
        self.only_dups = QCheckBox('Только дубликаты')
        self.only_dups.toggled.connect(self.refresh_list)
        dup_controls.addWidget(self.only_dups)
        self.cancel_button = QPushButton('Отмена')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_duplicates)
        dup_controls.addWidget(self.cancel_button)
        lv.addLayout(dup_controls)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lv.addWidget(self.progress)
        self.dup_label = QLabel('Дубликаты ещё не искали')
        self.dup_label.setProperty('muted', True)
        self.dup_label.setWordWrap(True)
        lv.addWidget(self.dup_label)

        viewer_panel = QWidget()
        rv = QVBoxLayout(viewer_panel)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        top = QHBoxLayout()
        self.title_label = QLabel('Выберите изображение')
        self.title_label.setStyleSheet('font-size: 13px; font-weight: 500; color: #d4dce7; padding: 6px 10px;')
        self.title_label.setMinimumWidth(150)
        top.addWidget(self.title_label, 1)
        self.image_info = QLabel('')
        self.image_info.setProperty('muted', True)
        top.addWidget(self.image_info)
        rv.addLayout(top)
        self.canvas = PictureView()
        self.canvas.setFrameShape(QFrame.Shape.NoFrame)
        rv.addWidget(self.canvas, 1)
        bottom = QHBoxLayout()
        self.counter_label = QLabel('0 / 0')
        self.counter_label.setStyleSheet('font-weight:600;')
        bottom.addWidget(self.counter_label)
        bottom.addStretch(1)
        self.filmstrip = []
        for n in range(-3, 4):
            button = QPushButton('')
            button.setFixedSize(92, 74)
            button.clicked.connect(lambda _=False, offset=n: self.move(offset))
            self.filmstrip.append(button)
            bottom.addWidget(button)
        bottom.addStretch(1)
        # Win7-style viewer: no permanent filmstrip; picture occupies the canvas.
        for button in self.filmstrip: button.hide()
        self.counter_label.setText('')
        rv.addLayout(bottom)
        split.addWidget(left)
        split.addWidget(viewer_panel)
        split.setSizes([330, 1050])
        self.library_panel.hide()   # Library appears only on user request.
        self._update_buttons()

    def toggle_library(self):
        self.library_panel.setVisible(not self.library_panel.isVisible())
        if self.library_panel.isVisible():
            self.list.setFocus()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Открыть папку изображений')
        if folder:
            self.set_folder(Path(folder))

    def choose_file(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Открыть изображение', '',
            'Изображения (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff *.gif);;Все файлы (*)')
        if name:
            self.set_folder(Path(name).parent, Path(name))

    def set_folder(self, folder: Path, selected: Path | None = None):
        self.cancel_duplicates()
        self.folder_generation += 1
        generation = self.folder_generation
        self.folder = folder
        self.path_label.setText(str(folder))
        self.dup_scan = None
        self.dup_map = {}
        self.dup_label.setText('Дубликаты ещё не искали')
        self.only_dups.blockSignals(True)
        self.only_dups.setChecked(False)
        self.only_dups.blockSignals(False)
        recursive = self.recursive.isChecked()
        self.statusBar().showMessage('Сканирование папки…')
        def finished(paths):
            if generation != self.folder_generation:
                return
            self.paths = paths
            self.sort_current()
            if selected in self.visible:
                self.select_path(selected)
            elif self.visible:
                self.select_path(self.visible[0])
            else:
                self.current = None
                self.list.clear()
                self.title_label.setText('В папке нет изображений')
                self.counter_label.setText('0 / 0')
                self.edit_action.setEnabled(False)
            self.statusBar().showMessage(f'Найдено {len(paths)} изображений', 6500)
        self.run_task(lambda _: discover_images(folder, recursive), finished)

    def rescan_folder(self):
        if self.folder:
            self.set_folder(self.folder, self.current)

    def sort_current(self):
        if not self.folder:
            return
        sort_mode = ('name', 'modified', 'size')[self.sorter.currentIndex()]
        try:
            self.paths = sort_images(self.paths, self.folder, sort_mode)
        except OSError as exc:
            self._error(str(exc))
        self.refresh_list()

    def refresh_list(self):
        self.visible = [p for p in self.paths if p in self.dup_map] if self.only_dups.isChecked() else self.paths[:]
        self.list.blockSignals(True)
        self.list.clear()
        for path in self.visible:
            rel = str(path.relative_to(self.folder)) if self.folder else path.name
            item = QListWidgetItem(rel)
            item.setData(Qt.ItemDataRole.UserRole, path)
            group = self.dup_map.get(path)
            if group is not None:
                item.setBackground(QColor(self.COLOR_GROUPS[(group-1) % len(self.COLOR_GROUPS)]))
                item.setToolTip(f'Точный дубликат, группа №{group}\n{path}')
            else:
                item.setToolTip(str(path))
            self.list.addItem(item)
        if self.current in self.visible:
            self.list.setCurrentRow(self.visible.index(self.current))
        self.list.blockSignals(False)
        self._update_buttons()
        self._schedule_thumbnails()

    def _select_row(self, index):
        if 0 <= index < len(self.visible):
            self.select_path(self.visible[index])

    def select_path(self, path: Path):
        if path not in self.visible:
            if self.folder and path.is_relative_to(self.folder):
                if not self.recursive.isChecked():
                    self.recursive.blockSignals(True)
                    self.recursive.setChecked(True)
                    self.recursive.blockSignals(False)
                    self.set_folder(self.folder, path)
                    return
            self.only_dups.setChecked(False)
        if path not in self.visible:
            return
        self.current = path
        self.image_generation += 1
        generation = self.image_generation
        self.title_label.setText(path.name)
        self.image_info.setText('Загрузка…')
        self.list.blockSignals(True)
        self.list.setCurrentRow(self.visible.index(path))
        self.list.blockSignals(False)
        self.list.scrollToItem(self.list.currentItem())
        self.counter_label.setText(f'{self.visible.index(path)+1} / {len(self.visible)}')
        self.edit_action.setEnabled(True)
        self._update_buttons()
        self._schedule_thumbnails()
        def finished(image):
            if generation != self.image_generation:
                return
            self.canvas.set_picture(image)
            self.image_info.setText(f'{image.width} × {image.height} px · {path.suffix.upper()[1:]}')
        self.run_task(lambda _: open_image(str(path)), finished,
                      failed=lambda msg: self.statusBar().showMessage(f'{path.name}: {msg}', 9000))

    def _update_buttons(self):
        active = self.current in self.visible
        index = self.visible.index(self.current) if active else -1
        for offset, button in zip(range(-3, 4), self.filmstrip):
            target = index+offset
            enabled = active and 0 <= target < len(self.visible)
            button.setEnabled(enabled)
            if enabled:
                button.setIcon(QIcon())
                button.setText(self.visible[target].stem[:12])
                button.setToolTip(str(self.visible[target]))
            else:
                button.setText('')
                button.setIcon(QIcon())

    def _schedule_thumbnails(self):
        if self.current not in self.visible:
            return
        self.thumb_generation += 1
        generation = self.thumb_generation
        idx = self.visible.index(self.current)
        nearby = [p for p in self.visible[max(0, idx-15):min(len(self.visible), idx+16)]
                  if p not in self.thumbnail_cache]
        if not nearby:
            self._apply_thumbnails()
            return
        def load(_):
            output = {}
            for path in nearby:
                try:
                    with Image.open(path) as src:
                        src.seek(0)
                        img = ImageOps.exif_transpose(src).convert('RGB')
                        img.thumbnail((100, 85), Image.Resampling.LANCZOS)
                        output[path] = img.copy()
                except (OSError, ValueError):
                    pass
            return output
        def ready(images):
            if generation != self.thumb_generation:
                return
            self.thumbnail_cache.update(images)
            while len(self.thumbnail_cache) > 150:
                self.thumbnail_cache.popitem(last=False)
            self._apply_thumbnails()
        self.run_task(load, ready, failed=lambda _: None)

    def _apply_thumbnails(self):
        for index, path in enumerate(self.visible):
            if path in self.thumbnail_cache:
                item = self.list.item(index)
                if item:
                    item.setIcon(QIcon(pixmap_from_pil(self.thumbnail_cache[path])))
        if self.current in self.visible:
            idx = self.visible.index(self.current)
            for offset, button in zip(range(-3, 4), self.filmstrip):
                k = idx+offset
                if 0 <= k < len(self.visible) and self.visible[k] in self.thumbnail_cache:
                    button.setIcon(QIcon(pixmap_from_pil(self.thumbnail_cache[self.visible[k]])))
                    button.setIconSize(QSize(76, 55))

    def move(self, offset):
        if self.current in self.visible:
            index = self.visible.index(self.current) + offset
            if 0 <= index < len(self.visible):
                self.select_path(self.visible[index])

    def picture_fit(self):
        self.canvas.fit_picture()

    def canvas_actual(self):
        self.canvas.actual_size()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            event.accept()
            return
        super().keyPressEvent(event)

    def find_duplicates(self):
        if not self.folder or self.scanning:
            return
        self.duplicate_generation += 1
        generation = self.duplicate_generation
        self.cancel_event = threading.Event()
        cancel = self.cancel_event
        folder = self.folder
        self.scanning = True
        self.cancel_button.setEnabled(True)
        self.dup_button.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.dup_label.setText('Поиск во всей папке и подпапках…')
        def scan(progress):
            all_files = discover_images(folder, True)
            progress(('total', len(all_files)))
            return find_duplicates(all_files, cancelled=cancel.is_set,
                                   progress=lambda i,n,p: progress(('progress',i,n)))
        def show_progress(value):
            if generation != self.duplicate_generation:
                return
            if value[0] == 'total':
                self.progress.setRange(0, value[1])
            else:
                self.progress.setValue(value[1])
        def finish(result):
            if generation != self.duplicate_generation:
                return
            self.scanning = False
            self.cancel_button.setEnabled(False)
            self.dup_button.setEnabled(True)
            self.progress.setVisible(False)
            if result.cancelled:
                self.dup_label.setText('Поиск отменён')
                return
            self.dup_scan = result
            self.dup_map = result.path_to_group()
            self.dup_label.setText(f'Групп: {len(result.groups)} · Файлов-дубликатов: {len(self.dup_map)}')
            # Reveal matches across subfolders, not just the top level.
            if not self.recursive.isChecked():
                self.recursive.blockSignals(True)
                self.recursive.setChecked(True)
                self.recursive.blockSignals(False)
                self.folder_generation += 1
                fg = self.folder_generation
                def collected(paths):
                    if fg != self.folder_generation:
                        return
                    self.paths = paths
                    self.sort_current()
                    if self.current in self.visible:
                        self.select_path(self.current)
                self.run_task(lambda _: discover_images(folder, True), collected)
            else:
                self.refresh_list()
            # Keep dialog alive as modeless (no nested event-loop).
            self.dialog = DuplicateDialog(result, self.select_path, self)
            self.dialog.show()
        def fail(message):
            if generation != self.duplicate_generation:
                return
            self.scanning = False
            self.cancel_button.setEnabled(False)
            self.dup_button.setEnabled(True)
            self.progress.setVisible(False)
            self._error(message)
        self.run_task(scan, finish, failed=fail, progress=show_progress)

    def cancel_duplicates(self):
        self.cancel_event.set()
        self.duplicate_generation += 1
        self.scanning = False
        if hasattr(self, 'cancel_button'):
            self.cancel_button.setEnabled(False)
            self.dup_button.setEnabled(True)
            self.progress.setVisible(False)

    def open_editor(self):
        if not self.current:
            return
        from qt_editor import EditorWindow
        editor = EditorWindow(self.current)
        editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        editor.destroyed.connect(lambda: self.editor_windows.remove(editor) if editor in self.editor_windows else None)
        self.editor_windows.append(editor)
        editor.show()