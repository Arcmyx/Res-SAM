# -*- coding: utf-8 -*-

from PySide6.QtWidgets import QMainWindow, QFileDialog, QLabel, QWidget, QVBoxLayout, QMessageBox
from PySide6.QtGui import QFontMetrics, QPixmap, QPainter, QImage, QPen, QColor, QDragEnterEvent, QDropEvent
from PySide6.QtCore import Qt, QObject, QEvent, QThread, Signal, Slot, QTimer
import numpy as np
from io import BytesIO
import cv2
import torch
from PIL import ImageQt, Image
from ui.ui_main import Ui_MainWindow
from sam.sam import *
from PatchRes.PatchRes import PatchRes
from ui.i18n import i18n


class SamPoints(object):
    def __init__(self):
        self.points = [[], []]
        self.labels = []

    def clear(self):
        self.points = [[], []]
        self.labels.clear()

    def add_point(self, x, y, label):
        self.points[label].append([x, y])
        self.labels.append(label)

    def undo(self):
        if self.labels:
            self.points[self.labels.pop()].pop()

    def get_prompt(self):
        point_coords = self.points[0] + self.points[1]
        point_labels = [0] * len(self.points[0]) + [1] * len(self.points[1])
        return np.array(point_coords), np.array(point_labels)

    def __bool__(self):
        return bool(self.labels)


class Worker(QThread):
    finished = Signal(object)

    def __init__(self, image, point_coords, point_labels):
        super().__init__()
        self.image = image
        self.point_coords = point_coords
        self.point_labels = point_labels

    def run(self):
        mask = predict(self.image, self.point_coords, self.point_labels)
        self.finished.emit([self.image, mask])

class HeatmapWorker(QObject):
    finished = Signal(object, int, int)   # mask, x0, y0
    failed = Signal(str)

    def __init__(self, get_image_fn, get_mask_fn):
        super().__init__()
        self.get_image = get_image_fn
        self.get_mask = get_mask_fn

    @Slot()
    def run(self):
        try:
            image = self.get_image()
            x0, y0 = 0, 0
            x1, y1 = image.shape[1], image.shape[0]
            mask = self.get_mask(image, x0, y0, x1, y1)
            self.finished.emit(mask, x0, y0)
        except Exception:
            ...


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.setStyleSheet("background-color: rgba(0, 0, 0, 150); border-radius: 10px;")
        
        label = QLabel(i18n("正在绘制，请稍候..."), self)
        label.setStyleSheet("color: white; font-size: 16px;")
        label.setAlignment(Qt.AlignCenter)

        self.label = label
        
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(layout)

    def show_centered_in(self, parent_widget, timeout=3000):
        font_metrics = QFontMetrics(self.label.font())
        text_width = font_metrics.horizontalAdvance(self.label.text())

        parent_rect = parent_widget.geometry()
        overlay_width = text_width + 200
        overlay_height = 100
        x = parent_rect.x() + (parent_rect.width() - overlay_width) // 2
        y = parent_rect.y() + (parent_rect.height() - overlay_height) // 2

        self.setGeometry(x, y, overlay_width, overlay_height)
        self.show()
        QTimer.singleShot(timeout, self.close)

