"""
LDCT Denoising - Comparison Widget Module
Yan yana ve slider karşılaştırma görünümleri.
"""

from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                              QSlider, QStackedWidget, QFrame)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QFont, QColor
import numpy as np


class ImageLabel(QLabel):
    """Görüntü gösteren özel label widget"""
    
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 12px;
                padding: 8px;
            }
        """)
        self.setMinimumSize(280, 280)
    
    def set_image(self, image: np.ndarray):
        """Numpy array'den görüntü ayarla"""
        if image is None:
            return
        
        h, w = image.shape[:2]
        
        if len(image.shape) == 2:
            # Grayscale
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:
            # RGB
            qimg = QImage(image.data, w, h, w * 3, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, 
                               Qt.SmoothTransformation)
        self.setPixmap(scaled)
    
    def resizeEvent(self, event):
        """Yeniden boyutlandırıldığında görüntüyü güncelle"""
        super().resizeEvent(event)
        if self.pixmap() and not self.pixmap().isNull():
            # Mevcut pixmap'i yeniden ölçekle
            pass


class SideBySideWidget(QWidget):
    """Yan yana karşılaştırma görünümü"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Sol taraf - Low Dose
        left_container = QVBoxLayout()
        self.left_title = QLabel("📉 Low Dose (Orijinal)")
        self.left_title.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-size: 16px;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.left_title.setAlignment(Qt.AlignCenter)
        
        self.left_image = ImageLabel("Low Dose")
        
        left_container.addWidget(self.left_title)
        left_container.addWidget(self.left_image, 1)
        
        # Sağ taraf - Enhanced
        right_container = QVBoxLayout()
        self.right_title = QLabel("✨ AI Enhanced (Full Dose)")
        self.right_title.setStyleSheet("""
            QLabel {
                color: #4ecdc4;
                font-size: 16px;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.right_title.setAlignment(Qt.AlignCenter)
        
        self.right_image = ImageLabel("Enhanced")
        
        right_container.addWidget(self.right_title)
        right_container.addWidget(self.right_image, 1)
        
        layout.addLayout(left_container, 1)
        layout.addLayout(right_container, 1)
    
    def set_images(self, low_dose: np.ndarray, enhanced: np.ndarray):
        """Her iki görüntüyü ayarla"""
        self.left_image.set_image(low_dose)
        self.right_image.set_image(enhanced)


class SliderComparisonWidget(QWidget):
    """Sürüklemeli slider karşılaştırma görünümü (Remini tarzı)"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.low_dose_img = None
        self.enhanced_img = None
        self.slider_position = 0.5  # 0.0 - 1.0 arası
        self.dragging = False
        
        self.setMinimumSize(400, 400)
        self.setMouseTracking(True)
        self.setCursor(Qt.SplitHCursor)
        
        self.setStyleSheet("""
            SliderComparisonWidget {
                background-color: #1a1a2e;
                border: 2px solid #16213e;
                border-radius: 12px;
            }
        """)
    
    def set_images(self, low_dose: np.ndarray, enhanced: np.ndarray):
        """Her iki görüntüyü ayarla"""
        self.low_dose_img = low_dose
        self.enhanced_img = enhanced
        self.update()
    
    def paintEvent(self, event):
        """Görüntüleri çiz"""
        super().paintEvent(event)
        
        if self.low_dose_img is None or self.enhanced_img is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Widget boyutları
        w, h = self.width(), self.height()
        padding = 20
        img_area_w = w - 2 * padding
        img_area_h = h - 2 * padding - 40  # Alt label için alan
        
        # Görüntüleri QImage'e dönüştür
        low_h, low_w = self.low_dose_img.shape[:2]
        enh_h, enh_w = self.enhanced_img.shape[:2]
        
        low_qimg = QImage(self.low_dose_img.data, low_w, low_h, low_w, 
                          QImage.Format_Grayscale8)
        enh_qimg = QImage(self.enhanced_img.data, enh_w, enh_h, enh_w, 
                          QImage.Format_Grayscale8)
        
        # Görüntü boyutunu hesapla (aspect ratio koru)
        img_size = min(img_area_w, img_area_h)
        
        low_pixmap = QPixmap.fromImage(low_qimg).scaled(
            img_size, img_size, 
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        enh_pixmap = QPixmap.fromImage(enh_qimg).scaled(
            img_size, img_size,
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Görüntü konumu (ortala)
        img_x = padding + (img_area_w - img_size) // 2
        img_y = padding + (img_area_h - img_size) // 2
        
        # Slider pozisyonu (piksel cinsinden)
        slider_x = int(img_x + img_size * self.slider_position)
        
        # Enhanced görüntüyü çiz (tam)
        painter.drawPixmap(img_x, img_y, enh_pixmap)
        
        # Low dose görüntüyü sol tarafa clip'le
        painter.setClipRect(img_x, img_y, slider_x - img_x, img_size)
        painter.drawPixmap(img_x, img_y, low_pixmap)
        painter.setClipping(False)
        
        # Slider çizgisi
        painter.setPen(QColor(255, 255, 255))
        painter.drawLine(slider_x, img_y, slider_x, img_y + img_size)
        
        # Slider handle (daire)
        handle_y = img_y + img_size // 2
        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(QPoint(slider_x, handle_y), 12, 12)
        
        # İç daire
        painter.setBrush(QColor(78, 205, 196))
        painter.drawEllipse(QPoint(slider_x, handle_y), 8, 8)
        
        # Etiketler
        font = QFont("Arial", 11, QFont.Bold)
        painter.setFont(font)
        
        # Sol etiket (Low Dose)
        painter.setPen(QColor(255, 107, 107))
        painter.drawText(img_x + 10, img_y + 25, "Low Dose")
        
        # Sağ etiket (Enhanced)
        painter.setPen(QColor(78, 205, 196))
        enh_text = "AI Enhanced"
        painter.drawText(img_x + img_size - 90, img_y + 25, enh_text)
        
        # Alt bilgi
        painter.setPen(QColor(150, 150, 150))
        font = QFont("Arial", 10)
        painter.setFont(font)
        info_text = "◀ Sürükleyerek karşılaştırın ▶"
        text_width = painter.fontMetrics().horizontalAdvance(info_text)
        painter.drawText((w - text_width) // 2, h - 15, info_text)
    
    def mousePressEvent(self, event):
        """Mouse basıldığında"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.update_slider_position(event.pos())
    
    def mouseMoveEvent(self, event):
        """Mouse hareket ettiğinde"""
        if self.dragging:
            self.update_slider_position(event.pos())
    
    def mouseReleaseEvent(self, event):
        """Mouse bırakıldığında"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
    
    def update_slider_position(self, pos):
        """Slider pozisyonunu güncelle"""
        padding = 20
        img_area_w = self.width() - 2 * padding
        img_area_h = self.height() - 2 * padding - 40
        img_size = min(img_area_w, img_area_h)
        img_x = padding + (img_area_w - img_size) // 2
        
        # Pozisyonu 0-1 arasına normalize et
        relative_x = pos.x() - img_x
        self.slider_position = max(0.0, min(1.0, relative_x / img_size))
        self.update()


class ComparisonContainer(QWidget):
    """İki görünüm modunu içeren container widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Stacked widget (mod değiştirme için)
        self.stack = QStackedWidget()
        
        # Yan yana görünüm
        self.side_by_side = SideBySideWidget()
        self.stack.addWidget(self.side_by_side)
        
        # Slider görünüm
        self.slider_view = SliderComparisonWidget()
        self.stack.addWidget(self.slider_view)
        
        layout.addWidget(self.stack)
    
    def set_mode(self, mode: str):
        """Görünüm modunu değiştir: 'side_by_side' veya 'slider'"""
        if mode == "side_by_side":
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)
    
    def set_images(self, low_dose: np.ndarray, enhanced: np.ndarray):
        """Her iki görünüme de görüntüleri ayarla"""
        self.side_by_side.set_images(low_dose, enhanced)
        self.slider_view.set_images(low_dose, enhanced)
