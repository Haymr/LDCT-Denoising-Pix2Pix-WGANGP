"""
LDCT Denoising - Main Application
Modern PyQt5 masaüstü uygulaması.
"""

import sys
import os

# Uygulama dizinini path'e ekle
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QPushButton, QFrame,
                              QFileDialog, QMessageBox, QButtonGroup, QRadioButton,
                              QProgressBar, QSizePolicy, QStackedWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMimeData, QPoint
from PyQt5.QtGui import QFont, QDragEnterEvent, QDropEvent, QIcon, QPalette, QColor

import numpy as np

from preprocessing import preprocess_dicom, postprocess_output, model_input_to_display
from model import load_model, predict, get_model
from comparison_widget import ComparisonContainer


# Proje kök dizini ve model yolu
PROJECT_ROOT = os.path.dirname(app_dir)
MODEL_PATH = os.path.join(PROJECT_ROOT, "G_epoch_50.h5")


class ProcessingThread(QThread):
    """Arka planda işleme yapan thread"""
    finished = pyqtSignal(np.ndarray, np.ndarray)  # low_dose, enhanced
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            # 1. Preprocessing
            self.progress.emit(30)
            model_input, _ = preprocess_dicom(self.file_path)
            
            # 2. Display için low dose görüntü
            low_dose_display = model_input_to_display(model_input)
            
            # 3. Model inference
            self.progress.emit(60)
            model_output = predict(model_input)
            
            # 4. Post-processing
            self.progress.emit(90)
            enhanced_display = postprocess_output(model_output)
            
            self.progress.emit(100)
            self.finished.emit(low_dose_display, enhanced_display)
            
        except Exception as e:
            self.error.emit(str(e))


class DropZone(QFrame):
    """Sürükle-bırak dosya yükleme alanı"""
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setup_ui()
        
    def setup_ui(self):
        self.setMinimumHeight(180)
        self.setStyleSheet("""
            DropZone {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(26, 26, 46, 0.8), stop:1 rgba(22, 33, 62, 0.8));
                border: 3px dashed #4a5568;
                border-radius: 16px;
            }
            DropZone:hover {
                border-color: #4ecdc4;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(78, 205, 196, 0.1), stop:1 rgba(26, 26, 46, 0.8));
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)
        
        # İkon
        icon_label = QLabel("📁")
        icon_label.setStyleSheet("font-size: 48px; background: transparent; border: none;")
        icon_label.setAlignment(Qt.AlignCenter)
        
        # Ana metin
        main_text = QLabel("DICOM Dosyasını Sürükleyip Bırakın")
        main_text.setStyleSheet("""
            QLabel {
                color: #e2e8f0;
                font-size: 18px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        main_text.setAlignment(Qt.AlignCenter)
        
        # Alt metin
        sub_text = QLabel("veya dosya seçmek için tıklayın")
        sub_text.setStyleSheet("""
            QLabel {
                color: #718096;
                font-size: 13px;
                background: transparent;
                border: none;
            }
        """)
        sub_text.setAlignment(Qt.AlignCenter)
        
        # Format bilgisi
        format_text = QLabel("📋 Sadece .dcm formatı kabul edilir")
        format_text.setStyleSheet("""
            QLabel {
                color: #4ecdc4;
                font-size: 12px;
                background: transparent;
                border: none;
                padding-top: 8px;
            }
        """)
        format_text.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(icon_label)
        layout.addWidget(main_text)
        layout.addWidget(sub_text)
        layout.addWidget(format_text)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            if url.toLocalFile().lower().endswith('.dcm'):
                event.acceptProposedAction()
                self.setStyleSheet("""
                    DropZone {
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 rgba(78, 205, 196, 0.2), stop:1 rgba(26, 26, 46, 0.9));
                        border: 3px solid #4ecdc4;
                        border-radius: 16px;
                    }
                """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            DropZone {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(26, 26, 46, 0.8), stop:1 rgba(22, 33, 62, 0.8));
                border: 3px dashed #4a5568;
                border-radius: 16px;
            }
            DropZone:hover {
                border-color: #4ecdc4;
            }
        """)
    
    def dropEvent(self, event: QDropEvent):
        self.dragLeaveEvent(None)
        
        url = event.mimeData().urls()[0]
        file_path = url.toLocalFile()
        
        if file_path.lower().endswith('.dcm'):
            self.file_dropped.emit(file_path)
        else:
            QMessageBox.warning(self, "Hatalı Format", 
                "Sadece DICOM (.dcm) dosyaları kabul edilir!")
    
    def mousePressEvent(self, event):
        """Tıklandığında dosya seçici aç"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "DICOM Dosyası Seç", "", "DICOM Files (*.dcm);;All Files (*)"
        )
        if file_path:
            self.file_dropped.emit(file_path)


class MainWindow(QMainWindow):
    """Ana uygulama penceresi"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LDCT Denoising - AI Görüntü İyileştirme")
        self.setMinimumSize(900, 700)
        self.setup_ui()
        self.load_model_on_start()
        
    def setup_ui(self):
        # Ana widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Arka plan stili
        central_widget.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0f0f23, stop:0.5 #1a1a2e, stop:1 #16213e);
                color: #e2e8f0;
            }
        """)
        
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Başlık
        header = self.create_header()
        layout.addWidget(header)
        
        # Drop zone
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.process_file)
        layout.addWidget(self.drop_zone)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a2e;
                border: 1px solid #4a5568;
                border-radius: 8px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4ecdc4, stop:1 #44a08d);
                border-radius: 7px;
            }
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Görünüm modu seçici
        mode_container = self.create_mode_selector()
        layout.addWidget(mode_container)
        
        # Karşılaştırma görünümü
        self.comparison = ComparisonContainer()
        self.comparison.setVisible(False)
        layout.addWidget(self.comparison, 1)
        
        # Alt bilgi
        footer = self.create_footer()
        layout.addWidget(footer)
    
    def create_header(self) -> QWidget:
        """Başlık bölümü"""
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 12)
        
        # Logo ve başlık
        title = QLabel("🏥 LDCT Denoising")
        title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 28px;
                font-weight: bold;
                background: transparent;
            }
        """)
        
        # Alt başlık
        subtitle = QLabel("Pix2Pix + WGAN-GP ile AI Tabanlı Görüntü İyileştirme")
        subtitle.setStyleSheet("""
            QLabel {
                color: #4ecdc4;
                font-size: 14px;
                background: transparent;
            }
        """)
        
        title_layout = QVBoxLayout()
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        title_layout.setSpacing(4)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Model durumu
        self.model_status = QLabel("⏳ Model yükleniyor...")
        self.model_status.setStyleSheet("""
            QLabel {
                color: #fbbf24;
                font-size: 12px;
                background: rgba(251, 191, 36, 0.1);
                padding: 6px 12px;
                border-radius: 12px;
                border: 1px solid #fbbf24;
            }
        """)
        layout.addWidget(self.model_status)
        
        return header
    
    def create_mode_selector(self) -> QWidget:
        """Görünüm modu seçici"""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 8, 0, 8)
        
        label = QLabel("Görünüm Modu:")
        label.setStyleSheet("color: #a0aec0; font-size: 13px; background: transparent;")
        layout.addWidget(label)
        
        # Yan yana butonu
        self.btn_side_by_side = QPushButton("📊 Yan Yana")
        self.btn_side_by_side.setCheckable(True)
        self.btn_side_by_side.setChecked(True)
        self.btn_side_by_side.clicked.connect(lambda: self.set_view_mode("side_by_side"))
        
        # Slider butonu
        self.btn_slider = QPushButton("🔀 Sürüklemeli")
        self.btn_slider.setCheckable(True)
        self.btn_slider.clicked.connect(lambda: self.set_view_mode("slider"))
        
        button_style = """
            QPushButton {
                background: rgba(74, 85, 104, 0.3);
                color: #e2e8f0;
                border: 1px solid #4a5568;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: rgba(78, 205, 196, 0.2);
                border-color: #4ecdc4;
            }
            QPushButton:checked {
                background: rgba(78, 205, 196, 0.3);
                border-color: #4ecdc4;
                color: #4ecdc4;
            }
        """
        self.btn_side_by_side.setStyleSheet(button_style)
        self.btn_slider.setStyleSheet(button_style)
        
        layout.addWidget(self.btn_side_by_side)
        layout.addWidget(self.btn_slider)
        layout.addStretch()
        
        return container
    
    def create_footer(self) -> QWidget:
        """Alt bilgi"""
        footer = QLabel("Tasarım Dersi Projesi • Pix2Pix + WGAN-GP Model")
        footer.setStyleSheet("""
            QLabel {
                color: #4a5568;
                font-size: 11px;
                background: transparent;
                padding: 8px;
            }
        """)
        footer.setAlignment(Qt.AlignCenter)
        return footer
    
    def load_model_on_start(self):
        """Başlangıçta modeli yükle"""
        try:
            if os.path.exists(MODEL_PATH):
                success = load_model(MODEL_PATH)
                if success:
                    self.model_status.setText("✅ Model Hazır")
                    self.model_status.setStyleSheet("""
                        QLabel {
                            color: #4ecdc4;
                            font-size: 12px;
                            background: rgba(78, 205, 196, 0.1);
                            padding: 6px 12px;
                            border-radius: 12px;
                            border: 1px solid #4ecdc4;
                        }
                    """)
                else:
                    self.show_model_error("Model yüklenemedi")
            else:
                self.show_model_error(f"Model dosyası bulunamadı:\n{MODEL_PATH}")
        except Exception as e:
            self.show_model_error(str(e))
    
    def show_model_error(self, message: str):
        """Model hata durumu göster"""
        self.model_status.setText("❌ Model Hatası")
        self.model_status.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-size: 12px;
                background: rgba(255, 107, 107, 0.1);
                padding: 6px 12px;
                border-radius: 12px;
                border: 1px solid #ff6b6b;
            }
        """)
        QMessageBox.critical(self, "Model Hatası", message)
    
    def set_view_mode(self, mode: str):
        """Görünüm modunu değiştir"""
        self.btn_side_by_side.setChecked(mode == "side_by_side")
        self.btn_slider.setChecked(mode == "slider")
        self.comparison.set_mode(mode)
    
    def process_file(self, file_path: str):
        """DICOM dosyasını işle"""
        # Model kontrolü
        model = get_model()
        if not model.is_loaded:
            QMessageBox.warning(self, "Uyarı", "Model henüz yüklenmedi!")
            return
        
        # Progress bar göster
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Drop zone'u gizle
        self.drop_zone.setMinimumHeight(80)
        
        # Thread başlat
        self.processing_thread = ProcessingThread(file_path)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.start()
    
    def on_processing_finished(self, low_dose: np.ndarray, enhanced: np.ndarray):
        """İşleme tamamlandığında"""
        self.progress_bar.setVisible(False)
        self.comparison.setVisible(True)
        self.comparison.set_images(low_dose, enhanced)
    
    def on_processing_error(self, error_message: str):
        """Hata durumunda"""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "İşleme Hatası", error_message)


def main():
    app = QApplication(sys.argv)
    
    # Uygulama stili
    app.setStyle("Fusion")
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(15, 15, 35))
    palette.setColor(QPalette.WindowText, QColor(226, 232, 240))
    palette.setColor(QPalette.Base, QColor(26, 26, 46))
    palette.setColor(QPalette.Text, QColor(226, 232, 240))
    palette.setColor(QPalette.Button, QColor(74, 85, 104))
    palette.setColor(QPalette.ButtonText, QColor(226, 232, 240))
    palette.setColor(QPalette.Highlight, QColor(78, 205, 196))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
