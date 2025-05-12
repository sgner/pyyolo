import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QCheckBox, QPushButton, QFileDialog,
    QListWidget, QTextEdit, QStatusBar, QProgressDialog, QListWidgetItem,
    QGroupBox, QTabWidget, QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, QTimer, QSize, QMimeData, QThread, pyqtSignal, QObject, QUrl
from PyQt5.QtGui import QImage, QPixmap, QIcon, QDragEnterEvent, QDropEvent, QColor
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import os
from monitor.system_monitor import SystemMonitor

# 加载配置文件
def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for model in config["models"].values():
            model["path"] = str(Path(model["path"])).replace("\\", "/")
        return config
    except Exception as e:
        raise RuntimeError(f"配置加载失败: {str(e)}")

CONFIG = load_config()

# 模型管理线程工作类
class ModelManagerWorker(QObject):
    model_download_progress = pyqtSignal(int, int)  # 当前字节数，总字节数
    model_load_started = pyqtSignal()
    model_loaded = pyqtSignal(str)  # 模型名称
    model_error = pyqtSignal(str)  # 错误信息

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.current_model = None
        self.current_model_name = None
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.download_finished)
        self.downloading_model_name = None
        self.reply = None

    def ensure_model_loaded(self, model_name):
        if self.current_model_name == model_name:
            self.model_loaded.emit(model_name)
        elif Path(self.models[model_name]["path"]).exists():
            self.load_model(model_name)
        else:
            self.downloading_model_name = model_name
            self.download_model(model_name)

    def download_model(self, model_name):
        model_cfg = self.models[model_name]
        url = model_cfg["url"]
        path = model_cfg["path"]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        request = QNetworkRequest(QUrl(url))
        self.reply = self.network_manager.get(request)
        self.reply.downloadProgress.connect(self.on_download_progress)

    def on_download_progress(self, bytes_received, bytes_total):
        self.model_download_progress.emit(bytes_received, bytes_total)

    def download_finished(self, reply):
        if reply.error():
            self.model_error.emit(f"下载失败: {reply.errorString()}")
            return
        model_name = self.downloading_model_name
        model_cfg = self.models[model_name]
        path = model_cfg["path"]
        with open(path, 'wb') as f:
            f.write(reply.readAll())
        self.load_model(model_name)

    def load_model(self, model_name):
        self.model_load_started.emit()
        model_cfg = self.models[model_name]
        path = model_cfg["path"]
        try:
            self.current_model = YOLO(path)
            self.current_model_name = model_name
            self.model_loaded.emit(model_name)
        except Exception as e:
            self.model_error.emit(f"模型加载失败: {str(e)}")

    def get_model(self, model_name):
        if self.current_model_name == model_name:
            return self.current_model
        return None

# 图像处理线程工作类
class ImageProcessorWorker(QObject):
    processing_finished = pyqtSignal(dict, np.ndarray, str)  # 分类结果，可视化图像，图像路径
    processing_error = pyqtSignal(str)  # 错误信息
    batch_progress = pyqtSignal(int, int)  # 当前进度，总数

    def __init__(self, model_manager_worker):
        super().__init__()
        self.model_manager_worker = model_manager_worker

    def process_image(self, img, model_name, confidence_threshold, enable_visualization, img_path=""):
        self.img = img
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.enable_visualization = enable_visualization
        self.img_path = img_path
        if self.model_manager_worker.current_model_name == model_name:
            self.do_process()
        else:
            self.model_manager_worker.model_loaded.connect(self.on_model_loaded)
            self.model_manager_worker.model_error.connect(self.on_model_error)
            self.model_manager_worker.ensure_model_loaded(model_name)

    def process_batch(self, img_paths, model_name, confidence_threshold, enable_visualization):
        self.batch_img_paths = img_paths
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.enable_visualization = enable_visualization
        if self.model_manager_worker.current_model_name == model_name:
            self.do_batch_process()
        else:
            self.model_manager_worker.model_loaded.connect(self.on_model_loaded)
            self.model_manager_worker.model_error.connect(self.on_model_error)
            self.model_manager_worker.ensure_model_loaded(model_name)

    def on_model_loaded(self, loaded_model_name):
        if loaded_model_name == self.model_name:
            if hasattr(self, 'batch_img_paths'):
                self.do_batch_process()
            else:
                self.do_process()
            self.model_manager_worker.model_loaded.disconnect(self.on_model_loaded)
            self.model_manager_worker.model_error.disconnect(self.on_model_error)

    def on_model_error(self, error):
        self.processing_error.emit(error)
        self.model_manager_worker.model_loaded.disconnect(self.on_model_loaded)
        self.model_manager_worker.model_error.disconnect(self.on_model_error)

    def do_process(self):
        try:
            model = self.model_manager_worker.get_model(self.model_name)
            if model is None:
                raise RuntimeError("模型未加载")
            # 预处理图像
            h, w = self.img.shape[:2]
            scale = 640 / max(h, w)
            processed_img = cv2.resize(self.img, (int(w * scale), int(h * scale)))
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            # 模型推理
            results = model(processed_img)
            probs = results[0].probs.data.tolist()
            all_results = {model.names[i]: float(probs[i]) for i in range(len(probs))}
            # 过滤结果
            filtered_results = {k: v for k, v in all_results.items() if v >= self.confidence_threshold}
            if not filtered_results:
                max_key = max(all_results, key=all_results.get)
                filtered_results = {max_key: all_results[max_key]}
            sorted_results = dict(sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)[:5])
            # 生成可视化
            visualization = None
            if self.enable_visualization:
                try:
                    vis = results[0].plot()
                    visualization = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                except:
                    visualization = np.zeros((640, 640, 3), dtype=np.uint8)
                    cv2.putText(visualization, "可视化失败", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            self.processing_finished.emit(sorted_results, visualization, self.img_path)
        except Exception as e:
            self.processing_error.emit(str(e))

    def do_batch_process(self):
        try:
            for i, img_path in enumerate(self.batch_img_paths):
                img = cv2.imread(img_path)
                if img is None:
                    continue
                self.img = img
                self.img_path = img_path
                self.do_process()
                self.batch_progress.emit(i + 1, len(self.batch_img_paths))
        except Exception as e:
            self.processing_error.emit(str(e))

# 自定义标题栏
class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setFixedHeight(30)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.title_label = QLabel("AI图像分类系统", self)
        self.title_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self.title_label, 1)

        self.minimize_btn = QPushButton("_", self)
        self.minimize_btn.setFixedSize(30, 30)
        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        self.minimize_btn.setStyleSheet("background: none; color: white;")

        self.maximize_btn = QPushButton("🗖", self)
        self.maximize_btn.setFixedSize(30, 30)
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        self.maximize_btn.setStyleSheet("background: none; color: white;")

        self.close_btn = QPushButton("✕", self)
        self.close_btn.setFixedSize(30, 30)
        self.close_btn.clicked.connect(self.parent.close)
        self.close_btn.setStyleSheet("background: none; color: white;")

        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)

        self.setStyleSheet("""
            QWidget {
                background-color: #333;
            }
            QLabel {
                color: white;
                font-weight: bold;
            }
            QPushButton {
                background: none;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background: #555;
            }
        """)

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_btn.setText("🗖")
        else:
            self.parent.showMaximized()
            self.maximize_btn.setText("🗗")

    def mousePressEvent(self, event):
        self.start_pos = self.parent.pos()
        self.start_mouse = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = event.globalPos() - self.start_mouse
            self.parent.move(self.start_pos + delta)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        self.toggle_maximize()

# 系统监控线程工作类
class MonitorWorker(QObject):
    monitor_data_ready = pyqtSignal(dict, str)

    def __init__(self, system_monitor):
        super().__init__()
        self.system_monitor = system_monitor
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_monitor)
        # Timer is not started here; it will be controlled by signals

    def start_monitoring(self):
        if not self.timer.isActive():
            self.timer.start(1000)

    def stop_monitoring(self):
        if self.timer.isActive():
            self.timer.stop()

    def update_monitor(self):
        self.system_monitor.update_hardware_info()
        monitor_data = self.system_monitor.get_monitor_data()
        text_info = self.system_monitor.get_text_info()
        self.monitor_data_ready.emit(monitor_data, text_info)

# 放大图像对话框
class ImageZoomDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像预览")
        self.setMinimumSize(800, 600)
        self.image_path = image_path
        self.scale_factor = 1.0
        self.initUI()
        self.load_image()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        self.setStyleSheet("""
            QDialog {
                background-color: #252526;
            }
            QScrollArea {
                background-color: #252526;
                border: 1px solid #3C3C3C;
            }
        """)

    def load_image(self):
        img = cv2.imread(self.image_path)
        if img is None:
            self.image_label.setText("无法加载图像")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update_image()

    def update_image(self):
        if not hasattr(self, 'original_pixmap'):
            return
        scaled_pixmap = self.original_pixmap.scaled(
            self.original_pixmap.size() * self.scale_factor,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()

    def wheelEvent(self, event):
        angle = event.angleDelta().y()
        if angle > 0:
            self.scale_factor *= 1.1  # Zoom in
        elif angle < 0:
            self.scale_factor /= 1.1  # Zoom out
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))  # Limit zoom range
        self.update_image()

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowTitle(CONFIG.get("title", "AI图像分类系统"))
        self.setGeometry(100, 100, 1440, 900)
        self.system_monitor = SystemMonitor()
        self.monitor_worker = MonitorWorker(self.system_monitor)
        self.monitor_thread = QThread()
        self.monitor_worker.moveToThread(self.monitor_thread)
        self.monitor_thread.start()
        self.monitor_worker.monitor_data_ready.connect(self.on_monitor_data_ready)

        # 设置模型管理线程
        self.model_thread = QThread()
        self.model_worker = ModelManagerWorker(CONFIG["models"])
        self.model_worker.moveToThread(self.model_thread)
        self.model_thread.start()
        self.model_worker.model_download_progress.connect(self.update_download_progress)
        self.model_worker.model_load_started.connect(self.on_model_load_started)
        self.model_worker.model_loaded.connect(self.on_model_loaded)
        self.model_worker.model_error.connect(self.on_model_error)

        # 设置图像处理线程
        self.process_thread = QThread()
        self.process_worker = ImageProcessorWorker(self.model_worker)
        self.process_worker.moveToThread(self.process_thread)
        self.process_thread.start()
        self.process_worker.processing_finished.connect(self.on_processing_finished)
        self.process_worker.processing_error.connect(self.on_processing_error)
        self.process_worker.batch_progress.connect(self.update_batch_progress)

        self.setup_ui()
        self.setAcceptDrops(True)
        self.current_image = None
        self.batch_files = []
        self.batch_results = []

        # 加载默认模型
        self.model_worker.ensure_model_loaded(self.model_selector.currentText())
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)

        # Connect tab change signal
        self.main_tabs.currentChanged.connect(self.on_tab_changed)

    def setup_ui(self):
        self.title_bar = TitleBar(self)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.title_bar)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(10, 10, 10, 10)
        content_layout.setSpacing(15)

        control_panel = QFrame()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(15)

        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)
        self.model_selector = QComboBox()
        self.model_selector.addItems(CONFIG["models"].keys())
        self.model_selector.setCursor(Qt.PointingHandCursor)
        model_layout.addWidget(QLabel("选择模型版本:"))
        model_layout.addWidget(self.model_selector)
        control_layout.addWidget(model_group)

        param_group = QGroupBox("处理参数")
        param_layout = QVBoxLayout(param_group)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_label = QLabel("置信度阈值: 50%")
        self.confidence_slider.valueChanged.connect(lambda v: self.confidence_label.setText(f"置信度阈值: {v}%"))
        self.visual_toggle = QCheckBox("启用可视化")
        self.visual_toggle.setChecked(True)
        param_layout.addWidget(self.confidence_label)
        param_layout.addWidget(self.confidence_slider)
        param_layout.addWidget(self.visual_toggle)
        control_layout.addWidget(param_group)

        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout(btn_group)
        self.btn_upload = QPushButton("📤 上传单图")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_batch = QPushButton("📁 批量上传")
        self.btn_batch.setCursor(Qt.PointingHandCursor)
        self.btn_batch.clicked.connect(self.batch_upload)
        self.btn_process = QPushButton("🚀 开始处理")
        self.btn_process.setCursor(Qt.PointingHandCursor)
        self.btn_process.clicked.connect(self.process_image)
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addWidget(self.btn_batch)
        btn_layout.addWidget(self.btn_process)
        control_layout.addWidget(btn_group)

        content_layout.addWidget(control_panel)

        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.West)
        content_layout.addWidget(self.main_tabs)

        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)
        self.drop_area = QLabel("拖放图像文件到这里或点击上传")
        self.drop_area.setObjectName("drop_area")
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setStyleSheet("""
            QLabel#drop_area {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                border-radius: 5px;
                color: #888;
                font-size: 16px;
                padding: 10px;
            }
            QLabel#drop_area:hover {
                border-color: #007ACC;
                color: #007ACC;
            }
        """)
        self.drop_area.setCursor(Qt.PointingHandCursor)
        self.drop_area.mousePressEvent = lambda e: self.upload_image()
        single_layout.addWidget(self.drop_area)

        result_group = QGroupBox("处理结果")
        result_layout = QHBoxLayout(result_group)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["Class", "Score"])
        self.result_table.setRowCount(0)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setStyleSheet("""
            QTableWidget {
                background-color: #252526;
                alternate-background-color: #333333;
                color: #DCDCDC;
                gridline-color: #404040;
            }
            QHeaderView::section {
                background-color: #333;
                color: #DCDCDC;
            }
        """)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.visual_label = QLabel()
        self.visual_label.setObjectName("visual_label")
        self.visual_label.setAlignment(Qt.AlignCenter)
        self.visual_label.setMinimumSize(400, 400)
        self.visual_label.setStyleSheet("""
            QLabel#visual_label {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                border-radius: 5px;
            }
        """)
        result_layout.addWidget(self.result_table, 1)
        result_layout.addWidget(self.visual_label)
        single_layout.addWidget(result_group)
        self.main_tabs.addTab(single_tab, "单图模式")

        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)

        # Thumbnail list
        thumbnail_group = QGroupBox("缩略图预览")
        thumbnail_layout = QVBoxLayout(thumbnail_group)
        self.batch_list = QListWidget()
        self.batch_list.setViewMode(QListWidget.IconMode)
        self.batch_list.setIconSize(QSize(100, 100))
        self.batch_list.setResizeMode(QListWidget.Adjust)
        self.batch_list.setSpacing(10)
        self.batch_list.itemClicked.connect(self.show_batch_detail)
        self.batch_list.itemDoubleClicked.connect(self.show_zoomed_image)
        thumbnail_layout.addWidget(self.batch_list)
        batch_layout.addWidget(thumbnail_group)

        # Detailed result view
        batch_result_group = QGroupBox("处理结果")
        batch_result_layout = QHBoxLayout(batch_result_group)
        self.batch_result_table = QTableWidget()
        self.batch_result_table.setColumnCount(2)
        self.batch_result_table.setHorizontalHeaderLabels(["Class", "Score"])
        self.batch_result_table.setRowCount(0)
        self.batch_result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_result_table.setStyleSheet("""
            QTableWidget {
                background-color: #252526;
                alternate-background-color: #333333;
                color: #DCDCDC;
                gridline-color: #404040;
            }
            QHeaderView::section {
                background-color: #333;
                color: #DCDCDC;
            }
        """)
        self.batch_result_table.horizontalHeader().setStretchLastSection(True)
        self.batch_visual_label = QLabel()
        self.batch_visual_label.setObjectName("batch_visual_label")
        self.batch_visual_label.setAlignment(Qt.AlignCenter)
        self.batch_visual_label.setMinimumSize(400, 400)
        self.batch_visual_label.setStyleSheet("""
            QLabel#batch_visual_label {
                background-color: #252526;
                border: 1px solid #3C3C3C;
                border-radius: 5px;
            }
        """)
        batch_result_layout.addWidget(self.batch_result_table, 1)
        batch_result_layout.addWidget(self.batch_visual_label)
        batch_layout.addWidget(batch_result_group)

        self.main_tabs.addTab(batch_tab, "批量模式")

        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        self.monitor_scroll = QScrollArea()
        self.monitor_scroll.setWidgetResizable(True)
        self.monitor_widget = QWidget()
        self.monitor_layout = QVBoxLayout(self.monitor_widget)
        self.monitor_scroll.setWidget(self.monitor_widget)
        monitor_layout.addWidget(self.monitor_scroll)
        self.main_tabs.addTab(monitor_tab, "系统监控")

        central_layout.addWidget(content_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        # Setup overlay for blocking operations
        self.overlay = QWidget(self)
        self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        self.overlay.hide()
        self.overlay.setGeometry(self.rect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.rect())

    def on_tab_changed(self, index):
        # Check if the system monitoring tab is selected (index 2)
        if index == 2:
            self.monitor_worker.start_monitoring()
        else:
            self.monitor_worker.stop_monitoring()

    def show_overlay(self):
        self.overlay.show()
        self.overlay.raise_()

    def hide_overlay(self):
        self.overlay.hide()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(valid_files) == 1:
            self.current_image = cv2.imread(valid_files[0])
            self.update_drop_preview(valid_files[0])
            self.status_bar.showMessage("已载入单张图像")
        elif len(valid_files) > 1:
            self.batch_files = valid_files
            self.batch_list.clear()
            for file in valid_files:
                self.add_thumbnail(file)
            self.status_bar.showMessage(f"已载入 {len(valid_files)} 张图像")
        else:
            self.status_bar.showMessage("错误: 不支持的文件类型")

    def add_thumbnail(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(100, 100, Qt.KeepAspectRatio)
        item = QListWidgetItem()
        item.setIcon(QIcon(pixmap))
        item.setText(os.path.basename(file_path))
        item.setData(Qt.UserRole, file_path)  # Store file path for zoom dialog
        self.batch_list.addItem(item)

    def update_drop_preview(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(600, 400, Qt.KeepAspectRatio)
        self.drop_area.setPixmap(pixmap)
        self.drop_area.setText("")

    def upload_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg)")
        if file:
            self.current_image = cv2.imread(file)
            self.update_drop_preview(file)

    def batch_upload(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择图像", "", "图像文件 (*.png *.jpg *.jpeg)")
        if files:
            self.batch_files = files
            self.batch_list.clear()
            for file in files:
                self.add_thumbnail(file)
            self.status_bar.showMessage(f"已选择 {len(files)} 张图像")

    def show_zoomed_image(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path:
            dialog = ImageZoomDialog(file_path, self)
            dialog.exec_()

    def process_image(self):
        if self.current_image is not None and not self.batch_files:
            model_name = self.model_selector.currentText()
            conf = self.confidence_slider.value() / 100
            enable_vis = self.visual_toggle.isChecked()
            self.btn_process.setEnabled(False)
            self.status_bar.showMessage("正在处理图像...")
            self.show_overlay()
            self.progress_dialog = QProgressDialog("处理图像...", None, 0, 0, self)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setWindowModality(Qt.NonModal)
            self.progress_dialog.show()
            self.process_worker.process_image(self.current_image, model_name, conf, enable_vis)
        elif self.batch_files:
            self.batch_results = []
            model_name = self.model_selector.currentText()
            conf = self.confidence_slider.value() / 100
            enable_vis = self.visual_toggle.isChecked()
            self.btn_process.setEnabled(False)
            self.status_bar.showMessage("正在处理批量图像...")
            self.show_overlay()
            self.progress_dialog = QProgressDialog("处理批量图像...", None, 0, len(self.batch_files), self)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setWindowModality(Qt.NonModal)
            self.progress_dialog.show()
            self.process_worker.process_batch(self.batch_files, model_name, conf, enable_vis)

    def on_model_selected(self, index):
        model_name = self.model_selector.itemText(index)
        self.status_bar.showMessage(f"正在加载模型 {model_name}...")
        self.show_overlay()
        self.model_worker.ensure_model_loaded(model_name)

    def update_download_progress(self, current, total):
        if not hasattr(self, 'progress_dialog'):
            self.progress_dialog = QProgressDialog("下载模型...", None, 0, total, self)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setWindowModality(Qt.NonModal)
            self.show_overlay()
            self.progress_dialog.show()
        self.progress_dialog.setMaximum(total)
        self.progress_dialog.setValue(current)
        if current == total:
            self.progress_dialog.close()
            self.hide_overlay()
            del self.progress_dialog

    def on_model_load_started(self):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setLabelText("加载模型中...")
            self.progress_dialog.setMaximum(0)
        else:
            self.progress_dialog = QProgressDialog("加载模型中...", None, 0, 0, self)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setWindowModality(Qt.NonModal)
            self.show_overlay()
            self.progress_dialog.show()

    def on_model_loaded(self, model_name):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            del self.progress_dialog
        self.hide_overlay()
        self.status_bar.showMessage(f"模型 {model_name} 已加载")
        self.btn_process.setEnabled(True)

    def on_model_error(self, error):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            del self.progress_dialog
        self.hide_overlay()
        QMessageBox.critical(self, "错误", error)
        self.status_bar.showMessage(f"错误: {error}")
        self.btn_process.setEnabled(True)

    def update_batch_progress(self, current, total):
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setValue(current)
            if current == total:
                self.progress_dialog.close()
                del self.progress_dialog
                self.hide_overlay()
                self.status_bar.showMessage("批量处理完成")
                self.btn_process.setEnabled(True)

    def on_processing_finished(self, results, vis, img_path):
        if img_path:  # Batch mode
            self.batch_results.append({
                'path': img_path,
                'results': results,
                'visualization': vis
            })
            # Show first result by default
            if len(self.batch_results) == 1:
                self.show_batch_detail(self.batch_list.item(0))
        else:  # Single mode
            self.result_table.setRowCount(0)
            for i, (cls, score) in enumerate(results.items()):
                row_position = self.result_table.rowCount()
                self.result_table.insertRow(row_position)
                self.result_table.setItem(row_position, 0, QTableWidgetItem(cls))
                self.result_table.setItem(row_position, 1, QTableWidgetItem(f"{score:.4f}"))
            if self.result_table.rowCount() > 0:
                for col in range(2):
                    item = self.result_table.item(0, col)
                    item.setBackground(QColor("#336699"))
                    item.setForeground(QColor("white"))
            if vis is not None:
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                h, w = vis.shape[:2]
                qimg = QImage(vis.data, w, h, w * 3, QImage.Format_RGB888)
                self.visual_label.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio))
            else:
                self.visual_label.clear()
            self.hide_overlay()
            self.status_bar.showMessage("处理完成")
            self.btn_process.setEnabled(True)

    def on_processing_error(self, error):
        self.status_bar.showMessage(f"错误: {error}")
        QMessageBox.critical(self, "处理错误", error)
        self.btn_process.setEnabled(True)
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            del self.progress_dialog
        self.hide_overlay()

    def on_monitor_data_ready(self, monitor_data, text_info):
        for i in reversed(range(self.monitor_layout.count())):
            self.monitor_layout.itemAt(i).widget().setParent(None)
        info_label = QLabel(text_info)
        self.monitor_layout.addWidget(info_label)
        charts_pixmap = QPixmap()
        charts_pixmap.loadFromData(monitor_data["charts"])
        charts_label = QLabel()
        charts_label.setPixmap(charts_pixmap)
        self.monitor_layout.addWidget(charts_label)

    def show_batch_detail(self, item):
        index = self.batch_list.row(item)
        result_data = self.batch_results[index] if index < len(self.batch_results) else None
        self.batch_result_table.setRowCount(0)
        if result_data:
            for i, (cls, score) in enumerate(result_data['results'].items()):
                row_position = self.batch_result_table.rowCount()
                self.batch_result_table.insertRow(row_position)
                self.batch_result_table.setItem(row_position, 0, QTableWidgetItem(cls))
                self.batch_result_table.setItem(row_position, 1, QTableWidgetItem(f"{score:.4f}"))
            if self.batch_result_table.rowCount() > 0:
                for col in range(2):
                    item = self.batch_result_table.item(0, col)
                    item.setBackground(QColor("#336699"))
                    item.setForeground(QColor("white"))
            vis = result_data['visualization']
            if vis is not None:
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                h, w = vis.shape[:2]
                qimg = QImage(vis.data, w, h, w * 3, QImage.Format_RGB888)
                self.batch_visual_label.setPixmap(QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio))
            else:
                self.batch_visual_label.clear()
        else:
            self.batch_visual_label.clear()

    def closeEvent(self, event):
        self.monitor_thread.quit()
        self.monitor_thread.wait()
        self.model_thread.quit()
        self.model_thread.wait()
        self.process_thread.quit()
        self.process_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1E1E1E, stop:1 #252526);
            color: #DCDCDC;
            font-family: 'Segoe UI';
            font-size: 14px;
        }
        QTableWidget {
            background-color: #252526;
            alternate-background-color: #333333;
            color: #DCDCDC;
            gridline-color: #404040;
        }
        QHeaderView::section {
            background-color: #333;
            color: #DCDCDC;
        }
        QGroupBox {
            border: 1px solid #3A3A3A;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 15px;
        }
        QGroupBox::title {
            color: #9CDCFE;
            subcontrol-origin: margin;
            left: 10px;
        }
        QComboBox {
            background: #252526;
            border: 1px solid #3C3C3C;
            border-radius: 4px;
            padding: 5px;
            min-width: 120px;
        }
        QComboBox:hover {
            border-color: #007ACC;
        }
        QComboBox::drop-down {
            width: 25px;
            border-left: 1px solid #3C3C3C;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #404040;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #007ACC;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 2px solid #3C3C3C;
            border-radius: 4px;
        }
        QCheckBox::indicator:checked {
            background-color: #007ACC;
            border-color: #007ACC;
        }
        QPushButton {
            background: #333;
            border: 1px solid #3C3C3C;
            border-radius: 4px;
            padding: 8px;
            min-width: 100px;
        }
        QPushButton:hover {
            background: #404040;
            border-color: #007ACC;
        }
        QTextEdit {
            background: #252526;
            border: 1px solid #3C3C3C;
            border-radius: 4px;
            padding: 8px;
        }
        QListWidget {
            background: #252526;
            border: 1px solid #3C3C3C;
            border-radius: 4px;
        }
        QListWidget::item {
            background: #333;
            border-radius: 4px;
            margin: 5px;
        }
        QListWidget::item:hover {
            background: #404040;
        }
        QListWidget::item:selected {
            background: #007ACC;
        }
        QStatusBar {
            background: #252526;
            border-top: 1px solid #3C3C3C;
        }
        QTabWidget::pane {
            border: 1px solid #3C3C3C;
            background: #252526;
        }
        QTabBar::tab {
            background: #000000;
            color: #FFFFFF;
            padding: 8px;
            min-width: 30px;
            border: none;
        }
        QTabBar::tab:selected {
            background: #333333;
            color: #FFFFFF;
            border-bottom: 2px solid #007ACC;
        }
        QTabBar::tab:hover {
            background: #1A1A1A;
            color: #FFFFFF;
        }
    """)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())