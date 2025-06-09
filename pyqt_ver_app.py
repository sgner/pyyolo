import sys
import gc
import psutil
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QCheckBox, QPushButton, QFileDialog,
    QListWidget, QTextEdit, QStatusBar, QProgressDialog, QListWidgetItem,
    QGroupBox, QTabWidget, QFrame, QScrollArea, QTableWidget, QTableWidgetItem,
    QMessageBox, QDialog, QMenu, QAction, QSplitter, QProgressBar, QSpinBox,
    QToolTip, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QSize, QMimeData, QThread, pyqtSignal, QObject, QUrl, QSettings, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon, QDragEnterEvent, QDropEvent, QColor, QFont, QKeySequence, QPainter, QBrush, QLinearGradient
from PyQt5.QtNetwork import QNetworkAccessManager, QNetworkRequest
import cv2
import numpy as np
from ultralytics import YOLO
import yaml
from pathlib import Path
import os
import time
import logging
from datetime import datetime
from monitor.system_monitor import SystemMonitor

# 确保日志目录存在
os.makedirs('logs', exist_ok=True)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 内存管理类
class MemoryManager:
    """内存管理器"""
    
    def __init__(self, max_memory_mb=2048):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = {}
        self.max_cache_size = 50
    
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def cleanup_if_needed(self):
        """根据需要清理内存"""
        current_memory = self.get_memory_usage()
        if current_memory > self.max_memory_bytes / 1024 / 1024:
            logger.info(f"内存使用过高 ({current_memory:.1f}MB)，开始清理")
            self.cleanup_cache()
            gc.collect()
    
    def cleanup_cache(self):
        """清理缓存"""
        if len(self.cache) > self.max_cache_size:
            # 删除一半的缓存项
            items_to_remove = len(self.cache) // 2
            for _ in range(items_to_remove):
                if self.cache:
                    self.cache.popitem()
        logger.info(f"缓存清理完成，剩余项目: {len(self.cache)}")

# 文件验证器
class FileValidator:
    """文件验证器"""
    
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    
    @classmethod
    def validate_image_file(cls, file_path):
        """验证图像文件"""
        try:
            path = Path(file_path)
            
            # 检查文件是否存在
            if not path.exists():
                return False, "文件不存在"
            
            # 检查文件扩展名
            if path.suffix.lower() not in cls.ALLOWED_EXTENSIONS:
                return False, f"不支持的文件类型: {path.suffix}"
            
            # 检查文件大小
            file_size = path.stat().st_size
            if file_size > cls.MAX_FILE_SIZE:
                size_mb = file_size / (1024 * 1024)
                return False, f"文件过大: {size_mb:.1f}MB (最大50MB)"
            
            # 基本的图像头验证
            if not cls._validate_image_header(path):
                return False, "文件内容无效"
            
            return True, None
            
        except Exception as e:
            logger.error(f"文件验证失败: {e}")
            return False, f"验证失败: {str(e)}"
    
    @classmethod
    def _validate_image_header(cls, file_path):
        """验证图像文件头"""
        try:
            signatures = {
                b'\xFF\xD8\xFF': 'JPEG',
                b'\x89PNG\r\n\x1a\n': 'PNG',
                b'BM': 'BMP',
                b'II*\x00': 'TIFF',
                b'MM\x00*': 'TIFF'
            }
            
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            for signature in signatures.keys():
                if header.startswith(signature):
                    return True
            
            return False
            
        except Exception:
            return False

# 加载配置文件
def load_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        for model in config["models"].values():
            model["path"] = str(Path(model["path"])).replace("\\", "/")
        logger.info("配置文件加载成功")
        return config
    except Exception as e:
        logger.error(f"配置加载失败: {str(e)}")
        raise RuntimeError(f"配置加载失败: {str(e)}")

CONFIG = load_config()

# 优化的模型管理线程工作类
class ModelManagerWorker(QObject):
    model_download_progress = pyqtSignal(int, int)
    model_load_started = pyqtSignal()
    model_loaded = pyqtSignal(str)
    model_error = pyqtSignal(str)

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.current_model = None
        self.current_model_name = None
        self.network_manager = QNetworkAccessManager()
        self.network_manager.finished.connect(self.download_finished)
        self.downloading_model_name = None
        self.reply = None
        self.model_cache = {}  # 模型缓存

    def ensure_model_loaded(self, model_name):
        try:
            if self.current_model_name == model_name:
                self.model_loaded.emit(model_name)
            elif model_name in self.model_cache:
                # 从缓存加载
                self.current_model = self.model_cache[model_name]
                self.current_model_name = model_name
                self.model_loaded.emit(model_name)
                logger.info(f"从缓存加载模型: {model_name}")
            elif Path(self.models[model_name]["path"]).exists():
                self.load_model(model_name)
            else:
                self.downloading_model_name = model_name
                self.download_model(model_name)
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model_error.emit(str(e))

    def download_model(self, model_name):
        try:
            model_cfg = self.models[model_name]
            url = model_cfg["url"]
            path = model_cfg["path"]
            os.makedirs(os.path.dirname(path), exist_ok=True)
            request = QNetworkRequest(QUrl(url))
            self.reply = self.network_manager.get(request)
            self.reply.downloadProgress.connect(self.on_download_progress)
            logger.info(f"开始下载模型: {model_name}")
        except Exception as e:
            logger.error(f"模型下载失败: {e}")
            self.model_error.emit(str(e))

    def on_download_progress(self, bytes_received, bytes_total):
        self.model_download_progress.emit(bytes_received, bytes_total)

    def download_finished(self, reply):
        try:
            if reply.error():
                self.model_error.emit(f"下载失败: {reply.errorString()}")
                return
            model_name = self.downloading_model_name
            model_cfg = self.models[model_name]
            path = model_cfg["path"]
            with open(path, 'wb') as f:
                f.write(reply.readAll())
            logger.info(f"模型下载完成: {model_name}")
            self.load_model(model_name)
        except Exception as e:
            logger.error(f"下载完成处理失败: {e}")
            self.model_error.emit(str(e))

    def load_model(self, model_name):
        self.model_load_started.emit()
        try:
            model_cfg = self.models[model_name]
            path = model_cfg["path"]
            self.current_model = YOLO(path)
            self.current_model_name = model_name
            
            # 缓存模型（限制缓存数量）
            if len(self.model_cache) >= 2:  # 最多缓存2个模型
                oldest_model = next(iter(self.model_cache))
                del self.model_cache[oldest_model]
                logger.info(f"清理模型缓存: {oldest_model}")
            
            self.model_cache[model_name] = self.current_model
            self.model_loaded.emit(model_name)
            logger.info(f"模型加载成功: {model_name}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model_error.emit(f"模型加载失败: {str(e)}")

    def get_model(self, model_name):
        if self.current_model_name == model_name:
            return self.current_model
        return None

# 优化的图像处理线程工作类
class ImageProcessorWorker(QObject):
    processing_finished = pyqtSignal(dict, np.ndarray, str)
    processing_error = pyqtSignal(str)
    batch_progress = pyqtSignal(int, int)
    processing_progress = pyqtSignal(str)  # 处理进度信号
    classification_ready = pyqtSignal(dict, str)  # 分类结果就绪信号（优先显示）
    visualization_ready = pyqtSignal(np.ndarray, str)  # 可视化结果就绪信号
    batch_item_finished = pyqtSignal(dict, np.ndarray, str, int, int)  # 批量单项完成信号
    batch_all_finished = pyqtSignal()  # 批量全部完成信号
    detailed_info_ready = pyqtSignal(dict, str)  # 详细信息就绪信号

    def __init__(self, model_manager_worker, memory_manager):
        super().__init__()
        self.model_manager_worker = model_manager_worker
        self.memory_manager = memory_manager
        self.image_cache = {}
        self.current_batch_index = 0
        self.current_batch_total = 0
        self.is_batch_processing = False

    def process_image(self, img, model_name, confidence_threshold, enable_visualization, img_path=""):
        self.img = img
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.enable_visualization = enable_visualization
        self.img_path = img_path
        self.is_batch_processing = False
        
        # 内存检查
        self.memory_manager.cleanup_if_needed()
        
        if self.model_manager_worker.current_model_name == model_name:
            self.do_process()
        else:
            self.model_manager_worker.model_loaded.connect(self.on_model_loaded)
            self.model_manager_worker.model_error.connect(self.on_model_error)
            self.model_manager_worker.ensure_model_loaded(model_name)

    def process_batch(self, img_paths, model_name, confidence_threshold, enable_visualization):
        logger.info(f"=== ImageProcessorWorker.process_batch 开始 ===")
        logger.info(f"接收到批量处理请求: {len(img_paths)} 张图像")
        logger.info(f"参数: model={model_name}, conf={confidence_threshold}, vis={enable_visualization}")
        
        self.batch_img_paths = img_paths
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.enable_visualization = enable_visualization
        self.current_batch_index = 0
        self.current_batch_total = len(img_paths)
        self.is_batch_processing = True
        
        logger.info(f"批量处理状态初始化完成: index={self.current_batch_index}, total={self.current_batch_total}")
        
        # 内存检查
        self.memory_manager.cleanup_if_needed()
        
        if self.model_manager_worker.current_model_name == model_name:
            logger.info("模型已加载，直接开始批量处理")
            self.do_batch_process()
        else:
            logger.info(f"需要加载模型: {model_name}")
            self.model_manager_worker.model_loaded.connect(self.on_model_loaded)
            self.model_manager_worker.model_error.connect(self.on_model_error)
            self.model_manager_worker.ensure_model_loaded(model_name)

    def on_model_loaded(self, loaded_model_name):
        try:
            if loaded_model_name == self.model_name:
                if self.is_batch_processing:
                    self.do_batch_process()
                else:
                    self.do_process()
                self.model_manager_worker.model_loaded.disconnect(self.on_model_loaded)
                self.model_manager_worker.model_error.disconnect(self.on_model_error)
        except Exception as e:
            logger.error(f"模型加载后处理失败: {e}")

    def on_model_error(self, error):
        self.processing_error.emit(error)
        try:
            self.model_manager_worker.model_loaded.disconnect(self.on_model_loaded)
            self.model_manager_worker.model_error.disconnect(self.on_model_error)
        except:
            pass

    def do_process(self):
        try:
            logger.info("=== 开始图像处理 ===")
            self.processing_progress.emit("正在加载模型...")
            
            model = self.model_manager_worker.get_model(self.model_name)
            if model is None:
                raise RuntimeError("模型未加载")
            
            logger.info("✓ 模型获取成功")
            
            # 缓存检查
            cache_key = f"{self.img_path}_{self.confidence_threshold}_{self.model_name}"
            if cache_key in self.image_cache:
                logger.info("使用缓存结果")
                self.processing_progress.emit("使用缓存结果...")
                cached_result = self.image_cache[cache_key]
                
                if self.is_batch_processing:
                    self.batch_item_finished.emit(cached_result[0], cached_result[1], self.img_path, 
                                                self.current_batch_index + 1, self.current_batch_total)
                else:
                    self.classification_ready.emit(cached_result[0], self.img_path)
                    self.visualization_ready.emit(cached_result[1], self.img_path)
                    
                return
            
            # 简化图像预处理
            logger.info("开始图像预处理...")
            self.processing_progress.emit("正在预处理图像...")
            
            h, w = self.img.shape[:2]
            logger.info(f"原始图像尺寸: {w}x{h}")
            
            # 使用固定的预处理逻辑，避免复杂计算
            target_size = 640
            scale = target_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            processed_img = cv2.resize(self.img, (new_w, new_h))
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            logger.info(f"✓ 图像预处理完成: {new_w}x{new_h}")
            
            # 模型推理
            logger.info("开始模型推理...")
            self.processing_progress.emit("正在进行AI推理...")
            
            start_time = time.time()
            results = model(processed_img)
            inference_time = time.time() - start_time
            logger.info(f"✓ 推理耗时: {inference_time:.3f}秒")
            
            # *** 关键修复：立即处理推理结果，添加详细日志 ***
            logger.info("立即开始处理推理结果...")
            
            try:
                result_obj = results[0]
                logger.info("✓ 结果对象获取成功")
                
                # 安全获取概率数据
                probs_data = result_obj.probs.data
                logger.info(f"✓ 概率张量获取成功，形状: {probs_data.shape}")
                
                probs = probs_data.tolist()
                logger.info(f"✓ 概率列表转换成功，长度: {len(probs)}")
                
                # 安全获取类别名称
                model_names = model.names
                logger.info(f"✓ 模型类别获取成功，数量: {len(model_names)}")
                
                # 构建结果字典
                all_results = {}
                for i in range(len(probs)):
                    if i < len(model_names):
                        all_results[model_names[i]] = float(probs[i])
                
                logger.info(f"✓ 全部结果构建成功，共{len(all_results)}个类别")
                
            except Exception as e:
                logger.error(f"❌ 结果处理失败: {e}")
                import traceback
                logger.error(f"详细错误: {traceback.format_exc()}")
                raise
            
            # 过滤结果
            logger.info("开始过滤结果...")
            filtered_results = {k: v for k, v in all_results.items() if v >= self.confidence_threshold}
            if not filtered_results:
                max_key = max(all_results, key=all_results.get)
                filtered_results = {max_key: all_results[max_key]}
            
            sorted_results = dict(sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)[:5])
            logger.info(f"✓ 过滤后结果数量: {len(sorted_results)}")
            
            # *** 立即发送分类结果（单图模式优先显示）***
            if not self.is_batch_processing:
                logger.info("单图模式 - 立即发送分类结果")
                try:
                    self.classification_ready.emit(sorted_results, self.img_path)
                    logger.info("✓ 分类结果信号发送成功")
                except Exception as e:
                    logger.error(f"❌ 分类结果发送失败: {e}")
                    raise
            
            # 生成可视化
            visualization = None
            if self.enable_visualization:
                logger.info("开始生成可视化结果...")
                try:
                    vis = result_obj.plot()
                    visualization = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                    logger.info("✓ 可视化生成成功")
                except Exception as e:
                    logger.warning(f"⚠️ 可视化生成失败: {e}")
                    # 创建简单的错误图像
                    visualization = np.zeros((640, 640, 3), dtype=np.uint8)
                    cv2.putText(visualization, "可视化失败", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                logger.info("跳过可视化生成")
            
            # 简化缓存处理
            logger.info("更新缓存...")
            try:
                if len(self.image_cache) > 10:  # 减小缓存大小
                    oldest_key = next(iter(self.image_cache))
                    del self.image_cache[oldest_key]
                self.image_cache[cache_key] = (sorted_results, visualization)
                logger.info("✓ 缓存更新成功")
            except Exception as e:
                logger.warning(f"⚠️ 缓存更新失败: {e}")
            
            # 根据模式发送信号
            if self.is_batch_processing:
                logger.info("发送批量项完成信号...")
                self.batch_item_finished.emit(sorted_results, visualization, self.img_path, 
                                            self.current_batch_index + 1, self.current_batch_total)
            else:
                logger.info("发送可视化结果信号...")
                if self.enable_visualization:
                    self.visualization_ready.emit(visualization, self.img_path)
                else:
                    self.visualization_ready.emit(None, self.img_path)
            
            # 发送最终完成信号
            logger.info(f"发送处理完成信号: img_path='{self.img_path}'")
            self.processing_finished.emit(sorted_results, visualization, self.img_path)
            
            logger.info("=== 图像处理完成 ===")
            
        except Exception as e:
            logger.error(f"❌ 图像处理失败: {e}")
            import traceback
            logger.error(f"详细错误: {traceback.format_exc()}")
            self.processing_error.emit(str(e))

    def do_batch_process(self):
        """改进的批量处理 - 异步处理每个图像"""
        try:
            if self.current_batch_index >= len(self.batch_img_paths):
                # 所有图像处理完成
                logger.info("批量处理全部完成")
                self.batch_all_finished.emit()
                return
            
            # 处理当前图像
            img_path = self.batch_img_paths[self.current_batch_index]
            logger.info(f"处理批量图像 {self.current_batch_index + 1}/{self.current_batch_total}: {img_path}")
            
            # 验证文件
            is_valid, error_msg = FileValidator.validate_image_file(img_path)
            if not is_valid:
                logger.warning(f"跳过无效文件 {img_path}: {error_msg}")
                self.current_batch_index += 1
                self.batch_progress.emit(self.current_batch_index, self.current_batch_total)
                # 递归处理下一个文件
                QTimer.singleShot(10, self.do_batch_process)
                return
            
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"无法读取图像: {img_path}")
                self.current_batch_index += 1
                self.batch_progress.emit(self.current_batch_index, self.current_batch_total)
                # 递归处理下一个文件
                QTimer.singleShot(10, self.do_batch_process)
                return
            
            # 设置当前处理的图像
            self.img = img
            self.img_path = img_path
            
            # 处理图像
            self.do_process()
            
            # 更新索引
            self.current_batch_index += 1
            self.batch_progress.emit(self.current_batch_index, self.current_batch_total)
            
            # 定期清理内存
            if self.current_batch_index % 10 == 0:
                self.memory_manager.cleanup_if_needed()
            
            # 清理图像数据
            del img
            
            # 异步处理下一个图像
            if self.current_batch_index < self.current_batch_total:
                QTimer.singleShot(50, self.do_batch_process)  # 50ms延迟，避免阻塞UI
            else:
                # 所有图像处理完成
                logger.info("批量处理全部完成")
                self.batch_all_finished.emit()
                
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            self.processing_error.emit(str(e))

# 优化的自定义标题栏
class TitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        self.setMouseTracking(True)

    def initUI(self):
        self.setFixedHeight(35)  # 增加高度
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 0)
        layout.setSpacing(0)

        # 应用图标
        self.icon_label = QLabel(self)
        icon_pixmap = QPixmap(16, 16)
        icon_pixmap.fill(QColor(0, 122, 204))
        self.icon_label.setPixmap(icon_pixmap)
        layout.addWidget(self.icon_label)
        
        layout.addSpacing(8)

        self.title_label = QLabel("AI图像分类系统", self)
        self.title_label.setStyleSheet("""
            QLabel {
                color: #323130;
                font-weight: bold;
                font-size: 14px;
                font-family: 'Microsoft YaHei', 'Segoe UI';
            }
        """)
        layout.addWidget(self.title_label, 1)

        # 系统信息标签
        self.info_label = QLabel("", self)
        self.info_label.setStyleSheet("""
            QLabel {
                color: #605E5C;
                font-size: 11px;
                font-family: 'Consolas', monospace;
            }
        """)
        layout.addWidget(self.info_label)
        
        layout.addSpacing(10)

        # 窗口控制按钮
        self.minimize_btn = QPushButton("🗕", self)
        self.maximize_btn = QPushButton("🗖", self)
        self.close_btn = QPushButton("🗙", self)
        
        buttons = [self.minimize_btn, self.maximize_btn, self.close_btn]
        for btn in buttons:
            btn.setFixedSize(35, 35)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #323130;
                    border: none;
                    font-size: 14px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: rgba(0, 0, 0, 0.1);
                }
            """)

        # 特殊样式的关闭按钮
        self.close_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #323130;
                border: none;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #E81123;
                color: white;
            }
        """)

        self.minimize_btn.clicked.connect(self.parent.showMinimized)
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        self.close_btn.clicked.connect(self.parent.close)

        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)

        # 设置渐变背景
        self.setStyleSheet("""
            TitleBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F8F9FA, stop:1 #FFFFFF);
                border-bottom: 1px solid #0078D4;
            }
        """)
        
        # 定时器更新系统信息
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_info)
        self.update_timer.start(2000)  # 每2秒更新一次

    def update_system_info(self):
        """更新系统信息显示"""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            info_text = f"CPU: {cpu_percent:.1f}% | 内存: {memory.percent:.1f}%"
            self.info_label.setText(info_text)
        except:
            pass

    def toggle_maximize(self):
        if self.parent.isMaximized():
            self.parent.showNormal()
            self.maximize_btn.setText("🗖")
        else:
            self.parent.showMaximized()
            self.maximize_btn.setText("🗗")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = self.parent.pos()
            self.start_mouse = event.globalPos()

    def mouseMoveEvent(self, event):
        if hasattr(self, 'start_pos') and event.buttons() == Qt.LeftButton:
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

    def start_monitoring(self):
        if not self.timer.isActive():
            self.timer.start(1000)

    def stop_monitoring(self):
        if self.timer.isActive():
            self.timer.stop()

    def update_monitor(self):
        try:
            self.system_monitor.update_hardware_info()
            monitor_data = self.system_monitor.get_monitor_data()
            text_info = self.system_monitor.get_text_info()
            self.monitor_data_ready.emit(monitor_data, text_info)
        except Exception as e:
            logger.error(f"监控更新失败: {e}")

# 优化的图像预览对话框
class ImageZoomDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("图像预览")
        self.setMinimumSize(800, 600)
        self.image_path = image_path
        self.scale_factor = 1.0
        self.initUI()
        self.load_image()
        self.setup_shortcuts()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 工具栏
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        
        # 缩放控制
        zoom_in_btn = QPushButton("放大 (+)")
        zoom_out_btn = QPushButton("缩小 (-)")
        reset_btn = QPushButton("重置 (R)")
        fit_btn = QPushButton("适应窗口 (F)")
        
        zoom_in_btn.clicked.connect(lambda: self.zoom(1.2))
        zoom_out_btn.clicked.connect(lambda: self.zoom(0.8))
        reset_btn.clicked.connect(self.reset_zoom)
        fit_btn.clicked.connect(self.fit_to_window)
        
        toolbar_layout.addWidget(QLabel("缩放:"))
        toolbar_layout.addWidget(zoom_in_btn)
        toolbar_layout.addWidget(zoom_out_btn)
        toolbar_layout.addWidget(reset_btn)
        toolbar_layout.addWidget(fit_btn)
        toolbar_layout.addStretch()
        
        # 缩放比例显示
        self.scale_label = QLabel("100%")
        toolbar_layout.addWidget(self.scale_label)
        
        layout.addWidget(toolbar)
        
        # 图像显示区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #F8F9FA;")
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        
        # 状态栏
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        self.status_label = QLabel()
        status_layout.addWidget(self.status_label)
        layout.addWidget(status_bar)

        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
                color: #323130;
            }
            QPushButton {
                background: #F8F9FA;
                border: 1px solid #D2D0CE;
                border-radius: 4px;
                padding: 5px 10px;
                color: #323130;
            }
            QPushButton:hover {
                background: #F0F0F0;
                border-color: #0078D4;
            }
            QScrollArea {
                background-color: #FFFFFF;
                border: 1px solid #D2D0CE;
            }
        """)

    def setup_shortcuts(self):
        """设置快捷键"""
        from PyQt5.QtWidgets import QShortcut
        
        QShortcut(QKeySequence("+"), self, lambda: self.zoom(1.2))
        QShortcut(QKeySequence("-"), self, lambda: self.zoom(0.8))
        QShortcut(QKeySequence("R"), self, self.reset_zoom)
        QShortcut(QKeySequence("F"), self, self.fit_to_window)
        QShortcut(QKeySequence("Escape"), self, self.close)

    def load_image(self):
        try:
            img = cv2.imread(self.image_path)
            if img is None:
                self.image_label.setText("无法加载图像")
                return
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            self.original_pixmap = QPixmap.fromImage(qimg)
            self.update_image()
            
            # 更新状态信息
            file_size = Path(self.image_path).stat().st_size / 1024  # KB
            self.status_label.setText(f"尺寸: {w}×{h} | 大小: {file_size:.1f} KB | 路径: {Path(self.image_path).name}")
            
        except Exception as e:
            logger.error(f"图像加载失败: {e}")
            self.image_label.setText(f"图像加载失败: {e}")

    def zoom(self, factor):
        """缩放图像"""
        self.scale_factor *= factor
        self.scale_factor = max(0.1, min(self.scale_factor, 10.0))
        self.update_image()

    def reset_zoom(self):
        """重置缩放"""
        self.scale_factor = 1.0
        self.update_image()

    def fit_to_window(self):
        """适应窗口大小"""
        if not hasattr(self, 'original_pixmap'):
            return
        
        scroll_size = self.scroll_area.size()
        pixmap_size = self.original_pixmap.size()
        
        scale_w = scroll_size.width() / pixmap_size.width()
        scale_h = scroll_size.height() / pixmap_size.height()
        self.scale_factor = min(scale_w, scale_h) * 0.9  # 留出边距
        
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
        
        # 更新缩放比例显示
        self.scale_label.setText(f"{int(self.scale_factor * 100)}%")

    def wheelEvent(self, event):
        # 使用Ctrl+滚轮进行缩放
        if event.modifiers() == Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom(1.1)
            else:
                self.zoom(0.9)
        else:
            super().wheelEvent(event)

# 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowTitle(CONFIG.get("title", "AI图像分类系统"))
        self.setGeometry(100, 100, 1600, 1000)  # 增大默认窗口尺寸
        
        # 初始化组件
        self.memory_manager = MemoryManager()
        self.settings = QSettings('AIClassifier', 'MainWindow')
        self.current_image = None
        self.current_image_path = ""  # 初始化当前图片路径
        self.batch_files = []
        self.batch_results = []
        self.recent_files = []
        
        # 处理状态管理
        self.is_processing = False
        
        # 加载设置
        self.load_settings()
        
        # 系统监控
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
        self.process_worker = ImageProcessorWorker(self.model_worker, self.memory_manager)
        self.process_worker.moveToThread(self.process_thread)
        self.process_thread.start()
        self.process_worker.processing_finished.connect(self.on_processing_finished)
        self.process_worker.processing_error.connect(self.on_processing_error)
        self.process_worker.batch_progress.connect(self.update_batch_progress)
        self.process_worker.processing_progress.connect(self.on_processing_progress)
        self.process_worker.classification_ready.connect(self.on_classification_ready)
        self.process_worker.visualization_ready.connect(self.on_visualization_ready)
        self.process_worker.batch_item_finished.connect(self.on_batch_item_finished)
        self.process_worker.batch_all_finished.connect(self.on_batch_all_finished)
        self.process_worker.detailed_info_ready.connect(self.on_detailed_info_ready)

        self.setup_ui()
        self.setup_shortcuts()
        self.setup_context_menus()
        self.setAcceptDrops(True)

        # 加载默认模型
        default_model = self.settings.value('default_model', 'nano')
        if default_model in CONFIG["models"]:
            index = list(CONFIG["models"].keys()).index(default_model)
            self.model_selector.setCurrentIndex(index)
        self.model_worker.ensure_model_loaded(self.model_selector.currentText())
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)

        # 连接标签页切换信号
        self.main_tabs.currentChanged.connect(self.on_tab_changed)
        
        # 状态更新定时器
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(5000)  # 每5秒更新状态

    def load_settings(self):
        """加载用户设置"""
        try:
            # 窗口几何
            geometry = self.settings.value('geometry')
            if geometry:
                self.restoreGeometry(geometry)
            
            # 最近文件
            self.recent_files = self.settings.value('recent_files', [])
            if not isinstance(self.recent_files, list):
                self.recent_files = []
                
        except Exception as e:
            logger.error(f"设置加载失败: {e}")

    def save_settings(self):
        """保存用户设置"""
        try:
            self.settings.setValue('geometry', self.saveGeometry())
            self.settings.setValue('recent_files', self.recent_files[:10])  # 保存最近10个文件
            self.settings.setValue('default_model', self.model_selector.currentText())
        except Exception as e:
            logger.error(f"设置保存失败: {e}")

    def setup_shortcuts(self):
        """设置快捷键"""
        from PyQt5.QtWidgets import QShortcut
        
        # 文件操作
        QShortcut(QKeySequence("Ctrl+O"), self, self.upload_image)
        QShortcut(QKeySequence("Ctrl+B"), self, self.batch_upload)
        QShortcut(QKeySequence("Ctrl+R"), self, self.process_image)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_results)
        QShortcut(QKeySequence("Ctrl+Q"), self, self.close)
        
        # 窗口操作
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+1"), self, lambda: self.main_tabs.setCurrentIndex(0))
        QShortcut(QKeySequence("Ctrl+2"), self, lambda: self.main_tabs.setCurrentIndex(1))
        QShortcut(QKeySequence("Ctrl+3"), self, lambda: self.main_tabs.setCurrentIndex(2))
        
        # 其他功能
        QShortcut(QKeySequence("F5"), self, self.refresh_interface)
        QShortcut(QKeySequence("Ctrl+Plus"), self, self.increase_confidence)
        QShortcut(QKeySequence("Ctrl+Minus"), self, self.decrease_confidence)

    def setup_context_menus(self):
        """设置右键菜单"""
        # 批量列表右键菜单
        self.batch_list_menu = QMenu(self)
        self.batch_list_menu.addAction("打开图像", self.open_selected_image)
        self.batch_list_menu.addAction("删除图像", self.remove_selected_image)
        self.batch_list_menu.addSeparator()
        self.batch_list_menu.addAction("清空列表", self.clear_batch_list)
        
        # 结果表格右键菜单
        self.result_table_menu = QMenu(self)
        self.result_table_menu.addAction("复制结果", self.copy_results)
        self.result_table_menu.addAction("导出结果", self.export_results)

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def increase_confidence(self):
        """增加置信度"""
        current = self.confidence_slider.value()
        self.confidence_slider.setValue(min(100, current + 5))

    def decrease_confidence(self):
        """减少置信度"""
        current = self.confidence_slider.value()
        self.confidence_slider.setValue(max(0, current - 5))

    def refresh_interface(self):
        """刷新界面"""
        self.status_bar.showMessage("界面已刷新", 2000)
        self.update_status()

    def save_results(self):
        """保存结果"""
        if not hasattr(self, 'last_results') or not self.last_results:
            QMessageBox.information(self, "提示", "没有可保存的结果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", f"分类结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "文本文件 (*.txt);;CSV文件 (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"AI图像分类结果\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"模型: {self.model_selector.currentText()}\n")
                    f.write(f"置信度阈值: {self.confidence_slider.value()}%\n\n")
                    
                    for category, confidence in self.last_results.items():
                        f.write(f"{category}: {confidence:.4f}\n")
                
                self.status_bar.showMessage(f"结果已保存到: {file_path}", 3000)
                QMessageBox.information(self, "成功", f"结果已保存到:\n{file_path}")
                
            except Exception as e:
                logger.error(f"保存结果失败: {e}")
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def copy_results(self):
        """复制结果到剪贴板"""
        if not hasattr(self, 'last_results') or not self.last_results:
            return
        
        try:
            result_text = "\n".join([f"{cat}: {conf:.4f}" for cat, conf in self.last_results.items()])
            QApplication.clipboard().setText(result_text)
            self.status_bar.showMessage("结果已复制到剪贴板", 2000)
        except Exception as e:
            logger.error(f"复制失败: {e}")

    def export_results(self):
        """导出结果（更详细）"""
        self.save_results()

    def open_selected_image(self):
        """打开选中的图像"""
        current_item = self.batch_list.currentItem()
        if current_item:
            file_path = current_item.data(Qt.UserRole)
            if file_path:
                dialog = ImageZoomDialog(file_path, self)
                dialog.exec_()

    def remove_selected_image(self):
        """删除选中的图像"""
        current_row = self.batch_list.currentRow()
        if current_row >= 0:
            self.batch_list.takeItem(current_row)
            if current_row < len(self.batch_files):
                del self.batch_files[current_row]
            self.status_bar.showMessage("图像已从列表中移除", 2000)

    def clear_batch_list(self):
        """清空批量列表"""
        reply = QMessageBox.question(self, "确认", "确定要清空批量列表吗？",
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.batch_list.clear()
            self.batch_files.clear()
            self.batch_results.clear()
            self.status_bar.showMessage("批量列表已清空", 2000)

    def add_to_recent_files(self, file_path):
        """添加到最近文件"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:10]  # 保持最近10个文件

    def update_status(self):
        """更新状态栏信息"""
        try:
            memory_usage = self.memory_manager.get_memory_usage()
            model_name = self.model_worker.current_model_name or "未加载"
            
            status_text = f"模型: {model_name} | 内存: {memory_usage:.1f}MB"
            
            if hasattr(self, 'batch_files') and self.batch_files:
                status_text += f" | 批量: {len(self.batch_files)}张"
            
            self.status_bar.showMessage(status_text)
            
        except Exception as e:
            logger.error(f"状态更新失败: {e}")

    def setup_ui(self):
        self.title_bar = TitleBar(self)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(self.title_bar)

        # 主内容区域
        main_content = QSplitter(Qt.Horizontal)  # 使用分割器
        main_content.setChildrenCollapsible(False)

        # 左侧控制面板
        control_panel = self.create_control_panel()
        control_panel.setMinimumWidth(320)
        control_panel.setMaximumWidth(450)
        
        # 右侧工作区
        work_area = self.create_work_area()
        
        main_content.addWidget(control_panel)
        main_content.addWidget(work_area)
        main_content.setStretchFactor(0, 0)
        main_content.setStretchFactor(1, 1)
        main_content.setSizes([350, 1250])  # 设置初始大小比例

        central_layout.addWidget(main_content)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统启动中...")
        
        # 状态栏添加额外信息
        self.memory_label = QLabel("内存: 0MB")
        self.memory_label.setStyleSheet("color: #605E5C; font-family: 'Consolas';")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        self.fps_label = QLabel("就绪")
        self.fps_label.setStyleSheet("color: #107C10; font-family: 'Consolas';")
        self.status_bar.addPermanentWidget(self.fps_label)

        # 设置遮罩层（已废弃，改为非阻塞处理）
        # self.overlay = QWidget(self)
        # self.overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150);")
        # self.overlay.hide()
        # self.overlay.setGeometry(self.rect())

    def create_control_panel(self):
        """创建控制面板"""
        control_panel = QFrame()
        control_panel.setFrameStyle(QFrame.StyledPanel)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_layout.setSpacing(20)

        # 模型设置组
        model_group = self.create_model_group()
        control_layout.addWidget(model_group)

        # 参数设置组
        param_group = self.create_parameter_group()
        control_layout.addWidget(param_group)

        # 操作按钮组
        action_group = self.create_action_group()
        control_layout.addWidget(action_group)
        
        # 最近文件组
        recent_group = self.create_recent_files_group()
        control_layout.addWidget(recent_group)

        control_layout.addStretch()
        
        # 应用现代化样式
        control_panel.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F8F9FA, stop:1 #FFFFFF);
                border-right: 2px solid #0078D4;
                border-radius: 0px;
            }
        """)
        
        return control_panel

    def create_model_group(self):
        """创建模型设置组"""
        model_group = QGroupBox("🤖 模型设置")
        model_layout = QVBoxLayout(model_group)
        
        # 模型选择
        self.model_selector = QComboBox()
        self.model_selector.addItems(CONFIG["models"].keys())
        self.model_selector.setCursor(Qt.PointingHandCursor)
        self.model_selector.setToolTip("选择要使用的YOLO模型版本")
        
        # 模型信息显示
        self.model_info_label = QLabel("模型信息加载中...")
        self.model_info_label.setStyleSheet("color: #605E5C; font-size: 11px;")
        self.model_info_label.setWordWrap(True)
        
        model_layout.addWidget(QLabel("模型版本:"))
        model_layout.addWidget(self.model_selector)
        model_layout.addWidget(self.model_info_label)
        
        return model_group

    def create_parameter_group(self):
        """创建参数设置组"""
        param_group = QGroupBox("⚙️ 处理参数")
        param_layout = QVBoxLayout(param_group)
        
        # 置信度设置
        confidence_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setToolTip("设置分类结果的最低置信度阈值")
        
        self.confidence_spinbox = QSpinBox()
        self.confidence_spinbox.setRange(0, 100)
        self.confidence_spinbox.setValue(50)
        self.confidence_spinbox.setSuffix("%")
        self.confidence_spinbox.setFixedWidth(70)
        
        # 同步滑块和数字框
        self.confidence_slider.valueChanged.connect(self.confidence_spinbox.setValue)
        self.confidence_spinbox.valueChanged.connect(self.confidence_slider.setValue)
        
        self.confidence_label = QLabel("置信度阈值:")
        
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_spinbox)
        
        # 其他选项
        self.visual_toggle = QCheckBox("启用可视化")
        self.visual_toggle.setChecked(True)
        self.visual_toggle.setToolTip("生成包含分类结果的可视化图像")
        
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 50)
        self.batch_size_spinbox.setValue(10)
        self.batch_size_spinbox.setToolTip("批量处理时的批次大小")
        
        param_layout.addWidget(self.confidence_label)
        param_layout.addLayout(confidence_layout)
        param_layout.addWidget(self.visual_toggle)
        param_layout.addWidget(QLabel("批处理大小:"))
        param_layout.addWidget(self.batch_size_spinbox)
        
        return param_group

    def create_action_group(self):
        """创建操作按钮组"""
        btn_group = QGroupBox("🚀 操作")
        btn_layout = QVBoxLayout(btn_group)
        
        # 主要操作按钮
        self.btn_upload = QPushButton("📤 上传单图 (Ctrl+O)")
        self.btn_batch = QPushButton("📁 批量上传 (Ctrl+B)")
        self.btn_process = QPushButton("🚀 开始处理 (Ctrl+R)")
        self.btn_save = QPushButton("💾 保存结果 (Ctrl+S)")
        
        buttons = [self.btn_upload, self.btn_batch, self.btn_process, self.btn_save]
        for btn in buttons:
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(40)
        
        # 连接信号
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_batch.clicked.connect(self.batch_upload)
        self.btn_process.clicked.connect(self.process_image)
        self.btn_save.clicked.connect(self.save_results)
        
        # 添加到布局
        for btn in buttons:
            btn_layout.addWidget(btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #D2D0CE;
                border-radius: 4px;
                text-align: center;
                background: #F8F9FA;
                color: #323130;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078D4, stop:1 #4F9EE8);
                border-radius: 3px;
            }
        """)
        
        btn_layout.addWidget(self.progress_bar)
        
        return btn_group

    def create_recent_files_group(self):
        """创建最近文件组"""
        recent_group = QGroupBox("📋 最近文件")
        recent_layout = QVBoxLayout(recent_group)
        
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(120)
        self.recent_list.itemDoubleClicked.connect(self.load_recent_file)
        
        recent_layout.addWidget(self.recent_list)
        self.update_recent_files_list()
        
        return recent_group

    def update_recent_files_list(self):
        """更新最近文件列表"""
        self.recent_list.clear()
        for file_path in self.recent_files:
            if Path(file_path).exists():
                item = QListWidgetItem(Path(file_path).name)
                item.setData(Qt.UserRole, file_path)
                item.setToolTip(file_path)
                self.recent_list.addItem(item)

    def load_recent_file(self, item):
        """加载最近文件"""
        file_path = item.data(Qt.UserRole)
        if file_path and Path(file_path).exists():
            # 清空之前的结果
            self.result_table.setRowCount(0)
            self.visual_label.clear()
            self.visual_label.setText("请点击处理按钮开始分析")
            
            # 清空图像处理缓存
            if hasattr(self.process_worker, 'image_cache'):
                self.process_worker.image_cache.clear()
                logger.info("已清空图像处理缓存")
            
            self.current_image = cv2.imread(file_path)
            self.current_image_path = file_path  # 保存当前图片路径
            if self.current_image is not None:
                self.update_drop_preview(file_path)
                self.status_bar.showMessage(f"已加载: {Path(file_path).name}", 3000)
                logger.info(f"从最近文件加载: {file_path}")
            else:
                QMessageBox.warning(self, "警告", "无法读取图像文件")
        else:
            QMessageBox.warning(self, "警告", "文件不存在或已被删除")
            self.recent_files.remove(file_path)
            self.update_recent_files_list()

    def create_work_area(self):
        """创建工作区域"""
        self.main_tabs = QTabWidget()
        self.main_tabs.setTabPosition(QTabWidget.North)
        
        # 创建各个标签页
        single_tab = self.create_single_mode_tab()
        batch_tab = self.create_batch_mode_tab()
        monitor_tab = self.create_monitor_tab()
        
        self.main_tabs.addTab(single_tab, "🖼️ 单图模式")
        self.main_tabs.addTab(batch_tab, "📁 批量模式")
        self.main_tabs.addTab(monitor_tab, "📊 系统监控")
        
        return self.main_tabs

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
        """拖拽放下事件"""
        self.dragLeaveEvent(event)  # 恢复样式
        
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        valid_files = []
        
        # 验证文件
        for file_path in files:
            is_valid, error_msg = FileValidator.validate_image_file(file_path)
            if is_valid:
                valid_files.append(file_path)
            else:
                logger.warning(f"跳过无效文件 {file_path}: {error_msg}")
        
        if len(valid_files) == 1:
            # 清空之前的结果
            self.result_table.setRowCount(0)
            self.visual_label.clear()
            self.visual_label.setText("请点击处理按钮开始分析")
            
            # 清空图像处理缓存
            if hasattr(self.process_worker, 'image_cache'):
                self.process_worker.image_cache.clear()
                logger.info("已清空图像处理缓存")
            
            self.current_image = cv2.imread(valid_files[0])
            self.current_image_path = valid_files[0]  # 保存当前图片路径
            if self.current_image is not None:
                self.update_drop_preview(valid_files[0])
                self.add_to_recent_files(valid_files[0])
                self.update_recent_files_list()
                self.status_bar.showMessage("已载入单张图像", 3000)
                logger.info(f"拖拽加载新图片: {valid_files[0]}")
                # 切换到单图模式
                self.main_tabs.setCurrentIndex(0)
            else:
                QMessageBox.warning(self, "警告", "无法读取图像文件")
                
        elif len(valid_files) > 1:
            logger.info(f"拖拽批量上传开始，有效文件数量: {len(valid_files)}")
            
            # 清空之前的数据
            self.batch_files = []
            self.batch_list.clear()
            self.batch_results.clear()
            
            # 去除重复文件
            unique_files = []
            seen_files = set()
            for file_path in valid_files:
                abs_path = os.path.abspath(file_path)
                if abs_path not in seen_files:
                    seen_files.add(abs_path)
                    unique_files.append(file_path)
                else:
                    logger.warning(f"发现重复文件，跳过: {file_path}")
            
            logger.info(f"去重后文件数量: {len(unique_files)}")
            self.batch_files = unique_files
            
            # 添加进度提示
            progress = QProgressDialog("正在加载缩略图...", "取消", 0, len(unique_files), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            for i, file_path in enumerate(unique_files):
                if progress.wasCanceled():
                    logger.info("用户取消了拖拽缩略图加载")
                    break
                    
                logger.info(f"拖拽添加缩略图 {i+1}/{len(unique_files)}: {file_path}")
                self.add_thumbnail(file_path)
                progress.setValue(i + 1)
                QApplication.processEvents()
            
            progress.close()
            
            logger.info(f"拖拽缩略图加载完成，列表中共有 {self.batch_list.count()} 个项目")
            self.update_batch_info()
            self.status_bar.showMessage(f"已载入 {len(unique_files)} 张图像", 3000)
            # 切换到批量模式
            self.main_tabs.setCurrentIndex(1)
            
        else:
            QMessageBox.warning(self, "错误", "没有找到有效的图像文件")

    def add_thumbnail(self, file_path):
        """添加缩略图"""
        try:
            img = cv2.imread(file_path)
            if img is None:
                logger.warning(f"无法读取图像: {file_path}")
                return
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            
            # 创建标准的缩略图
            pixmap = QPixmap.fromImage(qimg).scaled(
                120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # 简化缩略图处理，避免圆角绘制问题
            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap))
            item.setText(Path(file_path).name)
            item.setData(Qt.UserRole, file_path)
            item.setToolTip(f"文件: {Path(file_path).name}\n路径: {file_path}")
            
            self.batch_list.addItem(item)
            logger.info(f"已添加缩略图: {Path(file_path).name}")
            
        except Exception as e:
            logger.error(f"缩略图创建失败 {file_path}: {e}")

    def update_drop_preview(self, path):
        """更新拖拽预览"""
        try:
            img = cv2.imread(path)
            if img is None:
                return
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            
            # 创建适合显示区域的图像
            max_size = 600
            if max(h, w) > max_size:
                pixmap = QPixmap.fromImage(qimg).scaled(
                    max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            else:
                pixmap = QPixmap.fromImage(qimg)
            
            self.drop_area.setPixmap(pixmap)
            self.drop_area.setText("")
            
            # 显示图像信息
            file_size = Path(path).stat().st_size / 1024  # KB
            info_text = f"📁 {Path(path).name} | 📏 {w}×{h} | 💾 {file_size:.1f} KB"
            self.drop_area.setToolTip(info_text)
            
        except Exception as e:
            logger.error(f"预览更新失败: {e}")

    def upload_image(self):
        """上传图像"""
        file, _ = QFileDialog.getOpenFileName(
            self, "选择图像", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if file:
            # 验证文件
            is_valid, error_msg = FileValidator.validate_image_file(file)
            if not is_valid:
                QMessageBox.warning(self, "文件验证失败", error_msg)
                return
            
            # 清空之前的结果
            self.result_table.setRowCount(0)
            self.visual_label.clear()
            self.visual_label.setText("请点击处理按钮开始分析")
            
            # 清空图像处理缓存
            if hasattr(self.process_worker, 'image_cache'):
                self.process_worker.image_cache.clear()
                logger.info("已清空图像处理缓存")
            
            self.current_image = cv2.imread(file)
            self.current_image_path = file  # 保存当前图片路径
            if self.current_image is not None:
                self.update_drop_preview(file)
                self.add_to_recent_files(file)
                self.update_recent_files_list()
                self.status_bar.showMessage(f"已加载: {Path(file).name}", 3000)
                logger.info(f"已加载新图片: {file}")
            else:
                QMessageBox.warning(self, "警告", "无法读取图像文件")

    def batch_upload(self):
        """批量上传"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图像", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if files:
            logger.info(f"批量上传开始，选择了 {len(files)} 个文件")
            
            # 验证文件
            valid_files = []
            invalid_count = 0
            
            for file_path in files:
                is_valid, error_msg = FileValidator.validate_image_file(file_path)
                if is_valid:
                    valid_files.append(file_path)
                else:
                    invalid_count += 1
                    logger.warning(f"跳过无效文件 {file_path}: {error_msg}")
            
            logger.info(f"验证后有效文件数量: {len(valid_files)}")
            
            if invalid_count > 0:
                QMessageBox.information(
                    self, "文件验证", 
                    f"跳过了 {invalid_count} 个无效文件，成功加载 {len(valid_files)} 个文件"
                )
            
            if valid_files:
                # 清空之前的数据
                logger.info("清空之前的批量数据")
                self.batch_files = []
                self.batch_list.clear()
                self.batch_results.clear()
                
                # 去除重复文件
                unique_files = []
                seen_files = set()
                for file_path in valid_files:
                    abs_path = os.path.abspath(file_path)
                    if abs_path not in seen_files:
                        seen_files.add(abs_path)
                        unique_files.append(file_path)
                    else:
                        logger.warning(f"发现重复文件，跳过: {file_path}")
                
                logger.info(f"去重后文件数量: {len(unique_files)}")
                self.batch_files = unique_files
                
                # 添加进度提示
                progress = QProgressDialog("正在加载缩略图...", "取消", 0, len(unique_files), self)
                progress.setWindowModality(Qt.WindowModal)
                progress.show()
                
                for i, file_path in enumerate(unique_files):
                    if progress.wasCanceled():
                        logger.info("用户取消了缩略图加载")
                        break
                        
                    logger.info(f"添加缩略图 {i+1}/{len(unique_files)}: {file_path}")
                    self.add_thumbnail(file_path)
                    progress.setValue(i + 1)
                    QApplication.processEvents()
                
                progress.close()
                
                logger.info(f"缩略图加载完成，列表中共有 {self.batch_list.count()} 个项目")
                self.update_batch_info()
                self.status_bar.showMessage(f"已选择 {len(unique_files)} 张图像", 3000)
                # 切换到批量模式
                self.main_tabs.setCurrentIndex(1)

    def process_image(self):
        """智能处理图像 - 根据当前标签页选择处理模式"""
        
        # 检查是否正在处理
        if self.is_processing:
            QMessageBox.information(
                self, "处理中", 
                "后台正在处理图像，请等待处理完成后再操作。"
            )
            return
        
        current_tab_index = self.main_tabs.currentIndex()
        
        if current_tab_index == 0:  # 单图模式
            self._process_single_image()
        elif current_tab_index == 1:  # 批量模式
            self._process_batch_images()
        else:
            QMessageBox.information(self, "提示", "请切换到单图模式或批量模式进行处理")
    
    def _process_single_image(self):
        """处理单张图像"""
        if self.current_image is None:
            QMessageBox.information(self, "提示", "请先上传单张图像")
            return
        
        # 清空之前的结果
        self.result_table.setRowCount(0)
        self.visual_label.clear()
        self.visual_label.setText("正在处理...")
        
        model_name = self.model_selector.currentText()
        conf = self.confidence_slider.value() / 100
        enable_vis = self.visual_toggle.isChecked()
        
        # 获取当前图片路径，确保缓存键唯一
        img_path = getattr(self, 'current_image_path', '')
        logger.info(f"处理单图，路径: {img_path}")
        
        # 设置处理状态
        self._start_processing("正在处理单张图像...")
        
        # 开始处理，传递正确的图片路径
        self.process_worker.process_image(self.current_image, model_name, conf, enable_vis, img_path)
    
    def _process_batch_images(self):
        """处理批量图像"""
        if not self.batch_files:
            QMessageBox.information(self, "提示", "请先上传批量图像")
            return
        
        # 检查是否有选中的图像（如果有选中，只处理选中的）
        selected_items = self.batch_list.selectedItems()
        if selected_items:
            # 处理选中的图像
            selected_files = []
            for item in selected_items:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    selected_files.append(file_path)
            
            if not selected_files:
                QMessageBox.warning(self, "警告", "选中的项目中没有有效的图像文件")
                return
            
            reply = QMessageBox.question(
                self, "确认处理", 
                f"检测到您选中了 {len(selected_files)} 张图像，是否只处理选中的图像？\n\n"
                f"点击 Yes 处理选中的 {len(selected_files)} 张\n"
                f"点击 No 处理全部 {len(self.batch_files)} 张",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Cancel:
                return
            elif reply == QMessageBox.Yes:
                files_to_process = selected_files
            else:
                files_to_process = self.batch_files
        else:
            files_to_process = self.batch_files
        
        logger.info(f"开始批量处理: {len(files_to_process)} 张图像")
        
        # *** 重要：清空之前的批量结果，防止状态混乱 ***
        logger.info("清空之前的批量结果")
        self.batch_results = []
        
        model_name = self.model_selector.currentText()
        conf = self.confidence_slider.value() / 100
        enable_vis = self.visual_toggle.isChecked()
        
        logger.info(f"处理参数: 模型={model_name}, 置信度={conf}, 可视化={enable_vis}")
        
        # 设置处理状态
        self._start_processing(f"正在批量处理 {len(files_to_process)} 张图像...")
        
        # 设置批量进度条
        self.progress_bar.setRange(0, len(files_to_process))
        self.progress_bar.setValue(0)
        
        logger.info("调用process_worker.process_batch开始处理")
        
        # 开始批量处理
        self.process_worker.process_batch(files_to_process, model_name, conf, enable_vis)
    
    def _start_processing(self, message):
        """开始处理状态设置"""
        self.is_processing = True
        
        # 更新UI状态
        self.btn_process.setText("⏳ 处理中...")
        self.btn_process.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # 更新状态栏
        self.status_bar.showMessage(message)
        self.fps_label.setText("处理中")
        self.fps_label.setStyleSheet("color: #FF8C00; font-family: 'Consolas';")
        
        # 显示处理提示（非阻塞）
        self._show_processing_indicator()
    
    def _finish_processing(self):
        """完成处理状态设置"""
        self.is_processing = False
        
        # 恢复UI状态
        self.btn_process.setText("🚀 开始处理 (Ctrl+R)")
        self.btn_process.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # 更新状态栏
        self.status_bar.showMessage("处理完成", 3000)
        self.fps_label.setText("就绪")
        self.fps_label.setStyleSheet("color: #107C10; font-family: 'Consolas';")
        
        # 隐藏处理提示
        self._hide_processing_indicator()
    
    def _show_processing_indicator(self):
        """显示处理指示器（非阻塞）"""
        # 创建处理指示器标签
        if not hasattr(self, 'processing_indicator'):
            self.processing_indicator = QLabel(self)
            self.processing_indicator.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                        stop:0 #FF8C00, stop:1 #FFA500);
                    color: white;
                    border-radius: 20px;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 12px;
                }
            """)
            self.processing_indicator.setAlignment(Qt.AlignCenter)
        
        self.processing_indicator.setText("🔄 后台处理中，请稍候...")
        self.processing_indicator.adjustSize()
        
        # 定位到右上角
        parent_rect = self.rect()
        indicator_rect = self.processing_indicator.rect()
        x = parent_rect.width() - indicator_rect.width() - 20
        y = 50  # 标题栏下方
        self.processing_indicator.move(x, y)
        
        self.processing_indicator.show()
        self.processing_indicator.raise_()
    
    def _hide_processing_indicator(self):
        """隐藏处理指示器"""
        if hasattr(self, 'processing_indicator'):
            self.processing_indicator.hide()

    def on_model_selected(self, index):
        """模型选择变化"""
        model_name = self.model_selector.itemText(index)
        self.status_bar.showMessage(f"正在加载模型 {model_name}...")
        
        # 更新模型信息
        if model_name in CONFIG["models"]:
            model_info = CONFIG["models"][model_name]
            info_text = f"大小: {model_info.get('size', 'N/A')}MB\n描述: {model_info.get('description', '无描述')}"
            self.model_info_label.setText(info_text)
        
        self.model_worker.ensure_model_loaded(model_name)

    def update_download_progress(self, current, total):
        """更新下载进度"""
        if not hasattr(self, 'progress_dialog'):
            self.progress_dialog = QProgressDialog("下载模型...", "取消", 0, total, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.show()
        
        self.progress_dialog.setMaximum(total)
        self.progress_dialog.setValue(current)
        
        if current == total:
            self.progress_dialog.close()
            if hasattr(self, 'progress_dialog'):
                del self.progress_dialog

    def on_model_load_started(self):
        """模型开始加载"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setLabelText("加载模型中...")
            self.progress_dialog.setRange(0, 0)
        else:
            self.progress_dialog = QProgressDialog("加载模型中...", None, 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.show()

    def on_model_loaded(self, model_name):
        """模型加载完成"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            del self.progress_dialog
        
        self.status_bar.showMessage(f"模型 {model_name} 已加载", 3000)
        self.btn_process.setEnabled(True)
        self.fps_label.setText("模型就绪")
        self.fps_label.setStyleSheet("color: #107C10; font-family: 'Consolas';")

    def on_model_error(self, error):
        """模型加载错误"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            del self.progress_dialog
        
        QMessageBox.critical(self, "模型错误", f"模型加载失败:\n{error}")
        self.status_bar.showMessage(f"错误: {error}", 5000)
        self.btn_process.setEnabled(True)
        self.fps_label.setText("错误")
        self.fps_label.setStyleSheet("color: #D13438; font-family: 'Consolas';")

    def update_batch_progress(self, current, total):
        """更新批量处理进度"""
        if hasattr(self, 'progress_bar') and self.progress_bar.isVisible():
            self.progress_bar.setValue(current)
            
            # 更新状态信息
            self.status_bar.showMessage(f"正在批量处理... ({current}/{total})")
            
            if current == total:
                # 不在这里结束处理状态，等待处理完成信号
                pass

    def on_processing_finished(self, results, vis, img_path):
        """处理完成"""
        logger.info(f"处理完成回调: img_path='{img_path}', results_count={len(results) if results else 0}")
        
        # *** 关键修复：使用process_worker的is_batch_processing属性来判断模式 ***
        if hasattr(self.process_worker, 'is_batch_processing') and self.process_worker.is_batch_processing:
            # 批量模式
            logger.info(f"批量模式处理完成: {img_path}")
            
            # 添加到批量结果
            self.batch_results.append({
                'path': img_path,
                'results': results,
                'visualization': vis
            })
            logger.info(f"已添加到batch_results，当前总数: {len(self.batch_results)}")
            
            # 显示第一个结果
            if len(self.batch_results) == 1:
                item = self.batch_list.item(0)
                if item:
                    logger.info("显示第一个批量结果")
                    self.show_batch_detail(item)
            
            logger.info("批量单项处理完成，等待batch_all_finished信号")
            
            # *** 添加超时保护机制 - 如果5秒内没有收到batch_all_finished信号，强制恢复状态 ***
            QTimer.singleShot(5000, self._emergency_finish_processing)
                
        else:
            # 单图模式
            logger.info(f"单图模式处理完成: {img_path}")
            self.last_results = results  # 保存结果用于导出
            self.update_result_table(self.result_table, results)
            self.update_visualization(self.visual_label, vis)
            
            # 单图处理完成
            self._finish_processing()
            logger.info("单图模式UI状态已恢复")

    def _emergency_finish_processing(self):
        """紧急恢复处理状态 - 防止UI卡住"""
        if self.is_processing:
            logger.warning("⚠️ 检测到处理状态超时，强制恢复UI状态")
            self._finish_processing()
            self.status_bar.showMessage("⚠️ 处理完成（状态已自动恢复）", 3000)

    def update_result_table(self, table, results):
        """更新结果表格"""
        table.setRowCount(0)
        
        for i, (cls, score) in enumerate(results.items()):
            row_position = table.rowCount()
            table.insertRow(row_position)
            
            # 添加类别
            class_item = QTableWidgetItem(cls)
            class_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            table.setItem(row_position, 0, class_item)
            
            # 添加置信度
            score_item = QTableWidgetItem(f"{score:.4f}")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(row_position, 1, score_item)
            
            # 高亮最高置信度
            if i == 0:
                class_item.setBackground(QColor("#0078D4"))
                class_item.setForeground(QColor("white"))
                score_item.setBackground(QColor("#0078D4"))
                score_item.setForeground(QColor("white"))

    def update_visualization(self, label, vis):
        """更新可视化显示"""
        if vis is not None:
            try:
                vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                h, w = vis_rgb.shape[:2]
                qimg = QImage(vis_rgb.data, w, h, w * 3, QImage.Format_RGB888)
                
                # 适应标签大小
                label_size = label.size()
                pixmap = QPixmap.fromImage(qimg).scaled(
                    label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                label.setPixmap(pixmap)
            except Exception as e:
                logger.error(f"可视化更新失败: {e}")
                label.setText("可视化显示失败")
        else:
            label.clear()
            label.setText("无可视化结果")

    def on_processing_error(self, error):
        """处理错误"""
        self.status_bar.showMessage(f"处理错误: {error}", 5000)
        QMessageBox.critical(self, "处理错误", f"图像处理失败:\n{error}")
        
        # 错误时也要恢复处理状态
        self._finish_processing()
        
        self.fps_label.setText("错误")
        self.fps_label.setStyleSheet("color: #D13438; font-family: 'Consolas';")

    def on_processing_progress(self, progress_text):
        """处理进度更新"""
        logger.info(f"处理进度: {progress_text}")
        self.status_bar.showMessage(progress_text, 1000)
        
        # 强制刷新UI
        QApplication.processEvents()

    def on_classification_ready(self, results, img_path):
        """分类结果就绪 - 优先显示"""
        logger.info(f"分类结果就绪回调: img_path='{img_path}', results_count={len(results) if results else 0}")
        
        # *** 修复：使用is_batch_processing属性来判断模式 ***
        if hasattr(self.process_worker, 'is_batch_processing') and self.process_worker.is_batch_processing:
            # 批量模式
            logger.info(f"批量模式分类结果就绪: {img_path}")
            # 批量模式的处理逻辑保持不变，等待完整结果
            return
        else:
            # 单图模式
            logger.info("单图模式分类结果就绪，立即更新结果表格")
            
            # 保存结果用于导出
            self.last_results = results
            
            # 立即更新结果表格
            self.update_result_table(self.result_table, results)
            
            # 强制刷新UI
            QApplication.processEvents()
            
            # 显示快速反馈 - 告知用户分类完成，下一步绘制可视化
            if results:
                top_result = next(iter(results.items()))
                self.status_bar.showMessage(f"✅ 分类完成 - {top_result[0]} ({top_result[1]:.3f}) | 🎨 正在绘制可视化结果...", 3000)
            
            logger.info("分类结果表格更新完成")

    def on_visualization_ready(self, visualization, img_path):
        """可视化结果就绪"""
        logger.info(f"可视化结果就绪回调: img_path='{img_path}', has_vis={visualization is not None}")
        
        # *** 修复：使用is_batch_processing属性来判断模式 ***
        if hasattr(self.process_worker, 'is_batch_processing') and self.process_worker.is_batch_processing:
            # 批量模式
            logger.info(f"批量模式可视化结果就绪: {img_path}")
            # 批量模式的处理逻辑保持不变
            return
        else:
            # 单图模式
            logger.info("单图模式可视化结果就绪，更新可视化显示")
            
            # 更新可视化显示
            if visualization is not None:
                self.update_visualization(self.visual_label, visualization)
                logger.info("可视化显示更新完成")
                status_msg = "🎨 可视化绘制完成"
            else:
                self.visual_label.clear()
                self.visual_label.setText("🚫 跳过可视化生成")
                logger.info("显示无可视化结果")
                status_msg = "⚡ 处理完成（跳过可视化）"
            
            # 强制刷新UI
            QApplication.processEvents()
            
            # 显示最终完成状态
            if hasattr(self, 'last_results') and self.last_results:
                top_result = next(iter(self.last_results.items()))
                self.status_bar.showMessage(f"✅ 全部完成 - {top_result[0]} ({top_result[1]:.3f}) | {status_msg}", 5000)
            
            # 完成处理状态
            self._finish_processing()
            
            logger.info("可视化更新完成")

    def on_batch_item_finished(self, results, vis, img_path, batch_index, total_batch):
        """批量单项完成处理"""
        logger.info(f"批量单项完成: {img_path} ({batch_index}/{total_batch})")
        
        # 添加到批量结果
        self.batch_results.append({
            'path': img_path,
            'results': results,
            'visualization': vis,
            'index': batch_index - 1
        })
        
        logger.info(f"当前batch_results长度: {len(self.batch_results)}")
        logger.info(f"当前batch_list项目数: {self.batch_list.count()}")
        logger.info(f"当前batch_files长度: {len(self.batch_files)}")
        
        # 显示第一个结果
        if batch_index == 1:
            item = self.batch_list.item(0)
            if item:
                logger.info(f"显示第一个批量结果: {item.data(Qt.UserRole)}")
                self.show_batch_detail(item)
        
        # 更新状态
        self.status_bar.showMessage(f"批量处理进度: {batch_index}/{total_batch} - {Path(img_path).name}")

    def on_batch_all_finished(self):
        """批量全部完成处理"""
        logger.info("=== 批量处理全部完成回调 ===")
        
        # 强制完成处理状态
        logger.info("强制恢复处理状态...")
        self._finish_processing()
        
        # 显示完成信息
        total_count = len(self.batch_results)
        success_count = len([r for r in self.batch_results if r.get('results')])
        
        logger.info(f"批量处理统计: 总计={total_count}, 成功={success_count}")
        
        completion_msg = f"批量处理完成: 成功 {success_count}/{total_count}"
        self.status_bar.showMessage(completion_msg, 5000)
        logger.info(f"状态栏已更新: {completion_msg}")
        
        # 强制刷新UI
        QApplication.processEvents()
        
        # 显示完成对话框（如果有多张图片）
        if total_count > 1:
            QMessageBox.information(
                self, "批量处理完成", 
                f"批量处理已完成！\n\n"
                f"总计: {total_count} 张图像\n"
                f"成功: {success_count} 张\n"
                f"失败: {total_count - success_count} 张"
            )
        
        logger.info("=== 批量处理完成回调结束 ===")

    def on_detailed_info_ready(self, detailed_info, img_path):
        """详细信息就绪处理"""
        try:
            logger.info(f"详细信息就绪: {img_path}")
            
            # 构建详细信息显示文本
            info_lines = []
            
            # 基本信息
            info_lines.append(f"图像: {Path(img_path).name if img_path else '当前图像'}")
            info_lines.append(f"模型: {detailed_info.get('model_name', 'Unknown')}")
            info_lines.append(f"原始尺寸: {detailed_info.get('image_shape', 'Unknown')}")
            info_lines.append(f"处理尺寸: {detailed_info.get('processed_shape', 'Unknown')}")
            
            # 推理时间信息
            inference_time = detailed_info.get('inference_time_ms', 0)
            info_lines.append(f"推理时间: {inference_time:.1f}ms")
            
            # 前5个最高置信度结果
            all_results = detailed_info.get('all_results', {})
            if all_results:
                sorted_all = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:5]
                info_lines.append("\n🎯 Top 5 分类结果:")
                for i, (cls, conf) in enumerate(sorted_all):
                    info_lines.append(f"{i}: {cls} {conf:.3f}")
            
            # 更新详细信息显示
            info_text = "\n".join(info_lines)
            
            # 根据模式更新不同的显示区域
            if img_path and img_path.strip():  # 批量模式
                # 批量模式的详细信息可以显示在批量结果区域
                logger.info(f"批量模式详细信息: {info_text[:100]}...")
            else:  # 单图模式
                # 单图模式在状态栏显示简化信息
                top_result = detailed_info.get('top_result', ("Unknown", 0.0))
                summary = f"推理: {inference_time:.1f}ms | {top_result[0]} ({top_result[1]:.3f})"
                self.status_bar.showMessage(summary, 8000)
                logger.info(f"详细信息显示在状态栏: {summary}")
                    
        except Exception as e:
            logger.error(f"详细信息处理失败: {e}")
            # 不要让详细信息错误影响主流程

    def on_monitor_data_ready(self, monitor_data, text_info):
        """监控数据准备就绪"""
        try:
            # 清除旧的监控组件
            for i in reversed(range(self.monitor_layout.count())):
                child = self.monitor_layout.itemAt(i).widget()
                if child:
                    child.setParent(None)
            
            # 添加文本信息
            info_label = QLabel(text_info)
            info_label.setStyleSheet("""
                QLabel {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #F8F9FA, stop:1 #FFFFFF);
                    border: 1px solid #D2D0CE;
                    border-radius: 8px;
                    padding: 15px;
                    color: #323130;
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                }
            """)
            info_label.setWordWrap(True)
            self.monitor_layout.addWidget(info_label)
            
            # 添加图表
            if "charts" in monitor_data:
                charts_pixmap = QPixmap()
                if charts_pixmap.loadFromData(monitor_data["charts"]):
                    charts_label = QLabel()
                    charts_label.setPixmap(charts_pixmap)
                    charts_label.setAlignment(Qt.AlignCenter)
                    charts_label.setStyleSheet("""
                        QLabel {
                            background: #FFFFFF;
                            border: 1px solid #D2D0CE;
                            border-radius: 8px;
                            padding: 10px;
                        }
                    """)
                    self.monitor_layout.addWidget(charts_label)
            
            # 更新状态栏内存信息
            memory_usage = self.memory_manager.get_memory_usage()
            self.memory_label.setText(f"内存: {memory_usage:.1f}MB")
            
        except Exception as e:
            logger.error(f"监控数据处理失败: {e}")

    def show_batch_detail(self, item):
        """显示批量详情"""
        if not item:
            return
        
        # 获取项目的实际文件路径
        file_path = item.data(Qt.UserRole)
        if not file_path:
            logger.warning("缩略图项目没有关联的文件路径")
            return
        
        # 在batch_results中查找对应的结果
        result_data = None
        for result in self.batch_results:
            if result.get('path') == file_path:
                result_data = result
                break
        
        if result_data:
            logger.info(f"显示批量详情: {file_path}")
            self.update_result_table(self.batch_result_table, result_data['results'])
            self.update_visualization(self.batch_visual_label, result_data['visualization'])
        else:
            # 没有找到对应的处理结果
            logger.warning(f"未找到文件 {file_path} 的处理结果")
            self.batch_result_table.setRowCount(0)
            self.batch_visual_label.clear()
            self.batch_visual_label.setText("等待处理...")
            
            # 显示文件信息作为占位符
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    h, w = img.shape[:2]
                    file_size = Path(file_path).stat().st_size / 1024  # KB
                    placeholder_text = f"文件: {Path(file_path).name}\n尺寸: {w}×{h}\n大小: {file_size:.1f} KB\n\n等待处理..."
                    self.batch_visual_label.setText(placeholder_text)
            except Exception as e:
                logger.error(f"读取文件信息失败: {e}")

    def closeEvent(self, event):
        """关闭事件"""
        try:
            # 保存设置
            self.save_settings()
            
            # 停止所有线程
            self.monitor_thread.quit()
            self.monitor_thread.wait(3000)
            
            self.model_thread.quit()
            self.model_thread.wait(3000)
            
            self.process_thread.quit()
            self.process_thread.wait(3000)
            
            logger.info("应用程序正常关闭")
            event.accept()
            
        except Exception as e:
            logger.error(f"关闭时发生错误: {e}")
            event.accept()

    def create_single_mode_tab(self):
        """创建单图模式标签页"""
        single_tab = QWidget()
        single_layout = QVBoxLayout(single_tab)
        single_layout.setSpacing(15)
        
        # 拖拽上传区域
        self.drop_area = QLabel("🖼️ 拖放图像文件到这里或点击上传单图按钮")
        self.drop_area.setObjectName("drop_area")
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.drop_area.setMinimumHeight(200)
        self.drop_area.setCursor(Qt.PointingHandCursor)
        self.drop_area.mousePressEvent = lambda e: self.upload_image()
        
        # 添加更多视觉效果
        self.drop_area.setStyleSheet("""
            QLabel#drop_area {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F8F9FA, stop:0.5 #FFFFFF, stop:1 #F8F9FA);
                border: 2px dashed #0078D4;
                border-radius: 15px;
                color: #605E5C;
                font-size: 16px;
                font-weight: bold;
                padding: 20px;
            }
            QLabel#drop_area:hover {
                border-color: #4F9EE8;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F0F0F0, stop:0.5 #F8F9FA, stop:1 #F0F0F0);
                color: #323130;
            }
        """)
        
        single_layout.addWidget(self.drop_area)

        # 处理结果区域
        result_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧：结果表格
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        
        # 分类结果表格
        result_table_group = QGroupBox("🎯 分类结果")
        result_table_layout = QVBoxLayout(result_table_group)
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["类别", "置信度"])
        self.result_table.setRowCount(0)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.result_table.customContextMenuRequested.connect(
            lambda pos: self.result_table_menu.exec_(self.result_table.mapToGlobal(pos))
        )
        
        # 设置表格样式
        self.result_table.setStyleSheet("""
            QTableWidget {
                background-color: #FFFFFF;
                alternate-background-color: #F8F9FA;
                color: #000000;
                gridline-color: #EDEBE9;
                border: 1px solid #D2D0CE;
                border-radius: 8px;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #F0F0F0, stop:1 #E5E5E5);
                color: #000000;
                border: none;
                border-right: 1px solid #D2D0CE;
                border-bottom: 1px solid #D2D0CE;
                padding: 8px;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #EDEBE9;
            }
            QTableWidget::item:selected {
                background-color: #0078D4;
                color: white;
            }
        """)
        
        self.result_table.horizontalHeader().setStretchLastSection(True)
        result_table_layout.addWidget(self.result_table)
        result_layout.addWidget(result_table_group)
        
        # 右侧：可视化显示
        visual_widget = QWidget()
        visual_layout = QVBoxLayout(visual_widget)
        visual_layout.addWidget(QLabel("👁️ 可视化结果"))
        
        self.visual_label = QLabel()
        self.visual_label.setObjectName("visual_label")
        self.visual_label.setAlignment(Qt.AlignCenter)
        self.visual_label.setMinimumSize(400, 400)
        self.visual_label.setStyleSheet("""
            QLabel#visual_label {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #FFFFFF, stop:1 #F8F9FA);
                border: 2px solid #D2D0CE;
                border-radius: 10px;
                color: #323130;
            }
        """)
        
        visual_layout.addWidget(self.visual_label)
        
        # 添加到分割器
        result_splitter.addWidget(result_widget)
        result_splitter.addWidget(visual_widget)
        result_splitter.setSizes([400, 600])
        
        single_layout.addWidget(result_splitter)
        
        return single_tab

    def create_batch_mode_tab(self):
        """创建批量模式标签页"""
        batch_tab = QWidget()
        batch_layout = QVBoxLayout(batch_tab)
        batch_layout.setSpacing(15)

        # 批量操作工具栏
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        
        # 添加工具按钮
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(self.select_all_batch_items)
        
        clear_selection_btn = QPushButton("取消选择")
        clear_selection_btn.clicked.connect(self.clear_batch_selection)
        
        remove_selected_btn = QPushButton("删除选中")
        remove_selected_btn.clicked.connect(self.remove_selected_batch_items)
        
        toolbar_layout.addWidget(QLabel("批量操作:"))
        toolbar_layout.addWidget(select_all_btn)
        toolbar_layout.addWidget(clear_selection_btn)
        toolbar_layout.addWidget(remove_selected_btn)
        toolbar_layout.addStretch()
        
        # 批量信息
        self.batch_info_label = QLabel("未选择文件")
        self.batch_info_label.setStyleSheet("color: #605E5C; font-weight: bold;")
        toolbar_layout.addWidget(self.batch_info_label)
        
        batch_layout.addWidget(toolbar)

        # 缩略图预览区域
        thumbnail_group = QGroupBox("📁 缩略图预览")
        thumbnail_layout = QVBoxLayout(thumbnail_group)
        
        self.batch_list = QListWidget()
        self.batch_list.setViewMode(QListWidget.IconMode)
        self.batch_list.setIconSize(QSize(120, 120))
        self.batch_list.setResizeMode(QListWidget.Adjust)
        self.batch_list.setSpacing(15)
        
        # 设置选择模式为多选
        self.batch_list.setSelectionMode(QListWidget.ExtendedSelection)
        
        self.batch_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.batch_list.customContextMenuRequested.connect(
            lambda pos: self.batch_list_menu.exec_(self.batch_list.mapToGlobal(pos))
        )
        self.batch_list.itemClicked.connect(self.show_batch_detail)
        self.batch_list.itemDoubleClicked.connect(self.show_zoomed_image)
        self.batch_list.itemSelectionChanged.connect(self.update_batch_info)
        
        # 设置批量列表样式
        self.batch_list.setStyleSheet("""
            QListWidget {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:1 #F8F9FA);
                border: 1px solid #D2D0CE;
                border-radius: 8px;
                padding: 10px;
            }
            QListWidget::item {
                background-color: #F8F9FA;
                border: 2px solid #EDEBE9;
                border-radius: 8px;
                margin: 5px;
                padding: 5px;
            }
            QListWidget::item:hover {
                border-color: #0078D4;
                background-color: #F0F0F0;
            }
            QListWidget::item:selected {
                border-color: #4F9EE8;
                background-color: #0078D4;
            }
        """)
        
        thumbnail_layout.addWidget(self.batch_list)
        batch_layout.addWidget(thumbnail_group)

        # 详细结果视图
        batch_result_splitter = QSplitter(Qt.Horizontal)
        
        # 批量结果表格
        batch_result_widget = QWidget()
        batch_result_layout = QVBoxLayout(batch_result_widget)
        batch_result_layout.addWidget(QLabel("📊 处理结果"))
        
        self.batch_result_table = QTableWidget()
        self.batch_result_table.setColumnCount(2)
        self.batch_result_table.setHorizontalHeaderLabels(["类别", "置信度"])
        self.batch_result_table.setRowCount(0)
        self.batch_result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_result_table.setStyleSheet(self.result_table.styleSheet())
        self.batch_result_table.horizontalHeader().setStretchLastSection(True)
        
        batch_result_layout.addWidget(self.batch_result_table)
        
        # 批量可视化
        batch_visual_widget = QWidget()
        batch_visual_layout = QVBoxLayout(batch_visual_widget)
        batch_visual_layout.addWidget(QLabel("🔍 详细视图"))
        
        self.batch_visual_label = QLabel()
        self.batch_visual_label.setObjectName("batch_visual_label")
        self.batch_visual_label.setAlignment(Qt.AlignCenter)
        self.batch_visual_label.setMinimumSize(400, 400)
        self.batch_visual_label.setStyleSheet(self.visual_label.styleSheet())
        
        batch_visual_layout.addWidget(self.batch_visual_label)
        
        batch_result_splitter.addWidget(batch_result_widget)
        batch_result_splitter.addWidget(batch_visual_widget)
        batch_result_splitter.setSizes([400, 600])
        
        batch_layout.addWidget(batch_result_splitter)

        return batch_tab

    def create_monitor_tab(self):
        """创建系统监控标签页"""
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        
        # 监控控制栏
        control_bar = QWidget()
        control_layout = QHBoxLayout(control_bar)
        
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.refresh_monitor)
        
        clear_btn = QPushButton("🧹 清空")
        clear_btn.clicked.connect(self.clear_monitor)
        
        auto_refresh_cb = QCheckBox("自动刷新")
        auto_refresh_cb.setChecked(True)
        auto_refresh_cb.toggled.connect(self.toggle_auto_refresh)
        
        control_layout.addWidget(refresh_btn)
        control_layout.addWidget(clear_btn)
        control_layout.addWidget(auto_refresh_cb)
        control_layout.addStretch()
        
        # 系统状态指示器
        self.status_indicator = QLabel("🟢 系统正常")
        self.status_indicator.setStyleSheet("color: #107C10; font-weight: bold;")
        control_layout.addWidget(self.status_indicator)
        
        monitor_layout.addWidget(control_bar)
        
        # 监控内容区域
        self.monitor_scroll = QScrollArea()
        self.monitor_scroll.setWidgetResizable(True)
        self.monitor_scroll.setStyleSheet("""
            QScrollArea {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #FFFFFF, stop:1 #F8F9FA);
                border: 1px solid #D2D0CE;
                border-radius: 8px;
            }
        """)
        
        self.monitor_widget = QWidget()
        self.monitor_layout = QVBoxLayout(self.monitor_widget)
        self.monitor_scroll.setWidget(self.monitor_widget)
        monitor_layout.addWidget(self.monitor_scroll)

        return monitor_tab

    def select_all_batch_items(self):
        """选择所有批量项目"""
        # 暂时断开信号连接，避免频繁更新
        self.batch_list.itemSelectionChanged.disconnect(self.update_batch_info)
        
        # 使用selectAll方法更可靠
        self.batch_list.selectAll()
        
        # 重新连接信号
        self.batch_list.itemSelectionChanged.connect(self.update_batch_info)
        
        # 手动更新信息
        self.update_batch_info()

    def clear_batch_selection(self):
        """清除批量选择"""
        # 暂时断开信号连接
        self.batch_list.itemSelectionChanged.disconnect(self.update_batch_info)
        
        self.batch_list.clearSelection()
        
        # 重新连接信号
        self.batch_list.itemSelectionChanged.connect(self.update_batch_info)
        
        # 手动更新信息
        self.update_batch_info()

    def remove_selected_batch_items(self):
        """删除选中的批量项目"""
        selected_items = self.batch_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "提示", "请先选择要删除的项目")
            return
        
        reply = QMessageBox.question(
            self, "确认删除", 
            f"确定要删除 {len(selected_items)} 个选中的项目吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 收集要删除的文件路径
            files_to_remove = []
            for item in selected_items:
                file_path = item.data(Qt.UserRole)
                if file_path:
                    files_to_remove.append(file_path)
            
            # 从batch_files列表中删除对应文件
            for file_path in files_to_remove:
                if file_path in self.batch_files:
                    self.batch_files.remove(file_path)
            
            # 从UI中删除项目（逆序删除，避免索引问题）
            for item in sorted(selected_items, key=lambda x: self.batch_list.row(x), reverse=True):
                row = self.batch_list.row(item)
                self.batch_list.takeItem(row)
            
            # 清空对应的结果
            self.batch_results = [result for result in self.batch_results 
                                if result.get('path') not in files_to_remove]
            
            self.update_batch_info()
            self.status_bar.showMessage(f"已删除 {len(selected_items)} 个项目", 2000)
            
            # 清空详细视图（如果没有剩余项目）
            if self.batch_list.count() == 0:
                self.batch_result_table.setRowCount(0)
                self.batch_visual_label.clear()
                self.batch_visual_label.setText("无内容显示")

    def update_batch_info(self):
        """更新批量信息"""
        total = self.batch_list.count()
        selected = len(self.batch_list.selectedItems())
        
        # 验证一致性
        files_count = len(self.batch_files)
        if total != files_count:
            logger.warning(f"⚠️ 批量列表不一致！缩略图数量: {total}, 文件列表数量: {files_count}")
            
            # 尝试修复不一致
            if total > files_count:
                logger.info("尝试修复：缩略图多于文件列表")
                # 移除多余的缩略图
                while self.batch_list.count() > files_count:
                    self.batch_list.takeItem(self.batch_list.count() - 1)
                total = self.batch_list.count()
        
        logger.info(f"批量信息更新: 缩略图={total}, 文件={files_count}, 选中={selected}")
        
        if total == 0:
            self.batch_info_label.setText("未选择文件")
        else:
            self.batch_info_label.setText(f"总计: {total} | 选中: {selected}")
            
        # 验证每个缩略图都有对应的文件路径
        for i in range(self.batch_list.count()):
            item = self.batch_list.item(i)
            file_path = item.data(Qt.UserRole)
            if not file_path:
                logger.error(f"缩略图项目 {i} 没有关联的文件路径")
            elif i < len(self.batch_files) and file_path != self.batch_files[i]:
                logger.warning(f"缩略图项目 {i} 路径不匹配: {file_path} != {self.batch_files[i]}")

    def refresh_monitor(self):
        """刷新监控"""
        self.status_bar.showMessage("监控数据已刷新", 2000)

    def clear_monitor(self):
        """清空监控"""
        for i in reversed(range(self.monitor_layout.count())):
            child = self.monitor_layout.itemAt(i).widget()
            if child:
                child.setParent(None)

    def toggle_auto_refresh(self, enabled):
        """切换自动刷新"""
        if enabled:
            self.monitor_worker.start_monitoring()
        else:
            self.monitor_worker.stop_monitoring()

    def show_zoomed_image(self, item):
        """显示放大的图像"""
        file_path = item.data(Qt.UserRole)
        if file_path:
            dialog = ImageZoomDialog(file_path, self)
            dialog.exec_()

    def resizeEvent(self, event):
        """窗口大小改变事件"""
        super().resizeEvent(event)
        # overlay已废弃，改为非阻塞处理
        # self.overlay.setGeometry(self.rect())
        
        # 更新处理指示器位置
        if hasattr(self, 'processing_indicator') and self.processing_indicator.isVisible():
            parent_rect = self.rect()
            indicator_rect = self.processing_indicator.rect()
            x = parent_rect.width() - indicator_rect.width() - 20
            y = 50
            self.processing_indicator.move(x, y)

    def dragEnterEvent(self, event: QDragEnterEvent):
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            # 添加视觉反馈
            if hasattr(self, 'drop_area'):
                self.drop_area.setStyleSheet(self.drop_area.styleSheet() + """
                    QLabel#drop_area {
                        border-color: #00D4AA !important;
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                            stop:0 #004D40, stop:0.5 #00695C, stop:1 #004D40) !important;
                    }
                """)

    def dragLeaveEvent(self, event):
        """拖拽离开事件"""
        # 恢复原始样式
        self.drop_area.setStyleSheet("""
            QLabel#drop_area {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F8F9FA, stop:0.5 #FFFFFF, stop:1 #F8F9FA);
                border: 2px dashed #0078D4;
                border-radius: 15px;
                color: #605E5C;
                font-size: 16px;
                font-weight: bold;
                padding: 20px;
            }
            QLabel#drop_area:hover {
                border-color: #4F9EE8;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #F0F0F0, stop:0.5 #F8F9FA, stop:1 #F0F0F0);
                color: #323130;
            }
        """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("AI图像分类系统")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("AI Studio")
    
    # 应用现代化样式表
    app.setStyleSheet("""
        /* 全局样式 */
        QWidget {
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #FFFFFF, stop:1 #F8F9FA);
            color: #323130;
            font-family: 'Microsoft YaHei', 'Segoe UI', sans-serif;
            font-size: 14px;
        }
        
        /* 表格样式 - 强制黑色字体 */
        QTableWidget {
            color: #000000 !important;
        }
        
        QTableWidget::item {
            color: #000000 !important;
        }
        
        QHeaderView::section {
            color: #000000 !important;
        }
        
        /* 分组框样式 */
        QGroupBox {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8F9FA, stop:1 #FFFFFF);
            border: 2px solid #D2D0CE;
            border-radius: 10px;
            margin-top: 15px;
            padding-top: 20px;
            font-weight: bold;
        }
        
        QGroupBox::title {
            color: #0078D4;
            subcontrol-origin: margin;
            left: 15px;
            font-size: 15px;
            font-weight: bold;
        }
        
        /* 下拉框样式 */
        QComboBox {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #FFFFFF, stop:1 #F8F9FA);
            border: 2px solid #D2D0CE;
            border-radius: 8px;
            padding: 8px 12px;
            min-width: 120px;
            font-size: 13px;
            color: #323130;
        }
        
        QComboBox:hover {
            border-color: #0078D4;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F0F0F0, stop:1 #F8F9FA);
        }
        
        QComboBox:focus {
            border-color: #4F9EE8;
        }
        
        QComboBox::drop-down {
            width: 30px;
            border-left: 1px solid #D2D0CE;
            border-top-right-radius: 8px;
            border-bottom-right-radius: 8px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border: 2px solid #605E5C;
            width: 6px;
            height: 6px;
            border-top: none;
            border-right: none;
            margin-right: 8px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #FFFFFF;
            border: 1px solid #0078D4;
            selection-background-color: #0078D4;
            color: #323130;
            border-radius: 5px;
        }
        
        /* 滑块样式 */
        QSlider::groove:horizontal {
            height: 8px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F0F0F0, stop:1 #E5E5E5);
            border-radius: 4px;
            border: 1px solid #D2D0CE;
        }
        
        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0078D4, stop:1 #106EBE);
            width: 20px;
            height: 20px;
            margin: -6px 0;
            border-radius: 10px;
            border: 2px solid #FFFFFF;
        }
        
        QSlider::handle:horizontal:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #4F9EE8, stop:1 #0078D4);
        }
        
        QSlider::handle:horizontal:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #106EBE, stop:1 #005A9E);
        }
        
        /* 复选框样式 */
        QCheckBox {
            spacing: 8px;
            color: #323130;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border: 2px solid #D2D0CE;
            border-radius: 6px;
            background: #FFFFFF;
        }
        
        QCheckBox::indicator:hover {
            border-color: #0078D4;
            background: #F8F9FA;
        }
        
        QCheckBox::indicator:checked {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0078D4, stop:1 #4F9EE8);
            border-color: #0078D4;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0IDdMMTEgMSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
        }
        
        /* 按钮样式 */
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8F9FA, stop:1 #F0F0F0);
            border: 2px solid #D2D0CE;
            border-radius: 8px;
            padding: 10px 20px;
            min-width: 100px;
            font-weight: bold;
            color: #323130;
        }
        
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F0F0F0, stop:1 #E5E5E5);
            border-color: #0078D4;
            color: #323130;
        }
        
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #E5E5E5, stop:1 #D2D0CE);
            border-color: #106EBE;
        }
        
        QPushButton:disabled {
            background: #F8F9FA;
            border-color: #EDEBE9;
            color: #A19F9D;
        }
        
        /* 数字输入框样式 */
        QSpinBox {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #FFFFFF, stop:1 #F8F9FA);
            border: 2px solid #D2D0CE;
            border-radius: 6px;
            padding: 6px;
            font-size: 13px;
            color: #323130;
        }
        
        QSpinBox:hover {
            border-color: #0078D4;
        }
        
        QSpinBox:focus {
            border-color: #4F9EE8;
        }
        
        /* 标签页样式 */
        QTabWidget::pane {
            border: 2px solid #D2D0CE;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8F9FA, stop:1 #FFFFFF);
            border-radius: 8px;
        }
        
        QTabBar::tab {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8F9FA, stop:1 #F0F0F0);
            color: #605E5C;
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
            min-width: 100px;
        }
        
        QTabBar::tab:selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #0078D4, stop:1 #106EBE);
            color: #FFFFFF;
            border-bottom: 3px solid #4F9EE8;
        }
        
        QTabBar::tab:hover:!selected {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F0F0F0, stop:1 #E5E5E5);
            color: #323130;
        }
        
        /* 滚动条样式 */
        QScrollBar:vertical {
            background: #F8F9FA;
            width: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical {
            background: #D2D0CE;
            border-radius: 6px;
            min-height: 20px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #C8C6C4;
        }
        
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background: #F8F9FA;
            height: 12px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal {
            background: #D2D0CE;
            border-radius: 6px;
            min-width: 20px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background: #C8C6C4;
        }
        
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* 状态栏样式 */
        QStatusBar {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8F9FA, stop:1 #F0F0F0);
            border-top: 2px solid #0078D4;
            color: #323130;
            padding: 5px;
        }
        
        /* 分割器样式 */
        QSplitter::handle {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #D2D0CE, stop:0.5 #0078D4, stop:1 #D2D0CE);
        }
        
        QSplitter::handle:horizontal {
            width: 4px;
        }
        
        QSplitter::handle:vertical {
            height: 4px;
        }
        
        QSplitter::handle:hover {
            background: #0078D4;
        }
        
        /* 工具提示样式 */
        QToolTip {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #F8F9FA, stop:1 #F0F0F0);
            color: #323130;
            border: 1px solid #0078D4;
            border-radius: 6px;
            padding: 8px;
            font-size: 12px;
        }
        
        /* 消息框样式 */
        QMessageBox {
            background: #FFFFFF;
            color: #323130;
        }
        
        QMessageBox QPushButton {
            min-width: 80px;
            padding: 8px 16px;
        }
    """)
    
    try:
        window = MainWindow()
        window.show()
        
        logger.info("应用程序启动成功")
        
        # 显示启动信息
        window.status_bar.showMessage("系统已启动，准备就绪", 5000)
        
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {e}")
        QMessageBox.critical(None, "启动错误", f"应用程序启动失败:\n{e}")
        sys.exit(1)