import gc

import psutil
import GPUtil
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from typing import Dict

class SystemMonitor:
    def __init__(self, max_points: int = 60):

        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory = []
        self.max_points = max_points
        self.inference_time = "N/A"
        self.model_info = "N/A"
        self.system_status = "就绪"

    def update_hardware_info(self):

        self.cpu_usage.append(psutil.cpu_percent(interval=1))
        self.memory_usage.append(psutil.virtual_memory().percent)
        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_usage.append(gpus[0].load * 100)
            self.gpu_memory.append(gpus[0].memoryUtil * 100)
        else:
            self.gpu_usage.append(0)
            self.gpu_memory.append(0)

        if len(self.cpu_usage) > self.max_points:
            self.cpu_usage = self.cpu_usage[-self.max_points:]
            self.memory_usage = self.memory_usage[-self.max_points:]
            self.gpu_usage = self.gpu_usage[-self.max_points:]
            self.gpu_memory = self.gpu_memory[-self.max_points:]

    def update_inference_info(self, inference_time: str, model_info: str):

        self.inference_time = inference_time
        self.model_info = model_info

    def update_system_status(self, status: str):

        self.system_status = status

    def generate_charts(self) -> BytesIO:
        """
        生成2x2布局的监控图表
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))
        x = range(len(self.cpu_usage))

        # CPU使用率
        ax1.plot(x, self.cpu_usage, 'b-', linewidth=1)
        ax1.set_title('CPU Usage (%)', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.tick_params(axis='both', labelsize=8)

        # 内存使用率
        ax2.plot(x, self.memory_usage, 'g-', linewidth=1)
        ax2.set_title('Memory Usage (%)', fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', labelsize=8)

        # GPU使用率
        ax3.plot(x, self.gpu_usage, 'r-', linewidth=1)
        ax3.set_title('GPU Usage (%)', fontsize=10)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.tick_params(axis='both', labelsize=8)

        # GPU内存使用率
        ax4.plot(x, self.gpu_memory, 'm-', linewidth=1)
        ax4.set_title('GPU Memory Usage (%)', fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)
        ax4.tick_params(axis='both', labelsize=8)

        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)  # 增加子图间距
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        gc.collect()
        return buf

    def get_text_info(self) -> str:

        return f"""
        推理耗时: {self.inference_time}
        模型信息: {self.model_info}
        系统状态: {self.system_status}
        """

    def get_monitor_data(self) -> Dict[str, str]:

        return {
            "CPU 使用率": f"{self.cpu_usage[-1] if self.cpu_usage else 0:.2f}%",
            "内存使用率": f"{self.memory_usage[-1] if self.memory_usage else 0:.2f}%",
            "GPU 使用率": f"{self.gpu_usage[-1] if self.gpu_usage else 0:.2f}%" if self.gpu_usage else "N/A",
            "GPU 内存使用率": f"{self.gpu_memory[-1] if self.gpu_memory else 0:.2f}%" if self.gpu_memory else "N/A",
            "推理耗时": self.inference_time,
            "模型信息": self.model_info,
            "系统状态": self.system_status,
            "charts": self.generate_charts().getvalue()
        }