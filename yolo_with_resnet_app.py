from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
import gradio as gr
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import time
import os
import requests
from tqdm import tqdm
from monitor.system_monitor import SystemMonitor


# 加载配置文件
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 路径标准化处理
        for model in config["models"].values():
            model["path"] = str(Path(model["path"])).replace("\\", "/")

        return config
    except Exception as e:
        raise RuntimeError(f"配置加载失败: {str(e)}")


CONFIG = load_config()


class ModelManager:
    def __init__(self):
        self.models = CONFIG["models"]
        self.current_model = None
        self.current_model_name = None
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # 移除分类层，保留特征提取部分
        for param in self.resnet.parameters():
            param.requires_grad = False

    def download_model(self, model_name: str, progress: gr.Progress()):
        """带进度条的模型下载"""
        model_cfg = self.models[model_name]
        os.makedirs(os.path.dirname(model_cfg["path"]), exist_ok=True)

        if Path(model_cfg["path"]).exists():
            return

        try:
            response = requests.get(model_cfg["url"], stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            progress(0, desc=f"下载 {model_name} ({model_cfg['size']}MB)")
            with open(model_cfg["path"], 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True,
                    desc=model_name, ascii=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        progress(pbar.n / total_size)
        except Exception as e:
            raise RuntimeError(f"下载失败: {str(e)}")

    def load_model(self, model_name: str, progress: gr.Progress()):
        """动态加载模型并返回模型对象"""
        if model_name == self.current_model_name:
            return self.current_model

        model_cfg = self.models[model_name]

        try:
            if not Path(model_cfg["path"]).exists():
                progress(0, desc="初始化下载...")
                self.download_model(model_name, progress)

            progress(0.5, desc="加载模型中...")
            self.current_model = YOLO(model_cfg["path"])
            self.current_model_name = model_name
            return self.current_model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")


model_manager = ModelManager()



def preprocess_image_resnet(img: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, torch.Tensor]:
    """图像预处理并为 ResNet 准备输入"""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # 为 ResNet 准备输入（转换为张量并归一化）
    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float() / 255.0  # [C, H, W]
    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
    img_tensor = nn.functional.interpolate(img_tensor, size=(224, 224), mode='bilinear',
                                           align_corners=False)  # ResNet 输入要求
    return resized_img, img_tensor



def classify_image_resnet(
        img: np.ndarray,
        model_name: str = "nano",
        confidence_threshold: float = 0.25,
        enable_visualization: bool = True,
        progress: gr.Progress = gr.Progress(),
        system_monitor: SystemMonitor = None
) -> Tuple[Dict, np.ndarray]:
    """图像分类主函数，集成 ResNet 特征增强"""
    start_time = time.time()
    try:
        # 输入验证
        if img is None or img.size == 0:
            raise ValueError("无效的输入图像")

        # 加载模型
        progress(0.1, desc="初始化模型...")
        model = model_manager.load_model(model_name, progress)

        # 预处理
        progress(0.3, desc="预处理图像...")
        processed_img, img_tensor = preprocess_image_resnet(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 使用 ResNet 提取特征
        progress(0.4, desc="提取 ResNet 特征...")
        with torch.no_grad():
            resnet_features = model_manager.resnet(img_tensor)  # [1, 2048, 7, 7]

        # 将特征融合到 YOLOv8 输入（这里简化为直接推理，实际可修改 YOLOv8 内部结构）
        progress(0.6, desc="推理中...")
        # results = model(processed_img)  # 暂时直接使用 YOLOv8 推理

        # 将 ResNet 特征融合到 YOLOv8 输入
        # 在 classify_image 中添加
        fusion_layer = nn.Conv2d(3 + 2048, 3, kernel_size=1)  # 调整通道数
        resnet_features_resized = nn.functional.interpolate(resnet_features, size=processed_img.shape[:2],
                                                            mode='bilinear')
        fused_input = torch.cat(
            [torch.from_numpy(processed_img).permute(2, 0, 1).unsqueeze(0), resnet_features_resized], dim=1)
        fused_input = fusion_layer(fused_input).permute(0, 2, 3, 1).numpy()[0]
        results = model(fused_input)

        # 解析结果
        progress(0.8, desc="解析结果...")
        probs = results[0].probs.data.tolist()
        all_results = {model.names[i]: float(probs[i]) for i in range(len(probs))}
        filtered_results = {k: v for k, v in all_results.items() if v >= confidence_threshold}
        if not filtered_results:
            max_key = max(all_results, key=all_results.get)
            filtered_results = {max_key: all_results[max_key]}

        sorted_results = dict(sorted(filtered_results.items(), key=lambda x: x[1], reverse=True))
        sorted_results = dict(list(sorted_results.items())[:5])

        # 可视化
        visualization = None
        if enable_visualization:
            try:
                vis = results[0].plot()
                visualization = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            except:
                visualization = np.zeros((640, 640, 3), dtype=np.uint8)
                visualization = cv2.putText(visualization, "可视化失败", (50, 320),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if system_monitor:
            system_monitor.update_inference_info(
                inference_time=f"{(time.time() - start_time):.2f} s",
                model_info=f"name:{model_name} size:{CONFIG['models'][model_name]['size']}MB"
            )
            system_monitor.update_system_status("运行单图模式")
        return (
            sorted_results,
            visualization,
        )
    except Exception as e:
        if system_monitor:
            system_monitor.update_system_status(f"单图模式错误: {str(e)}")
        return (
            {"错误": str(e)},
            None,
        )

def batch_process(files: List[str], model_name: str, progress: gr.Progress = gr.Progress(),
                  system_monitor: SystemMonitor = None):
    """批量处理函数（返回4个输出）"""
    gallery_images = []  # 存储可视化结果
    label_images = []  # 存储带标签 的缩略图
    all_results = []  # 存储详细结果
    start_time = time.time()
    for i, f in enumerate(files):

        try:
            progress(i / len(files), desc=f"处理文件 {i + 1}/{len(files)}")

            # 处理图像
            img = cv2.imread(f.name)
            data, vis = classify_image_resnet(img, model_name=model_name, system_monitor=SystemMonitor())

            # 生成带标签的缩略图
            h, w = vis.shape[:2] if vis is not None else (100, 100)
            label_img = np.zeros((h + 60, w, 3), dtype=np.uint8)
            if vis is not None:
                label_img[:h] = vis
            cv2.putText(label_img,
                        f"Top1: {list(data.keys())[0]} {list(data.values())[0]:.2f}",
                        (10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 收集结果
            gallery_images.append(vis if vis is not None else np.zeros((100, 100, 3)))
            label_images.append(label_img)
            all_results.append(data)
            if system_monitor:
                system_monitor.update_system_status(f"运行批量模式 ({i + 1}/{len(files)})")
                system_monitor.model_info = f"name:{model_name} size:{CONFIG['models'][model_name]['size']}MB"
        except Exception as e:
            error_img = np.zeros((160, 160, 3), dtype=np.uint8)
            cv2.putText(error_img, "处理失败", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            gallery_images.append(error_img)
            label_images.append(error_img)
            all_results.append({"错误": str(e)})
            if system_monitor:
                system_monitor.update_system_status(f"批量模式错误: {str(e)}")
    if system_monitor:
        system_monitor.inference_time = f"{(time.time() - start_time):.2f} s"
    return (
        label_images,  # 显示带标签的缩略图
        all_results[0],  # 默认显示第一个结果
    )


with gr.Blocks(title=CONFIG.get("title", "图像分类系统")) as app:
    gr.Markdown(f"## {CONFIG.get('title', '基于YOLOv8的增强图像分类系统')}")
    system_monitor = SystemMonitor()
    with gr.Row():
        with gr.Column(scale=1):
            model_selector = gr.Dropdown(
                label="模型版本",
                choices=list(CONFIG["models"].keys()),
                value=next(iter(CONFIG["models"]), "nano"),
                interactive=True
            )
            confidence_slider = gr.Slider(0, 1, 0.5,
                                          label="置信度阈值", step=0.01)
            visualize_toggle = gr.Checkbox(True, label="显示可视化")
            batch_upload = gr.File(
                file_count="multiple",
                file_types=CONFIG["app_settings"]["allowed_extensions"],
                label="批量模式上传窗口"
            )
            status = gr.Textbox("就绪", label="系统状态", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("单图模式"):
                    image_input = gr.Image(type="numpy", label="单图模式输入窗口")
                    image_output = gr.Image(label="可视化结果")
                    single_result = gr.Label(num_top_classes=5, label="分类结果")
                    single_btn = gr.Button("开始分类", variant="primary")

                with gr.TabItem("批量模式"):
                    with gr.Row():
                        batch_gallery = gr.Gallery(
                            label="结果预览（点击查看详情）",
                            columns=4,
                            object_fit="cover",
                            height=400,
                            interactive=False,
                            preview=True,
                        )
                        with gr.Column():
                            batch_detail_label = gr.Label(label="分类详情")

                    batch_btn = gr.Button("开始批量处理", variant="primary")

    with gr.Tab("系统监控"):
        monitor_chart = gr.Image(label="硬件监控图表", type="pil", every=1)
        monitor_text = gr.Textbox(label="系统信息", interactive=False)


        def update_monitor():
            from PIL import Image
            system_monitor.update_hardware_info()
            chart_buf = system_monitor.generate_charts()
            chart_image = Image.open(chart_buf)
            text_info = system_monitor.get_text_info()
            return chart_image, text_info


        # Update every second
        gr.Timer(1).tick(update_monitor, outputs=[monitor_chart, monitor_text])

    # 事件绑定
    single_btn.click(
        lambda img, model, conf, vis: classify_image_resnet(img, model, conf, vis, system_monitor=system_monitor),
        [image_input, model_selector, confidence_slider, visualize_toggle],
        [single_result, image_output],
        api_name="classify"
    )

    batch_btn.click(
        lambda model, files: batch_process(files, model_name=model, system_monitor=system_monitor),
        [model_selector, batch_upload],
        [batch_gallery, batch_detail_label],
        api_name="batch"
    )

    model_selector.change(
        lambda name: (model_manager.load_model(name, progress=gr.Progress()), f"已选择模型: {name}")[1],
        [model_selector],
        [status],
        show_progress="minimal"
    )

if __name__ == "__main__":
    app_settings = CONFIG.get("app_settings", {})
    app.launch(
        server_name=app_settings.get("server_name", "0.0.0.0"),
        server_port=app_settings.get("server_port", 7870),
        share=app_settings.get("share", False)
    )
