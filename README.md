好的，这是一个非常全面的 `README.md` 文件，它结合了您项目的**训练**部分和**预测**部分，并包含了您要求的所有内容。

---

# YOLOv8 图像分类训练与预测平台

本项目提供了一个基于 Gradio 的交互式 Web 界面，用于端到端的图像分类任务。它集成了 `ultralytics` YOLOv8 框架，支持从模型训练、实时监控到最终图像预测的全过程。

![项目动态图](gridio_yolov8.gif)

## 主要功能

- **交互式模型训练**: 通过 Web 界面配置所有训练参数（如数据集路径、模型架构、学习率、epochs 等），一键启动训练。
- **实时训练监控**:
    - 实时更新训练状态和当前 epoch。
    - 动态展示训练过程中的批次样本图像 (`train_batch.jpg`)。
- **可视化结果分析**:
    - 训练成功后，自动展示由 YOLOv8 生成的详细结果图表，包括：
        - 损失/指标曲线 (`results.png`)
        - 混淆矩阵 (`confusion_matrix.png`)
        - 验证集批次样本 (`val_batch0.jpg`)
    - 提供一份清晰的训练结果总结，包含最佳模型路径和最终验证集指标。
- **模型推理/预测**: 提供一个独立的 Gradio 标签页，用于上传图像并使用训练好的模型进行实时预测。
- **MLflow 集成**: 自动记录训练参数、指标和产物（包括 ONNX 模型）到 MLflow，便于实验跟踪和管理。

## 项目结构

```
.
├── app.py                  # Gradio 应用主程序
├── train_core.py           # 核心训练逻辑脚本
├── predict_core.py         # 核心预测逻辑脚本 (可选，也可集成在 app.py)
├── ../dataset/             # 存放原始数据集
│   ├── class_A/
│   └── class_B/
├── ../pts/                 # 存放 YOLOv8 预训练模型 (.pt 和 .yaml)
├── runs/                   # YOLOv8 训练的默认输出目录
├── model/                  # 存放用于预测的最佳模型 (best.pt)
├── examples/               # 存放用于测试预测功能的示例图片
└── README.md               # 本文档
```

## 模块一：模型训练

### 安装依赖

```bash
pip install gradio pandas ultralytics scikit-learn pyyaml mlflow matplotlib
```

### 使用方法

1.  **准备数据集**: 按照 `../dataset/class_A/image.jpg` 的结构组织您的分类数据集。
2.  **准备预训练模型**: 将 YOLOv8 分类模型的 `.pt` 和 `.yaml` 文件（例如 `yolov8s-cls.pt` 和 `yolov8s-cls.yaml`）放置在 `../pts/` 目录下。
3.  **启动训练界面**:
    ```bash
    python app.py
    ```
4.  **操作界面**:
    - 在浏览器中打开提供的链接 (通常是 `http://127.0.0.1:7860`)。
    - 在左侧面板配置您的数据集路径、输出目录和所有训练超参数。
    - 点击 "🚀 开始训练" 按钮。
    - 在右侧面板实时观察训练进度和最终结果。

## 模块二：图像预测

该项目同样支持使用训练好的模型进行图像预测，能够识别并返回图像中物体的名称及其对应的概率。

### 安装依赖

```bash
pip install gradio ultralytics pillow scikit-image
```

### 使用方法

1.  **放置模型**: 将您训练得到的最佳模型文件（通常名为 `best.pt`）放置在 `model/` 目录下。
2.  **启动预测界面**:
    ```bash
    python your_predict_script.py  # 假设预测功能在另一个脚本中
    ```
    *注意: 您也可以将训练和预测功能集成在同一个 `app.py` 中，使用 Gradio 的 `gr.TabbedInterface`。*
3.  **进行预测**: 打开浏览器，访问提供的链接，上传图像即可看到预测结果。

### 函数说明

#### `predict(img)`

该函数接受一张图像作为输入，并返回预测结果。

-   **参数**:
    -   `img`: 输入的图像，类型为 PIL 图像或 NumPy 数组。
-   **返回**:
    -   返回一个字典，键是类别名称，值是对应的置信度（概率）。

## 许可证

该项目遵循 [MIT 许可证](LICENSE)。
