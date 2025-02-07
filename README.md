# 图像预测结果读取

该项目使用 YOLO 模型进行图像预测，能够识别并返回图像中物体的名称及其对应的概率。

## 安装依赖

在使用之前，请确保安装了以下依赖项：
bash
pip install gradio pandas ultralytics scikit-image pillow

## 使用方法

1. 将模型文件 `best.pt` 放置在 `model/` 目录下。
2. 运行以下命令启动 Gradio 接口：
bash
python your_script.py

3. 打开浏览器，访问提供的链接，上传图像进行预测。

## 函数说明

### `predict(img)`

该函数接受一张图像作为输入，并返回预测结果。

- **参数**:
  - `img`: 输入的图像，类型为 PIL 图像。

- **返回**:
  - 返回一个字典，包含物体名称及其对应的概率。

### 示例图片

以下是一些示例图片，可以用于测试预测功能：

- ![示例图片1](examples/250116073019.094-6-4.jpg)
- ![示例图片2](examples/sample_image_2.jpg)
- ![示例图片3](examples/sample_image_3.jpg)

## 许可证

该项目遵循
