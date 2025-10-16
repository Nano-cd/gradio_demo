# app.py
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
import argparse
from pathlib import Path
import logging

# 从我们的核心训练脚本中导入 main 函数
from train_core import main as train_main

# 全局变量来跟踪训练状态和结果路径
training_state = {
    "is_running": False,
    "results_dir": None,
    "thread": None,
    "args": None  # 保存参数以便查找结果目录
}


def run_training_and_update_ui(
        data_dir, output_dir, experiment_name, test_size,
        model_arch, epochs, batch_size, img_size, optimizer,
        learning_rate, dropout, seed
):
    """
    一个生成器函数，启动训练并持续 yield 更新给 UI。
    这是兼容旧版 Gradio 的方法。
    """
    if training_state["is_running"]:
        yield "一个训练任务已在运行中。", None, None
        return

    # 1. 构造 args 对象并保存
    args = argparse.Namespace(
        data_dir=data_dir, output_dir=output_dir, experiment_name=experiment_name,
        test_size=test_size, model_arch=model_arch, epochs=epochs, batch_size=batch_size,
        img_size=img_size, optimizer=optimizer, learning_rate=learning_rate,
        dropout=dropout, seed=seed
    )
    training_state["args"] = args

    # 2. 定义线程目标函数
    def training_thread_target():
        try:
            training_state["is_running"] = True
            save_dir = train_main(args)
            training_state["results_dir"] = Path(save_dir)
        except Exception as e:
            logging.error(f"训练出错: {e}")
        finally:
            training_state["is_running"] = False

    # 3. 启动训练线程
    training_state["thread"] = threading.Thread(target=training_thread_target)
    training_state["thread"].start()
    yield "训练已开始，正在初始化...", None, None
    time.sleep(5)  # 等待 YOLO 初始化并创建目录

    # 4. 循环更新 UI
    while training_state["thread"].is_alive():
        # 查找最新的运行目录
        results_dir = training_state["results_dir"]
        if results_dir is None:
            # 尝试主动查找
            base_path = Path(args.output_dir) / args.experiment_name
            if base_path.exists():
                exp_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
                if exp_dirs:
                    results_dir = exp_dirs[-1]
                    training_state["results_dir"] = results_dir

        status_text = "训练正在进行中..."
        loss_plot = None
        batch_image = None

        if results_dir and results_dir.exists():
            results_csv_path = results_dir / "results.csv"
            if results_csv_path.exists():
                try:
                    df = pd.read_csv(results_csv_path)
                    df.columns = df.columns.str.strip()

                    current_epoch = df['epoch'].max()
                    status_text += f"\nEpoch: {current_epoch}/{epochs}"

                    fig, ax = plt.subplots()
                    if 'train/cls_loss' in df.columns:
                        ax.plot(df['epoch'], df['train/cls_loss'], label='Train Loss')
                    if 'val/cls_loss' in df.columns:
                        ax.plot(df['epoch'], df['val/cls_loss'], label='Validation Loss')
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Classification Loss")
                    ax.legend()
                    ax.grid(True)
                    plt.tight_layout()
                    loss_plot = fig
                except Exception:
                    plt.close('all')  # 清理资源

            image_files = sorted(results_dir.glob("train_batch*.jpg"))
            if image_files:
                batch_image = str(image_files[-1])

        yield status_text, loss_plot, batch_image
        time.sleep(5)  # 轮询间隔

    # 5. 训练结束后，发送最终状态
    status_text = "训练完成！"
    # 这里可以再次获取最终的图表和图像
    final_plot, final_image = None, None
    if training_state["results_dir"]:
        # (可以复制上面的绘图和图像查找逻辑来获取最终结果)
        # 为简化起见，我们只更新文本
        pass

    yield status_text, None, None  # 你可以完善这里返回最终的图表


# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# 🚀 YOLOv8 分类模型训练器 (旧版兼容)")
    # ... UI 组件定义和之前一样 ...
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 配置参数")

            with gr.Accordion("数据和路径设置", open=True):
                data_dir_input = gr.Textbox(value="../dataset", label="数据目录")
                output_dir_input = gr.Textbox(value="runs", label="输出目录")
                experiment_name_input = gr.Textbox(value="Tube_Classification_Gradio", label="实验名称")
                test_size_input = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="验证集比例")

            with gr.Accordion("模型和训练设置", open=True):
                model_arch_input = gr.Dropdown(["yolov8n-cls", "yolov8s-cls", "yolov8m-cls"], value="yolov8s-cls",
                                               label="模型架构")
                epochs_input = gr.Slider(1, 100, value=10, step=1, label="训练周期 (Epochs)")
                batch_size_input = gr.Slider(2, 64, value=16, step=2, label="批次大小 (Batch Size)")
                img_size_input = gr.Slider(128, 640, value=224, step=32, label="图像尺寸 (Image Size)")
                optimizer_input = gr.Dropdown(['SGD', 'Adam', 'AdamW'], value="Adam", label="优化器")
                learning_rate_input = gr.Number(value=0.001, label="学习率", precision=5)
                dropout_input = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Dropout率")
                seed_input = gr.Number(value=42, label="随机种子", precision=0)

            start_button = gr.Button("🚀 开始训练", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 2. 实时训练监控")
            status_output = gr.Textbox(label="训练状态", interactive=False, value="等待训练开始...")
            with gr.Row():
                loss_plot_output = gr.Plot(label="损失曲线")
                batch_image_output = gr.Image(label="最新训练批次样本", type="filepath")

    # 绑定事件
    all_inputs = [
        data_dir_input, output_dir_input, experiment_name_input, test_size_input,
        model_arch_input, epochs_input, batch_size_input, img_size_input,
        optimizer_input, learning_rate_input, dropout_input, seed_input
    ]
    all_outputs = [status_output, loss_plot_output, batch_image_output]

    # 将按钮点击事件绑定到一个生成器函数
    start_button.click(
        fn=run_training_and_update_ui,
        inputs=all_inputs,
        outputs=all_outputs
    )

if __name__ == "__main__":
    demo.launch()
