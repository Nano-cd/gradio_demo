# app.py
import gradio as gr
import pandas as pd
import threading
import time
import argparse
from pathlib import Path
import logging
import yaml

# 从我们的核心训练脚本中导入 main 函数
from train_core import main as train_main

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量来跟踪训练状态和结果路径
training_state = {
    "is_running": False,
    "results_dir": None,
    "thread": None,
    "args": None,
    "error_message": None
}


def find_latest_train_batch(results_dir):
    """辅助函数：查找最新的训练批次图像"""
    if not results_dir or not results_dir.exists():
        return None
    image_files = sorted(results_dir.glob("train_batch*.jpg"))
    return str(image_files[-1]) if image_files else None


def run_training_and_update_ui(
        data_dir, output_dir, experiment_name, test_size,
        model_arch, epochs, batch_size, img_size, optimizer,
        learning_rate, dropout, seed
):
    """
    一个生成器函数，启动训练并持续 yield 更新给 UI。
    直接显示YOLO生成的图像文件，而不是动态绘制。
    """
    if training_state["is_running"]:
        # 如果已经在运行，只返回一个状态文本
        yield "一个训练任务已在运行中。", None, None, None, None, None, gr.Accordion(visible=False)
        return

    # 1. 重置状态并构造 args
    training_state.update({"is_running": True, "results_dir": None, "error_message": None})
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
            save_dir = train_main(args)
            training_state["results_dir"] = Path(save_dir)
        except Exception as e:
            error_msg = f"训练出错: {e}"
            logging.error(error_msg, exc_info=True)
            training_state["error_message"] = error_msg
        finally:
            training_state["is_running"] = False

    # 3. 启动训练线程并返回初始状态
    training_state["thread"] = threading.Thread(target=training_thread_target)
    training_state["thread"].start()
    yield "训练已开始，正在初始化...", None, None, None, None, gr.Markdown(visible=False), gr.Accordion(visible=False)
    time.sleep(5)

    # 4. 循环更新 UI (只更新实时部分)
    while training_state["thread"].is_alive():
        # 主动查找结果目录
        if training_state["results_dir"] is None:
            base_path = Path(args.output_dir) / args.experiment_name
            if base_path.exists():
                exp_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
                if exp_dirs: training_state["results_dir"] = exp_dirs[-1]

        status_text = f"训练正在进行中... (Epochs: {epochs})"
        live_train_batch = find_latest_train_batch(training_state["results_dir"])

        # 在循环中，只更新状态和实时批次图，其他部分保持不变(None)
        yield status_text, live_train_batch, None, None, None, gr.Markdown(visible=False), gr.Accordion(visible=True,
                                                                                                        open=False)
        time.sleep(5)

    # 5. 训练结束后，发送最终状态
    # 检查是否有错误发生
    if training_state["error_message"]:
        final_status = f"🔴 **训练失败**\n\n错误信息: {training_state['error_message']}"
        yield final_status, None, None, None, None, gr.Markdown(visible=False), gr.Accordion(visible=False)
        return

    # 如果成功，准备并显示最终结果
    final_status = "✅ **训练成功完成！**"
    results_dir = training_state["results_dir"]

    # 定义要查找的图像文件路径
    results_png = results_dir / "results.png" if results_dir else None
    confusion_matrix_png = results_dir / "confusion_matrix.png" if results_dir else None
    val_batch_jpg = results_dir / "val_batch0.jpg" if results_dir else None

    # 检查文件是否存在，如果不存在则设为None
    final_results_png = str(results_png) if results_png and results_png.exists() else None
    final_confusion_matrix_png = str(
        confusion_matrix_png) if confusion_matrix_png and confusion_matrix_png.exists() else None
    final_val_batch_jpg = str(val_batch_jpg) if val_batch_jpg and val_batch_jpg.exists() else None

    # 保留最后一张训练批次图
    final_train_batch = find_latest_train_batch(results_dir)

    # 构建结果摘要 Markdown
    summary_text = "### 训练结果总结\n\n"
    if results_dir and results_dir.exists():
        summary_text += f"- **结果保存路径**: `{results_dir}`\n"
        best_model_path = results_dir / "weights" / "best.pt"
        if best_model_path.exists(): summary_text += f"- **最佳模型**: `{best_model_path}`\n"

        results_csv_path = results_dir / "results.csv"
        if results_csv_path.exists() and not pd.read_csv(results_csv_path).empty:
            df = pd.read_csv(results_csv_path)
            df.columns = df.columns.str.strip()
            final_metrics = df.iloc[-1]
            acc_top1 = final_metrics.get('metrics/accuracy_top1', 'N/A')
            if acc_top1 != 'N/A': acc_top1 = f"{acc_top1:.4f}"
            val_loss = final_metrics.get('val/cls_loss', 'N/A')
            if val_loss != 'N/A': val_loss = f"{val_loss:.4f}"
            summary_text += f"- **最终验证集Top-1准确率**: {acc_top1}\n"
            summary_text += f"- **最终验证集损失**: {val_loss}\n"

    final_summary_md = gr.Markdown(value=summary_text, visible=True)

    # 最终 yield，更新所有组件并展开折叠面板
    yield final_status, final_train_batch, final_results_png, final_confusion_matrix_png, final_val_batch_jpg, final_summary_md, gr.Accordion(
        visible=True, open=True)


# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue")) as demo:
    gr.Markdown("# 🚀 YOLOv8 分类模型训练器")
    with gr.Row():
        with gr.Column(scale=1):
            # ... (输入组件部分不变) ...
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
            gr.Markdown("### 2. 实时训练监控与结果")
            status_output = gr.Textbox(label="训练状态", interactive=False, value="等待训练开始...")

            # 用于实时显示训练批次的组件
            train_batch_output = gr.Image(label="实时训练批次样本 (train_batch)", type="filepath")

            # 用于显示最终结果的折叠面板
            with gr.Accordion("最终训练结果图像", visible=False, open=False) as results_accordion:
                with gr.Row():
                    results_png_output = gr.Image(label="损失/指标曲线 (results.png)", type="filepath")
                    confusion_matrix_output = gr.Image(label="混淆矩阵 (confusion_matrix.png)", type="filepath")
                val_batch_output = gr.Image(label="验证批次样本 (val_batch0.jpg)", type="filepath")

            summary_output = gr.Markdown(visible=False)

    # 绑定事件
    all_inputs = [
        data_dir_input, output_dir_input, experiment_name_input, test_size_input,
        model_arch_input, epochs_input, batch_size_input, img_size_input,
        optimizer_input, learning_rate_input, dropout_input, seed_input
    ]
    all_outputs = [
        status_output,
        train_batch_output,
        results_png_output,
        confusion_matrix_output,
        val_batch_output,
        summary_output,
        results_accordion
    ]

    start_button.click(
        fn=run_training_and_update_ui,
        inputs=all_inputs,
        outputs=all_outputs
    )

if __name__ == "__main__":
    demo.launch()
