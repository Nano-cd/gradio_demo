# train_core.py
import argparse
import json
import logging
import shutil
import yaml
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ultralytics import YOLO, settings

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(data_dir: Path, test_size: float, random_state: int):
    """
    加载数据、进行分层抽样并返回分割后的数据和元信息。
    """
    logging.info(f"Loading data from: {data_dir}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    file_paths = []
    labels = []

    for label_idx, class_name in enumerate(classes):
        class_dir = data_dir / class_name
        for img_file in class_dir.glob('*.*'):  # 支持多种图像格式
            file_paths.append(str(img_file))
            labels.append(label_idx)

    if not file_paths:
        raise ValueError(f"No images found in {data_dir}")

    X_train, X_test, y_train, y_test = train_test_split(
        file_paths,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    class_dist = {
        cls: {
            "total": labels.count(i),
            "train": y_train.count(i),
            "val": y_test.count(i)
        } for i, cls in enumerate(classes)
    }
    logging.info(f"Class distribution: {json.dumps(class_dist, indent=2)}")

    df = pd.DataFrame({
        "path": X_train + X_test,
        "split": ["train"] * len(X_train) + ["val"] * len(X_test),
        "label": [classes[l] for l in y_train + y_test]  # 使用类名而不是索引
    })

    return X_train, X_test, y_train, y_test, classes, class_dist, df


def prepare_yolo_dataset(X_train, X_test, y_train, y_test, classes, dataset_dir: Path):
    """
    创建符合YOLO分类标准的数据集结构和YAML文件。
    """
    logging.info(f"Preparing YOLO dataset at: {dataset_dir}")
    if dataset_dir.exists():
        logging.warning(f"Dataset directory {dataset_dir} already exists. Removing it.")
        shutil.rmtree(dataset_dir)

    splits = {"train": (X_train, y_train), "val": (X_test, y_test)}

    for split_name, (file_list, labels) in splits.items():
        for img_path_str, label_idx in zip(file_list, labels):
            src = Path(img_path_str)
            class_name = classes[label_idx]
            dst_dir = dataset_dir / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

    yolo_config = {
        "path": str(dataset_dir.resolve()),
        "train": "train",
        "val": "val",
        "nc": len(classes),
        "names": classes
    }

    config_path = dataset_dir / "dataset.yaml"
    with open(config_path, "w") as f:
        yaml.dump(yolo_config, f, sort_keys=False, default_flow_style=False)

    logging.info(f"YOLO dataset.yaml created at: {config_path}")
    return config_path


def main(args):
    """
    主函数，执行数据准备、模型训练和结果记录。
    """
    mlflow_enabled = False
    try:
        import mlflow
        settings.update({"mlflow": True})
        mlflow_enabled = True
        logging.info("MLflow integration enabled for YOLOv8.")
    except ImportError:
        mlflow = None
        settings.update({"mlflow": False})
        logging.warning("MLflow not found. Running without MLflow tracking.")

    # 提前设置实验，YOLOv8会自动在该实验下创建run
    if mlflow_enabled:
        mlflow.set_experiment(args.experiment_name)

    run_id = None
    try:
        # ==================== 数据准备阶段 ====================
        data_dir = Path(args.data_dir)
        X_train, X_test, y_train, y_test, classes, class_dist, df = load_data(
            data_dir, args.test_size, args.seed
        )

        # 准备需要手动记录的产物
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        class_dist_file = output_path / "class_distribution.json"
        data_split_file = output_path / "data_split.csv"

        with open(class_dist_file, "w") as f:
            json.dump(class_dist, f, indent=4)
        df.to_csv(data_split_file, index=False)

        yolo_dataset_dir = output_path / "yolo_dataset"
        yaml_path = prepare_yolo_dataset(
            X_train, X_test, y_train, y_test, classes, yolo_dataset_dir
        )

        # ==================== 模型训练阶段 ====================
        logging.info("Starting model training...")
        model_yaml_path = Path(f'../pts/{args.model_arch}.yaml')
        model_pt_path = Path(f'../pts/{args.model_arch}.pt')

        if not model_yaml_path.exists() or not model_pt_path.exists():
            raise FileNotFoundError(f"Model files not found: {model_yaml_path} or {model_pt_path}")
        model = YOLO(model_yaml_path).load(model_pt_path)

        training_data_path = str(yolo_dataset_dir.resolve())
        logging.info(f"Passing dataset root directory to trainer: {training_data_path}")
        # 【关键】调用 .train()。如果MLflow已启用，YOLO会在此处自动创建并开始一个run
        results = model.train(
            project=args.output_dir,
            name=args.experiment_name,
            exist_ok=True,
            data=training_data_path,  # <--- 使用目录路径！
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            optimizer=args.optimizer,
            lr0=args.learning_rate,
            dropout=args.dropout,
            seed=args.seed,
            close_mosaic=0,
            val=True,
            verbose=True
        )
        logging.info("Model training completed.")

        # ==================== 手动记录和模型导出阶段 ====================
        if mlflow_enabled:
            # 【关键】获取YOLOv8刚刚创建的run
            last_run = mlflow.last_active_run()
            if not last_run:
                raise RuntimeError("YOLOv8 did not create an MLflow run as expected.")

            run_id = last_run.info.run_id
            logging.info(f"YOLOv8 created MLflow Run with ID: {run_id}. Resuming it to log custom artifacts.")

            # 【关键】使用获取到的run_id，重新进入这个run的上下文，以便追加记录
            with mlflow.start_run(run_id=run_id):
                logging.info("Logging custom artifacts to the existing MLflow run...")

                # 记录数据准备阶段的产物
                mlflow.log_artifact(str(class_dist_file))
                mlflow.log_artifact(str(data_split_file))
                mlflow.log_artifact(str(yaml_path))

                # 导出并记录ONNX模型
                logging.info("Exporting model to ONNX format...")
                try:
                    onnx_path = model.export(format='onnx', imgsz=args.img_size, opset=12)
                    logging.info(f"Model exported to ONNX at: {onnx_path}")
                    mlflow.log_artifact(onnx_path, artifact_path="model")
                except Exception as e:
                    logging.error(f"Failed to export model to ONNX: {e}")

                # YOLOv8已经自动记录了它的训练结果（如混淆矩阵、权重等），无需手动调用`log_artifacts`
                # mlflow.log_artifacts(results.save_dir, artifact_path="yolo_train_results") # 这行是多余的
                logging.info("All artifacts successfully logged to MLflow.")

        return results.save_dir

    except Exception as e:
        logging.error(f"An error occurred during the main process: {e}", exc_info=True)
        # 如果在MLflow run创建后发生错误，为其添加失败标签
        if mlflow_enabled and run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.set_tag("status", "failed")
                # 记录部分错误信息以便在UI中快速查看
                mlflow.log_param("error_message", str(e)[:500])
        raise e

# ... (如果需要，可以在这里保留 argparse 的解析代码)
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train a YOLOv8 classification model with MLflow integration.")
#     parser.add_argument('--data_dir', type=str, required=True, help="Path to the raw image data directory.")
#     parser.add_argument('--output_dir', type=str, default="runs/train", help="Directory to save training outputs and dataset.")
#     parser.add_argument('--model_arch', type=str, default='yolov8n-cls', help="YOLOv8 model architecture (e.g., yolov8n-cls).")
#     parser.add_argument('--experiment_name', type=str, default="YOLOv8_Classification", help="Name of the MLflow experiment.")
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--batch_size', type=int, default=16)
#     parser.add_argument('--img_size', type=int, default=224)
#     parser.add_argument('--test_size', type=float, default=0.2)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--optimizer', type=str, default='AdamW', help="Optimizer to use (e.g., 'SGD', 'Adam', 'AdamW').")
#     parser.add_argument('--learning_rate', type=float, default=0.001)
#     parser.add_argument('--dropout', type=float, default=0.0)
#
#     args = parser.parse_args()
#     main(args)
