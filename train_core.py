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
    # ... (此处省略，使用你已有的 load_data 函数)
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
    logging.info(f"Class distribution: {class_dist}")

    df = pd.DataFrame({
        "path": X_train + X_test,
        "split": ["train"] * len(X_train) + ["val"] * len(X_test),
        "label": y_train + y_test
    })

    return X_train, X_test, y_train, y_test, classes, class_dist, df


def prepare_yolo_dataset(X_train, X_test, y_train, y_test, classes, dataset_dir: Path):
    """
    创建符合YOLO分类标准的数据集结构和YAML文件。
    """
    logging.info(f"Preparing YOLO dataset at: {dataset_dir}")
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)  # 清理旧数据

    splits = {"train": (X_train, y_train), "val": (X_test, y_test)}

    for split_name, (file_list, labels) in splits.items():
        for img_path_str, label_idx in zip(file_list, labels):
            src = Path(img_path_str)
            class_name = classes[label_idx]
            dst_dir = dataset_dir / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst_dir / src.name)

    # 生成YAML配置文件
    yolo_config = {
        # 使用数据集根目录的绝对路径
        "path": str(dataset_dir.resolve()),
        "train": "train",  # train 目录相对于 path
        "val": "val",  # val 目录相对于 path
        "nc": len(classes),
        # YOLOv8 推荐直接使用类别名称的列表
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
    # ... (此处省略，使用你重构后的 main 函数)
    # 注意：确保这个函数不包含 mlflow.start_run()，因为我们将从外部管理
    try:
        import mlflow
        settings.update({"mlflow": True})
        logging.info("MLflow integration enabled.")
    except ImportError:
        mlflow = None
        settings.update({"mlflow": False})
        logging.warning("MLflow not found. Running without MLflow tracking.")

    mlflow.set_experiment(args.experiment_name)
    # ==================== 关键修改 ====================
    # 使用 'with' 语句来管理 run 的生命周期
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logging.info(f"Starting MLflow Run with ID: {run_id}")
        try:
            data_dir = Path(args.data_dir)
            X_train, X_test, y_train, y_test, classes, class_dist, df = load_data(
                data_dir, args.test_size, args.seed
            )
            with open("class_distribution.json", "w") as f:
                json.dump(class_dist, f, indent=4)
            mlflow.log_artifact("class_distribution.json")
            df.to_csv("data_split.csv", index=False)
            mlflow.log_artifact("data_split.csv")

            yolo_dataset_dir = Path(args.output_dir).resolve() / "yolo_dataset"  # 使用 resolve() 获取绝对路径
            # 确保输出目录存在
            yolo_dataset_dir.parent.mkdir(parents=True, exist_ok=True)

            yaml_path = prepare_yolo_dataset(
                X_train, X_test, y_train, y_test, classes, yolo_dataset_dir
            )
            mlflow.log_artifact(str(yaml_path))

            logging.info("Starting model training...")
            model_yaml_path = Path(f'../pts/{args.model_arch}.yaml')
            model_pt_path = Path(f'../pts/{args.model_arch}.pt')

            # 在日志中确认文件已创建
            if not yaml_path.exists():
                logging.error(f"FATAL: YAML file was not created at {yaml_path}")
                raise FileNotFoundError(f"YAML file not found at {yaml_path}")
            logging.info(f"Verified: YAML file exists at {yaml_path}")
            absolute_yaml_path = yaml_path.as_posix()
            logging.info(f"Using absolute path for dataset config: {absolute_yaml_path}")

            if not model_yaml_path.exists() or not model_pt_path.exists():
                raise FileNotFoundError(f"Model files not found: {model_yaml_path} or {model_pt_path}")

            model = YOLO(model_yaml_path).load(model_pt_path)

            results = model.train(
                project=args.output_dir,
                name=args.experiment_name,
                exist_ok=True,  # 允许覆盖之前的运行
                data=str(yolo_dataset_dir),
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

            logging.info("Exporting model to ONNX format...")
            try:
                onnx_path = model.export(format='onnx', imgsz=args.img_size, opset=12)
                logging.info(f"Model exported to ONNX at: {onnx_path}")
                if mlflow:
                    mlflow.log_artifact(onnx_path, "model")
            except Exception as e:
                logging.error(f"Failed to export model to ONNX: {e}")

            if mlflow:
                logging.info("Logging training results to MLflow...")
                mlflow.log_artifacts(results.save_dir, artifact_path="yolo_train_results")
            return results.save_dir  # 返回保存结果的目录

        except Exception as e:
            logging.error(f"An error occurred during the MLflow run {run_id}: {e}", exc_info=True)
            # 记录错误标签，方便在MLflow UI中筛选失败的运行
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(e)[:250]) # 记录部分错误信息
            # 重新抛出异常，以便 Gradio UI 可以捕获它
            raise e
