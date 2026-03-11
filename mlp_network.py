from __future__ import annotations

"""
File: mlp_network.py

Mô tả: Chương trình thực hiện huấn luyện mô hình phân loại 4 loại dụng cụ cơ khí 
(Búa, Cờ lê, Kìm, Tua vít) sử dụng kỹ thuật Transfer Learning với kiến trúc MobileNetV2.

Cách thức vận hành đề xuất:
  python mlp_network.py --data_dir tools_data --out_dir runs/output_realtime --img_size 192 --batch_size 64 --cache
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu(memory_growth: bool = True) -> bool:
    """Thiết lập cấu hình GPU và tối ưu hóa bộ nhớ cho quá trình huấn luyện."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False
    if memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
    return True


def maybe_enable_mixed_precision(enable: bool, has_gpu: bool) -> None:
    """Tối ưu hóa hiệu năng tính toán trên GPU nếu được hỗ trợ."""
    if enable and has_gpu:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass


def count_images_per_class(train_dir: Path, class_names: list[str]) -> dict[str, int]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    counts: dict[str, int] = {}
    for c in class_names:
        cdir = train_dir / c
        n = 0
        if cdir.is_dir():
            for p in cdir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    n += 1
        counts[c] = n
    return counts


def compute_class_weights(counts: dict[str, int], class_names: list[str]) -> dict[int, float]:
    total = sum(counts.values())
    num_classes = len(class_names)
    class_weight: dict[int, float] = {}
    for i, c in enumerate(class_names):
        ci = max(1, counts.get(c, 0))
        class_weight[i] = total / (num_classes * ci)
    return class_weight


def build_model_mnv2(
    img_size: int,
    num_classes: int,
    dropout: float,
    l2_reg: float,
    alpha: float = 1.0,
) -> tf.keras.Model:
    """Xây dựng cấu trúc mô hình dựa trên MobileNetV2 kết hợp các lớp tùy chỉnh."""
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    # Cơ chế Tăng cường dữ liệu (Data Augmentation) để cải thiện khả năng tổng quát hóa
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.15),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )
    x = aug(inputs)

    # Tiền xử lý dữ liệu theo chuẩn của MobileNetV2 (đưa giá trị pixel về đoạn [-1, 1])
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preprocess")(x)

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_size, img_size, 3),
        alpha=alpha,
    )
    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)

    kernel_reg = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    # dtype=float32 để an toàn khi bật mixed precision
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=kernel_reg,
        name="pred",
        dtype="float32",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="tools_mnv2")
    return model


def find_backbone_in_model(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenetv2" in layer.name.lower():
            return layer
    raise RuntimeError("Không tìm thấy backbone MobileNetV2 trong model. (Tên layer có thể khác)")


def build_deploy_model(trained_model: tf.keras.Model, img_size: int) -> tf.keras.Model:
    """
    Tạo model để deploy/realtime:
    - Bỏ augmentation
    - Giữ preprocess + backbone + head
    """
    preprocess = trained_model.get_layer("preprocess")
    gap = trained_model.get_layer("gap")
    dropout = trained_model.get_layer("dropout")
    pred = trained_model.get_layer("pred")
    backbone = find_backbone_in_model(trained_model)

    inp = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = preprocess(inp)
    x = backbone(x, training=False)
    x = gap(x)
    x = dropout(x, training=False)
    out = pred(x)

    deploy_model = tf.keras.Model(inp, out, name="tools_mnv2_deploy")
    return deploy_model


def make_callbacks(out_dir: Path, csv_append: bool):
    ckpt_path = out_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(out_dir / "train_history.csv"),
            append=csv_append,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    return callbacks


def plot_history(history: dict[str, list[float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if "loss" in history:
        plt.figure()
        plt.plot(history.get("loss", []))
        if "val_loss" in history:
            plt.plot(history.get("val_loss", []))
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["train", "val"], loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "loss_plot.png", dpi=160)
        plt.close()

    if "accuracy" in history:
        plt.figure()
        plt.plot(history.get("accuracy", []))
        if "val_accuracy" in history:
            plt.plot(history.get("val_accuracy", []))
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["train", "val"], loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / "acc_plot.png", dpi=160)
        plt.close()


def evaluate_and_save_reports(model, test_ds, class_names: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # y_true
    y_true = []
    for _, labels in test_ds:
        y_true.append(np.argmax(labels.numpy(), axis=1))
    y_true = np.concatenate(y_true, axis=0)

    # y_pred
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # confusion matrix image
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    with (out_dir / "confusion_matrix.json").open("w", encoding="utf-8") as f:
        json.dump({"class_names": class_names, "matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)


def save_model_summary(model: tf.keras.Model, out_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lambda s: lines.append(s))
    out_path.write_text("\n".join(lines), encoding="utf-8")




def main() -> None:
    parser = argparse.ArgumentParser(description="Train tool classifier (MobileNetV2).")
    parser.add_argument("--data_dir", type=str, default="tools_data", help="Folder đã có train/val/test.")
    parser.add_argument("--out_dir", type=str, default="runs/tools_mnv2", help="Folder lưu output.")
    parser.add_argument("--img_size", type=int, default=192, help="Khuyến nghị realtime: 160 hoặc 192 (224 cũng được).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--initial_epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)

    ft = parser.add_mutually_exclusive_group()
    ft.add_argument("--fine_tune", dest="fine_tune", action="store_true", help="Bật fine-tune backbone.")
    ft.add_argument("--no_fine_tune", dest="fine_tune", action="store_false", help="Tắt fine-tune backbone.")
    parser.set_defaults(fine_tune=True)

    parser.add_argument("--fine_tune_epochs", type=int, default=20)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    parser.add_argument("--fine_tune_last", type=int, default=40, help="Chỉ mở N layer cuối của backbone (0 = mở hết).")

    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--alpha", type=float, default=1.0, help="MobileNetV2 width multiplier (0.5/0.75/1.0).")

    parser.add_argument("--use_class_weights", action="store_true", help="Tính class_weight nếu lệch lớp.")
    parser.add_argument("--cache", action="store_true", help="Cache dataset (tốn RAM).")

    parser.add_argument("--mixed_precision", action="store_true", help="Bật tối ưu hóa bộ nhớ GPU.")
    args = parser.parse_args()

    set_seed(args.seed)
    has_gpu = configure_gpu()
    maybe_enable_mixed_precision(args.mixed_precision, has_gpu)

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Không thấy train_dir: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Không thấy val_dir: {val_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Không thấy test_dir: {test_dir}")

    # Fix class order ổn định
    class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()], key=lambda s: s.lower())
    if not class_names:
        raise RuntimeError(f"train_dir không có thư mục class: {train_dir}")

    # Load dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        labels="inferred",
        label_mode="categorical",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=class_names,
        labels="inferred",
        label_mode="categorical",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        labels="inferred",
        label_mode="categorical",
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    AUTOTUNE = tf.data.AUTOTUNE

    def _cast(images, labels):
        return tf.cast(images, tf.float32), labels

    train_ds = train_ds.map(_cast, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_cast, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(_cast, num_parallel_calls=AUTOTUNE)

    if args.cache:
        train_ds = train_ds.cache()
        val_ds = val_ds.cache()
        test_ds = test_ds.cache()

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # Save class names
    (out_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # class_weight optional
    class_weight = None
    if args.use_class_weights:
        counts = count_images_per_class(train_dir, class_names)
        class_weight = compute_class_weights(counts, class_names)
        (out_dir / "class_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "class_weight.json").write_text(json.dumps(class_weight, ensure_ascii=False, indent=2), encoding="utf-8")

    # Build model
    model = build_model_mnv2(
        img_size=args.img_size,
        num_classes=len(class_names),
        dropout=args.dropout,
        l2_reg=args.l2,
        alpha=args.alpha,
    )

    save_model_summary(model, out_dir / "model_summary.txt")

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing)

    # ========= Giai đoạn 1 =========
    print("\n========== GIAI ĐOẠN 1: Huấn luyện lớp phân loại (Đóng băng Backbone) ==========")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks1 = make_callbacks(out_dir, csv_append=False)

    t0 = time.time()
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.initial_epochs,
        callbacks=callbacks1,
        class_weight=class_weight,
        verbose=1,
    )
    t1 = time.time()
    print(f"[INFO] Stage 1 time: {(t1 - t0)/60:.2f} minutes")

    # ========= Giai đoạn 2 =========
    history2 = None
    if args.fine_tune:
        print("\n========== GIAI ĐOẠN 2: Tinh chỉnh mạng (Fine-tune backbone) ==========")

        # Giải phóng các lớp của backbone để huấn luyện
        backbone = find_backbone_in_model(model)
        backbone.trainable = True

        # Chỉ cho phép cập nhật N lớp cuối cùng để bảo toàn các đặc trưng cơ bản
        if args.fine_tune_last and args.fine_tune_last > 0:
            for layer in backbone.layers[:-args.fine_tune_last]:
                layer.trainable = False

        # Giữ BatchNorm frozen để ổn định
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=loss_fn,
            metrics=["accuracy"],
        )

        start_epoch = (history1.epoch[-1] + 1) if history1.epoch else 0
        end_epoch = start_epoch + args.fine_tune_epochs

        callbacks2 = make_callbacks(out_dir, csv_append=True)

        t2 = time.time()
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=end_epoch,
            initial_epoch=start_epoch,
            callbacks=callbacks2,
            class_weight=class_weight,
            verbose=1,
        )
        t3 = time.time()
        print(f"[INFO] Stage 2 time: {(t3 - t2)/60:.2f} minutes")
    else:
        print("\n(Fine-tune đang tắt)")

    # Load best model
    ckpt_path = out_dir / "best_model.keras"
    if ckpt_path.exists():
        model = tf.keras.models.load_model(str(ckpt_path))

    # Đánh giá trên tập kiểm thử (Test set)
    print("\n========== ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ ==========")
    test_metrics = model.evaluate(test_ds, verbose=1)
    metrics_dict = {k: float(v) for k, v in zip(model.metrics_names, test_metrics)}
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Test metrics:", metrics_dict)

    # Reports
    evaluate_and_save_reports(model, test_ds, class_names, out_dir)

    # Save final model (full training model)
    model.save(str(out_dir / "final_model.keras"))

    # Export deploy_model (no augmentation)
    deploy_model = build_deploy_model(model, img_size=args.img_size)
    deploy_model.save(str(out_dir / "deploy_model.keras"))


    # Merge history để plot
    merged: dict[str, list[float]] = {}

    def _merge(hist):
        if hist is None:
            return
        for k, v in hist.history.items():
            merged.setdefault(k, [])
            merged[k].extend([float(x) for x in v])

    _merge(history1)
    _merge(history2)

    (out_dir / "history.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_history(merged, out_dir)

    print("\n==> Done.")
    print(f"Saved best model : {ckpt_path}")
    print(f"Saved deploy model: {out_dir / 'deploy_model.keras'}")
    print(f"Saved final model: {out_dir / 'final_model.keras'}")
    print(f"Reports & plots  : {out_dir}")


if __name__ == "__main__":
    main()