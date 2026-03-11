from __future__ import annotations

"""
File: prepare_tools_data_resplit.py

Mô tả: Chương trình thực hiện tiền xử lý, chuẩn hoá và phân chia tập dữ liệu ảnh 
thành các tập train/validation/test phục vụ cho quá trình huấn luyện mô hình.

Cách thức vận hành:
  python prepare_tools_data_resplit.py --input_dir raw_data --output_dir tools_data --val_ratio 0.15 --test_ratio 0.15 --seed 42
"""

import argparse
import csv
import json
import os
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def detect_input_mode(input_dir: Path) -> str:
    """Xác định cấu trúc thư mục đầu vào (dạng thô hay đã phân chia sẵn)."""
    has_train = (input_dir / "train").is_dir()
    has_val = (input_dir / "val").is_dir()
    has_test = (input_dir / "test").is_dir()
    return "already_split" if (has_train or has_val or has_test) else "raw"


def collect_class_to_files(input_dir: Path) -> dict[str, list[Path]]:
    mode = detect_input_mode(input_dir)
    class_to_files: dict[str, list[Path]] = {}

    if mode == "already_split":
        for split in ["train", "val", "test"]:
            split_dir = input_dir / split
            if not split_dir.is_dir():
                continue
            for class_dir in split_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                class_name = class_dir.name
                class_to_files.setdefault(class_name, [])
                for p in class_dir.rglob("*"):
                    if is_image_file(p):
                        class_to_files[class_name].append(p)
    else:
        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            files = [p for p in class_dir.rglob("*") if is_image_file(p)]
            if files:
                class_to_files[class_name] = files

    for k in list(class_to_files.keys()):
        class_to_files[k] = sorted(class_to_files[k], key=lambda p: str(p).lower())
        if len(class_to_files[k]) == 0:
            del class_to_files[k]

    if not class_to_files:
        raise RuntimeError(
            f"Không tìm thấy ảnh hợp lệ trong: {input_dir}\n"
            f"Hãy đảm bảo có folder class và ảnh có đuôi: {sorted(IMAGE_EXTS)}"
        )

    return dict(sorted(class_to_files.items(), key=lambda x: x[0].lower()))


def allocate_split_counts(n: int, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    """
    Tính toán số lượng ảnh phân bổ cho các tập train/val/test dựa trên tỷ lệ đầu vào.
    Đảm bảo tính cân bằng và duy trì tối thiểu các tập dữ liệu cần thiết.
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0  # ưu tiên có val để EarlyStopping hoạt động

    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    # đảm bảo n_train >= 1
    while n_train < 1 and (n_val > 0 or n_test > 0):
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        n_train = n - n_val - n_test

    # nếu dataset đủ lớn mà val/test = 0 thì set tối thiểu 1
    if n >= 3:
        if n_val == 0 and val_ratio > 0:
            n_val = 1
        if n_test == 0 and test_ratio > 0 and (n - n_val) >= 2:
            n_test = 1
        n_train = n - n_val - n_test

    # fix cuối: tổng đúng n
    diff = n - (n_train + n_val + n_test)
    n_train += diff
    if n_train < 1:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    return n_train, n_val, n_test


def clear_split_folders(output_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        d = output_dir / split
        if d.exists():
            shutil.rmtree(d)


def transfer_file(src: Path, dst: Path) -> None:
    """Sao chép tệp tin từ thư mục nguồn sang thư mục đích."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân chia tập dữ liệu thành train/val/test.")
    parser.add_argument("--input_dir", type=str, required=True, help="Thư mục dữ liệu gốc.")
    parser.add_argument("--output_dir", type=str, default="tools_data", help="Thư mục đích sau khi phân chia.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Tỷ lệ tập Validation.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Tỷ lệ tập Test.")
    parser.add_argument("--seed", type=int, default=42, help="Số ngẫu nhiên để tái lập kết quả.")
    parser.add_argument("--clean_output", action="store_true", help="Xoá dữ liệu cũ nếu đã tồn tại.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir không tồn tại: {input_dir}")

    if args.val_ratio < 0 or args.test_ratio < 0 or (args.val_ratio + args.test_ratio) >= 1.0:
        raise ValueError("val_ratio và test_ratio phải >= 0 và tổng < 1.0")

    if output_dir.exists() and args.clean_output:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_split_folders(output_dir)

    class_to_files = collect_class_to_files(input_dir)
    class_names = list(class_to_files.keys())

    rng = random.Random(args.seed)

    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "classes": class_names,
        "counts": {"train": {}, "val": {}, "test": {}, "total": {}},
        "totals": {"train": 0, "val": 0, "test": 0, "all": 0},
    }

    splits_rows: list[dict[str, str]] = []

    print("==> Các lớp dữ liệu được tìm thấy:")
    for c in class_names:
        print(f"  - {c}: {len(class_to_files[c])} images")

    for class_name, files in class_to_files.items():
        files = list(files)
        files = sorted(files, key=lambda p: str(p).lower())
        rng.shuffle(files)

        n = len(files)
        n_train, n_val, n_test = allocate_split_counts(n, args.val_ratio, args.test_ratio)

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:n_train + n_val + n_test]

        stats["counts"]["train"][class_name] = len(train_files)
        stats["counts"]["val"][class_name] = len(val_files)
        stats["counts"]["test"][class_name] = len(test_files)
        stats["counts"]["total"][class_name] = n

        # Khởi tạo thư mục cho các nhãn lớp (đảm bảo tính đồng nhất giữa các tập dữ liệu)
        for split_name in ["train", "val", "test"]:
            (output_dir / split_name / class_name).mkdir(parents=True, exist_ok=True)

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            dst_class_dir = output_dir / split_name / class_name

            for i, src_path in enumerate(split_files):
                ext = src_path.suffix.lower()
                new_name = f"{class_name}_{i:06d}{ext}"
                dst_path = dst_class_dir / new_name

                transfer_file(src_path, dst_path)

                splits_rows.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "src": str(src_path),
                        "dst": str(dst_path.relative_to(output_dir)),
                    }
                )

    # totals
    for split in ["train", "val", "test"]:
        stats["totals"][split] = int(sum(stats["counts"][split].values()))
    stats["totals"]["all"] = int(stats["totals"]["train"] + stats["totals"]["val"] + stats["totals"]["test"])

    # write splits.csv
    csv_path = output_dir / "splits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "class", "src", "dst"])
        writer.writeheader()
        writer.writerows(splits_rows)

    # write stats json
    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n==> Hoàn tất quy trình xử lý dữ liệu.")
    print(f"Output: {output_dir}")
    print(f"- splits.csv: {csv_path}")
    print(f"- dataset_stats.json: {stats_path}")
    print("\n==> Báo cáo tổng kết tập dữ liệu:")
    for split in ["train", "val", "test"]:
        print(f"  {split}: {stats['totals'][split]} images")
    print("Classes:", ", ".join(class_names))


if __name__ == "__main__":
    main()