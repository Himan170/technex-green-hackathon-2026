"""
HEAD-ONLY Fine-tune: Freeze backbone, train only detection head.
Directly targets Paper/Metal misclassification.
"""
import multiprocessing
multiprocessing.freeze_support()
from ultralytics import YOLO

def main():
    model = YOLO("waste_seg/runs/segment/train2/weights/best.pt")

    results = model.train(
        data="waste_seg/data.yaml",
        epochs=15,
        imgsz=640,
        batch=4,
        device=0,
        workers=0,        # ZERO workers to avoid memory crash on Windows
        name="head_only",
        exist_ok=True,
        freeze=10,        # Freeze backbone layers 0-9
        cos_lr=True,
        lr0=0.005,
        lrf=0.01,
        warmup_epochs=1.0,
        cls=1.0,
        box=7.5,
        dfl=1.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        close_mosaic=3,
        patience=0,
        plots=True,
        save=True,
        verbose=True,
        optimizer="SGD",
        weight_decay=0.0005,
        momentum=0.937,
        overlap_mask=True,
        mask_ratio=4,
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model: runs/segment/head_only/weights/best.pt")
    print(f"  Box mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Box mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("="*60)

if __name__ == "__main__":
    main()
