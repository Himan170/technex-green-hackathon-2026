"""
GENTLE Fine-tuning: Resume from best model with minimal disruption.
Keep augmentation moderate, lower cls weight, use SGD for stability.
"""

import multiprocessing
multiprocessing.freeze_support()

from ultralytics import YOLO


def main():
    model = YOLO("waste_seg/runs/segment/train2/weights/best.pt")

    results = model.train(
        data="waste_seg/data.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        name="finetune_v3",
        exist_ok=True,

        # Cosine LR for smooth decay
        cos_lr=True,

        # VERY low LR for fine-tuning — don't destroy learned features
        lr0=0.0002,
        lrf=0.01,
        warmup_epochs=2.0,

        # Moderate cls boost — not too aggressive this time
        cls=0.8,
        box=7.5,
        dfl=1.5,

        # Standard augmentation — no mixup/copy-paste (caused memory + convergence issues)
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=10.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,

        close_mosaic=5,
        patience=0,

        plots=True,
        save=True,
        verbose=True,

        # SGD is more stable for fine-tuning than AdamW
        optimizer="SGD",
        weight_decay=0.0005,
        momentum=0.937,

        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,  # No dropout — keep it simple
    )

    print("\n" + "="*60)
    print("GENTLE FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Best model: runs/segment/finetune_v3/weights/best.pt")
    print(f"  Box mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Box mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Mask mAP50:   {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
    print(f"  Mask mAP50-95:{results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")
    print("="*60)


if __name__ == "__main__":
    main()
