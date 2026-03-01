"""
Aggressive Fine-tuning Script for Waste Segmentation Model
Target: Improve accuracy from ~35% to as high as possible in 20 epochs
Focus: Fix Paper and Metal classification issues
"""

import multiprocessing
multiprocessing.freeze_support()

from ultralytics import YOLO


def main():
    # Resume from the best model we already have
    model = YOLO("waste_seg/runs/segment/train2/weights/best.pt")

    # Aggressive fine-tuning with optimized hyperparameters
    results = model.train(
        data="waste_seg/data.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        name="finetune_v2",
        exist_ok=True,

        # === KEY: Cosine LR for smoother convergence ===
        cos_lr=True,

        # === Learning rate: Low start for fine-tuning ===
        lr0=0.001,
        lrf=0.1,
        warmup_epochs=1.0,

        # === CRITICAL: Boost classification loss to fix Paper/Metal confusion ===
        cls=1.5,      # Was 0.5 — tripled to force better class discrimination
        box=7.5,
        dfl=1.5,

        # === Stronger augmentation to improve generalization ===
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.3,
        degrees=15.0,
        translate=0.2,
        scale=0.7,
        shear=5.0,
        flipud=0.3,
        fliplr=0.5,
        hsv_h=0.02,
        hsv_s=0.8,
        hsv_v=0.5,
        erasing=0.3,

        # === Close mosaic later for cleaner final epochs ===
        close_mosaic=5,

        # === Patience: Don't stop early, we need all 20 epochs ===
        patience=0,

        # === Save & display ===
        plots=True,
        save=True,
        verbose=True,

        # === Optimizer ===
        optimizer="AdamW",
        weight_decay=0.001,

        # === Overlap mask for better segmentation ===
        overlap_mask=True,
        mask_ratio=4,

        # === Dropout for regularization ===
        dropout=0.1,
    )

    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Best model saved at: runs/segment/finetune_v2/weights/best.pt")
    print(f"Final metrics:")
    print(f"  Box mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  Box mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Mask mAP50: {results.results_dict.get('metrics/mAP50(M)', 'N/A')}")
    print(f"  Mask mAP50-95: {results.results_dict.get('metrics/mAP50-95(M)', 'N/A')}")
    print("="*60)
    print("\nTo use the new model, update app.py to load:")
    print('  model = YOLO("runs/segment/finetune_v2/weights/best.pt")')


if __name__ == "__main__":
    main()
