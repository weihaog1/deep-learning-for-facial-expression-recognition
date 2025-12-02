# visualize_data.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, EMOTION_CLASSES, RANDOM_SEED, BATCH_SIZE
from dataset import (
    load_and_clean_labels,
    FacialExpressionDataset,
    get_train_transform,
    get_test_transform,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_class_distribution(df, save_path: Optional[Path] = None, show: bool = False) -> None:
    counts = df["emotion"].value_counts().reindex(EMOTION_CLASSES, fill_value=0)
    counts = counts[counts > 0]

    labels = counts.index.tolist()
    sizes = counts.values.astype(float)
    total = sizes.sum()
    percents = sizes / total * 100.0

    plt.figure(figsize=(9, 6.5))

    wedges, _ = plt.pie(
        sizes,
        labels=None,
        autopct=None,
        startangle=90,
    )
    plt.title("Class Distribution (All Cleaned Samples)")
    plt.axis("equal")

    legend_labels = [
        f"{lab}: {int(cnt)} ({pct:.1f}%)"
        for lab, cnt, pct in zip(labels, sizes, percents)
    ]
    plt.legend(
        wedges,
        legend_labels,
        title="Classes (#images, %)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=11,
        title_fontsize=12,
        frameon=False,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization for visualization.
    x: (3,H,W) in normalized space.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(3, 1, 1)
    x = x * std + mean
    return x.clamp(0.0, 1.0)


def _tensor_to_img(x: torch.Tensor) -> np.ndarray:
    x = _denormalize_imagenet(x).detach().cpu()
    # (3,H,W) -> (H,W,3)
    return x.permute(1, 2, 0).numpy()


def show_random_grid(
    dataset: FacialExpressionDataset,
    title: str,
    n: int = 25,
    cols: int = 5,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    n = min(n, len(dataset))
    idxs = np.random.choice(len(dataset), size=n, replace=False)

    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, idx in enumerate(idxs, start=1):
        x, y = dataset[idx]
        ax = plt.subplot(rows, cols, i)
        ax.imshow(_tensor_to_img(x))
        ax.set_title(EMOTION_CLASSES[int(y)])
        ax.axis("off")

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def show_per_class_grid(
    df,
    transform,
    per_class: int = 4,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    # sample indices per class
    samples = []
    for emo in EMOTION_CLASSES:
        sub = df[df["emotion"] == emo]
        if len(sub) == 0:
            continue
        take = min(per_class, len(sub))
        samples.append(sub.sample(take, random_state=RANDOM_SEED))

    if not samples:
        return

    sample_df = np.concatenate([s[["image_path", "label"]].to_numpy() for s in samples], axis=0)
    # Build a tiny dataset
    temp_df = df.iloc[:0].copy()
    temp_df["image_path"] = []
    temp_df["label"] = []

    # Make a proper DataFrame
    import pandas as pd
    temp_df = pd.DataFrame(sample_df, columns=["image_path", "label"])
    temp_df["label"] = temp_df["label"].astype(int)

    ds = FacialExpressionDataset(temp_df, transform=transform)

    cols = per_class
    rows = len(EMOTION_CLASSES)
    plt.figure(figsize=(cols * 3, rows * 2.6))

    # Iterate class-by-class in order for stable layout
    r = 0
    for emo in EMOTION_CLASSES:
        sub = temp_df[temp_df["label"] == EMOTION_CLASSES.index(emo)]
        if len(sub) == 0:
            r += 1
            continue
        for c in range(min(cols, len(sub))):
            x, y = ds[sub.index[c]]
            ax = plt.subplot(rows, cols, r * cols + c + 1)
            ax.imshow(_tensor_to_img(x))
            ax.set_title(emo if c == 0 else "", fontsize=10)
            ax.axis("off")
        r += 1

    plt.suptitle("Per-Class Samples (balanced peek)", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def show_augmentation_preview(
    df,
    n_variants: int = 8,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    # pick one random image
    row = df.sample(1, random_state=RANDOM_SEED).iloc[0]
    img_path = Path(row["image_path"])
    label = row["emotion"]

    # Create a dataset with just this one sample so we reuse your transforms
    import pandas as pd
    one = pd.DataFrame([{"image_path": str(img_path), "label": int(row["label"])}])

    train_tfm = get_train_transform()

    ds = FacialExpressionDataset(one, transform=train_tfm)

    cols = 4
    rows = int(np.ceil(n_variants / cols))
    plt.figure(figsize=(cols * 3, rows * 3))

    for i in range(n_variants):
        x, _ = ds[0]  # each call re-applies random augmentations
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(_tensor_to_img(x))
        ax.axis("off")

    plt.suptitle(f"Augmentation Preview (label: {label})", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


@torch.no_grad()
def estimate_mean_std(
    df,
    max_batches: int = 30,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate channel mean/std AFTER ToTensor (but BEFORE normalization).
    We build a transform that does Resize + ToTensor only.
    """
    from torchvision import transforms
    from config import IMG_SIZE

    import pandas as pd
    small = df.sample(min(len(df), max_batches * batch_size), random_state=RANDOM_SEED)
    small = small[["image_path", "label"]].copy()
    small["label"] = small["label"].astype(int)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    ds = FacialExpressionDataset(small, transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    n = 0
    mean = torch.zeros(3)
    m2 = torch.zeros(3)

    # Welford over pixels
    for b, (x, _) in enumerate(loader):
        if b >= max_batches:
            break
        # x: (B,3,H,W) -> flatten pixels per channel
        x = x.view(x.size(0), x.size(1), -1)  # (B,3,P)
        pixels = x.permute(1, 0, 2).reshape(3, -1)  # (3, B*P)

        batch_n = pixels.size(1)
        batch_mean = pixels.mean(dim=1)
        batch_var = pixels.var(dim=1, unbiased=False)

        if n == 0:
            mean = batch_mean
            m2 = batch_var * batch_n
            n = batch_n
        else:
            delta = batch_mean - mean
            total_n = n + batch_n
            mean = mean + delta * (batch_n / total_n)
            m2 = m2 + batch_var * batch_n + (delta**2) * (n * batch_n / total_n)
            n = total_n

    var = m2 / max(n, 1)
    std = torch.sqrt(var)
    return mean.numpy(), std.numpy()


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset before training")
    parser.add_argument("--out", type=str, default=str(OUTPUT_DIR / "visualized_data"), help="Output folder for figures")
    parser.add_argument("--show", action="store_true", help="Also show plots interactively")
    parser.add_argument("--n_grid", type=int, default=25, help="How many random samples in grid")
    parser.add_argument("--per_class", type=int, default=4, help="How many samples per class")
    parser.add_argument("--aug", type=int, default=8, help="How many augmentation variants to preview")
    parser.add_argument("--stats", action="store_true", help="Estimate dataset mean/std (slow-ish)")
    parser.add_argument("--num_workers", type=int, default=0, help="Workers for stats estimation loader")
    args = parser.parse_args()

    _set_seed(RANDOM_SEED)
    out_dir = _ensure_dir(Path(args.out))

    df = load_and_clean_labels()

    # 1) class distribution
    plot_class_distribution(df, save_path=out_dir / "class_distribution.png", show=args.show)

    # 2) Random sample grid (test transform so it looks like inference)
    test_ds = FacialExpressionDataset(df[["image_path", "label"]].copy(), transform=get_test_transform())
    show_random_grid(
        test_ds,
        title="Random Samples (test transform: resize + normalize)",
        n=args.n_grid,
        cols=5,
        save_path=out_dir / "random_grid.png",
        show=args.show,
    )

    # 3) Per-class samples (balanced peek)
    show_per_class_grid(
        df,
        transform=get_test_transform(),
        per_class=args.per_class,
        save_path=out_dir / "per_class_grid.png",
        show=args.show,
    )

    # 4) Augmentation preview (train transform)
    show_augmentation_preview(
        df,
        n_variants=args.aug,
        save_path=out_dir / "augmentation_preview.png",
        show=args.show,
    )

    # 5) Optional mean/std estimation
    if args.stats:
        mean, std = estimate_mean_std(df, max_batches=30, batch_size=64, num_workers=args.num_workers)
        txt = out_dir / "estimated_mean_std.txt"
        txt.write_text(
            "Estimated RGB mean/std (Resize+ToTensor only; NOT ImageNet-normalized)\n"
            f"mean: {mean.tolist()}\nstd:  {std.tolist()}\n"
        )
        print(f"[stats] wrote -> {txt}")

    print(f"Saved EDA figures -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
