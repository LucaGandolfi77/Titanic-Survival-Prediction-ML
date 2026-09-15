"""Lab 7 - PyTorch: tensors, Datasets/DataLoaders, MLP, loss, optimisers,
training and evaluation loops (deck pp. 44-56).

Every code example shown in the deck is executed here, in slide order, and then
used to build and train a real model on FashionMNIST (the dataset the deck names
on slide 50).

======  ==========================================================================
Slide   Content implemented here
======  ==========================================================================
45      what PyTorch is: automatic differentiation + GPU acceleration
46-48   tensors: creation from lists, ``torch.zeros``, ``torch.rand``; operations
        ``.to()``, indexing/slicing, element-wise arithmetic, ``torch.matmul``;
        a 3-D tensor (RGB image) and its shape
49      ``Dataset`` / ``DataLoader`` with ``batch_size`` and ``shuffle``
50      ``torchvision`` pre-built dataset: ``FashionMNIST(root, train, download)``
51      transforms: ``ToTensor()`` ([0,255] -> [0,1], HWC -> CHW) and
        ``Normalize(mean, std)``, composed with ``transforms.Compose``
52-53   MLP and ``nn.Linear``: y = Wx + b, and the required input shape
54      activation functions: ``nn.Sigmoid`` and ``nn.ReLU``
55      loss function ``nn.CrossEntropyLoss`` and optimisers ``SGD`` / ``Adam``
56      the training loop (``model.train()``, ``zero_grad``, ``backward``,
        ``step``) and the evaluation loop (``model.eval()``, ``torch.no_grad()``)
======  ==========================================================================

Beyond reproducing the examples, two things are measured because the deck states
them as claims:

* slide 51 claims that ``Normalize`` "speeds up convergence and stabilises
  training" - tested by training the same MLP with and without it;
* the MLP is compared against the classical classifiers already built earlier in
  this report (kNN and a decision tree) on the same data, which is the natural
  question once a neural model is available.

Outputs
-------
results/lab7_tensor_demos.txt          printed output of the slide 46-48 tensor examples
results/fashion_mnist_history.csv      per-epoch loss and accuracy
results/optimizer_comparison.csv       SGD vs Adam vs SGD+momentum
results/normalization_comparison.csv   with vs without Normalize (slide 51 claim)
results/classical_vs_mlp.csv           MLP vs kNN vs decision tree
results/lab7_summary.json              every number, machine-readable
figures/lab7_training_curves.png       loss and accuracy per epoch
figures/lab7_samples.png               FashionMNIST sample images and class names
figures/lab7_confusion.png             MLP confusion matrix on the test set
figures/lab7_normalization.png         the slide 51 convergence claim, tested
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "lab1", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arff_utils import LAB1_DIR, write_csv  # noqa: E402

LAB7_DIR = os.path.join(LAB1_DIR, "..", "lab7")
LAB7_RESULTS = os.path.join(LAB7_DIR, "results")
LAB7_FIGURES = os.path.join(LAB7_DIR, "figures")
DATA_ROOT = os.path.join(LAB7_DIR, "data")

SEED = 42
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def pick_device():
    """Prefer CUDA, then Apple MPS, then CPU (deck slide 45: GPU acceleration)."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Slides 46-48 - tensors
# --------------------------------------------------------------------------


def slides_46_to_48(device) -> str:
    import torch

    L: list[str] = []
    L.append("=" * 74)
    L.append("DECK SLIDES 46-48: tensors, the core data structure in PyTorch")
    L.append("=" * 74)
    L.append("")

    # --- How can we create tensors? (slide 47) ---
    L.append("# From Python lists:")
    L.append("torch.Tensor([[1,2], [3,4]])")
    t_list = torch.Tensor([[1, 2], [3, 4]])
    L.append(f"  -> {t_list.tolist()}   dtype={t_list.dtype}  shape={tuple(t_list.shape)}")

    L.append("# With fixed values:  torch.zeros(2,3)")
    t_zeros = torch.zeros(2, 3)
    L.append(f"  -> shape={tuple(t_zeros.shape)}  sum={t_zeros.sum().item()}")

    L.append("# With random values: torch.rand(2,3)")
    t_rand = torch.rand(2, 3)
    L.append(f"  -> shape={tuple(t_rand.shape)}  values in "
             f"[{t_rand.min().item():.4f}, {t_rand.max().item():.4f}]")
    L.append("")

    # --- What operations can we do with tensors? (slide 47) ---
    L.append("# Move to GPU/accelerator: t.to('cuda')  (here: the available device)")
    moved = t_list.to(device)
    L.append(f"  -> tensor now on '{moved.device.type}' "
             f"(device selected for this run: {device.type})")

    L.append("# Indexing and slicing: t[2], t[:,0]")
    t3 = torch.arange(12).reshape(3, 4)
    L.append(f"  t3 = {t3.tolist()}")
    L.append(f"  t3[2]     -> {t3[2].tolist()}")
    L.append(f"  t3[:,0]   -> {t3[:, 0].tolist()}")
    L.append(f"  t3[1,:2]  -> {t3[1, :2].tolist()}")

    L.append("# Element-wise ops: x+y, x*y")
    x = torch.Tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.Tensor([[10.0, 20.0], [30.0, 40.0]])
    L.append(f"  x + y -> {(x + y).tolist()}")
    L.append(f"  x * y -> {(x * y).tolist()}")

    L.append("# Matrix multiplication: torch.matmul(x, y)")
    L.append(f"  matmul(x, y) -> {torch.matmul(x, y).tolist()}")
    L.append(f"  x @ y        -> {(x @ y).tolist()}   (identical)")
    L.append("")

    # --- Why tensors matter (slide 48): an RGB image is a 3-D tensor ---
    L.append("# An RGB image is a 3-D tensor (C, H, W)")
    img = torch.rand(3, 28, 28)
    L.append(f"  random RGB image shape = {tuple(img.shape)} "
             f"(channels={img.shape[0]}, {img.shape[1]}x{img.shape[2]})")
    L.append(f"  one channel shape      = {tuple(img[0].shape)}")
    L.append("")

    # --- the property the deck highlights: autograd ---
    L.append("# Automatic differentiation (slide 45): gradients come for free")
    w = torch.tensor([2.0], requires_grad=True)
    loss = (w ** 2 + 3 * w).sum()          # d/dw = 2w + 3 = 7 at w = 2
    loss.backward()
    L.append(f"  f(w) = w^2 + 3w at w=2 -> {loss.item():.1f};  autograd gives "
             f"df/dw = {w.grad.item():.1f}  (analytic: 2*2+3 = 7)")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Slides 49-51 - Dataset, DataLoader, transforms
# --------------------------------------------------------------------------


def build_loaders(batch_size: int = 32, normalize: bool = True):
    """Slides 49-51: FashionMNIST + transforms + DataLoader."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import FashionMNIST

    if normalize:
        # slide 51: ToTensor converts [0,255] -> [0.0,1.0] and HWC -> CHW;
        # Normalize(mean, std) then standardises to roughly zero mean / unit std.
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    train_ds = FashionMNIST(root=DATA_ROOT, train=True, download=True,
                            transform=transform)
    test_ds = FashionMNIST(root=DATA_ROOT, train=False, download=True,
                           transform=transform)

    # slide 49: DataLoader supports batching, shuffling and parallel loading
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0)
    return train_loader, test_loader, train_ds, test_ds


# --------------------------------------------------------------------------
# Slides 52-54 - the MLP
# --------------------------------------------------------------------------


def make_mlp(hidden=(256, 128), activation: str = "relu"):
    """Slides 52-54: a feed-forward network of Linear layers + activations.

    The DataLoader yields images shaped ``(batch, 1, 28, 28)`` (slide 51: ToTensor
    produces CHW), but ``nn.Linear`` expects ``(batch, in_features)`` (slide 53).
    The leading ``nn.Flatten`` performs that reshape, turning each image into the
    784-element vector the first Linear layer is declared for.  Omitting it is the
    single most common error in this exercise - the layer then receives a 28x28
    matrix and raises a shape error.
    """
    import torch.nn as nn

    act = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}[activation]
    layers: list = [nn.Flatten()]                      # (B,1,28,28) -> (B,784)
    in_features = 28 * 28
    for h in hidden:
        layers += [nn.Linear(in_features, h), act()]   # slide 53: y = Wx + b
        in_features = h
    layers.append(nn.Linear(in_features, 10))          # 10 FashionMNIST classes
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# Slide 56 - training and evaluation loops
# --------------------------------------------------------------------------


def train_model(model, train_loader, test_loader, device, epochs: int = 5,
                lr: float = 0.01, optimizer_name: str = "sgd",
                momentum: float = 0.0, verbose: bool = True):
    """The exact loop structure of slide 56."""
    import torch
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()                  # slide 55
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(optimizer_name)

    history = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # ------------------- training loop (slide 56) -------------------
        model.train()
        running_loss, n_seen, n_correct = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)
            n_correct += (outputs.argmax(1) == labels).sum().item()
        train_loss = running_loss / n_seen
        train_acc = n_correct / n_seen

        # ------------------- evaluation loop (slide 56) ------------------
        model.eval()
        val_loss, n_val, n_val_correct = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                n_val += labels.size(0)
                n_val_correct += (outputs.argmax(1) == labels).sum().item()
        test_loss = val_loss / n_val
        test_acc = n_val_correct / n_val

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "test_loss": test_loss,
                        "test_acc": test_acc,
                        "elapsed_s": round(time.time() - t0, 2)})
        if verbose:
            print(f"    epoch {epoch}/{epochs}  train_loss={train_loss:.4f} "
                  f"train_acc={train_acc:.4f}  test_loss={test_loss:.4f} "
                  f"test_acc={test_acc:.4f}  [{time.time()-t0:.1f}s]")
    return history


def predict_all(model, loader, device):
    import torch

    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds.append(model(inputs).argmax(1).cpu().numpy())
            truths.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(truths)


# --------------------------------------------------------------------------


def main() -> None:
    import torch

    os.makedirs(LAB7_RESULTS, exist_ok=True)
    os.makedirs(LAB7_FIGURES, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = pick_device()

    out: dict = {"lab": 7, "title": "PyTorch: deep learning in Python",
                 "deck_pages": "44-56"}

    print("=" * 74)
    print("LAB 7 - PyTorch (deck pp. 44-56)")
    print("=" * 74)
    print(f"torch {torch.__version__}   device = {device.type}")
    print()

    # ------------------------------------------------- slides 46-48 tensors
    tensor_text = slides_46_to_48(device)
    print(tensor_text)
    with open(os.path.join(LAB7_RESULTS, "lab7_tensor_demos.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(tensor_text + "\n")
    out["device"] = device.type
    out["torch_version"] = torch.__version__

    # --------------------------------------------- slides 49-51 loaders/data
    print()
    print("=" * 74)
    print("DECK SLIDES 49-51: Datasets, DataLoaders and transforms")
    print("=" * 74)
    train_loader, test_loader, train_ds, test_ds = build_loaders(
        batch_size=32, normalize=True)
    print(f"FashionMNIST train: {len(train_ds)} images   test: {len(test_ds)} images")
    print(f"DataLoader batch_size=32 -> {len(train_loader)} training batches")
    xb, yb = next(iter(train_loader))
    print(f"one batch: inputs {tuple(xb.shape)}  labels {tuple(yb.shape)}")
    print(f"  ToTensor(): pixel range [{xb.min().item():.2f}, {xb.max().item():.2f}] "
          f"after Normalize(0.5, 0.5)  ->  about [-1, 1]")
    L = [f"class {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    print("classes: " + " | ".join(L))
    out["dataset"] = {"train": len(train_ds), "test": len(test_ds),
                      "batch_size": 32, "n_train_batches": len(train_loader),
                      "classes": CLASS_NAMES,
                      "batch_shape": list(xb.shape)}

    # a raw (un-normalised) comparison for the ToTensor range claim
    raw_loader, _, _, _ = build_loaders(batch_size=32, normalize=False)
    xraw, _ = next(iter(raw_loader))
    print(f"  ToTensor() alone: pixel range [{xraw.min().item():.2f}, "
          f"{xraw.max().item():.2f}]  (slide 51's [0,255] -> [0,1] claim)")
    out["dataset"]["tensor_range_raw"] = [float(xraw.min()), float(xraw.max())]
    out["dataset"]["tensor_range_normalized"] = [float(xb.min()), float(xb.max())]

    # sample-images figure
    fig, axes = plt.subplots(2, 5, figsize=(11, 5.0))
    seen = {}
    for i in range(len(train_ds)):
        img, lab = train_ds[i]
        if lab not in seen:
            seen[lab] = img
        if len(seen) == 10:
            break
    for lab, ax in zip(sorted(seen), axes.ravel()):
        ax.imshow(seen[lab].squeeze(), cmap="gray")
        ax.set_title(f"{lab}: {CLASS_NAMES[lab]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Lab 7 - FashionMNIST samples (one per class), as loaded by the "
                 "DataLoader pipeline", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB7_FIGURES, "lab7_samples.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------- slides 52-56 train the MLP
    print()
    print("=" * 74)
    print("DECK SLIDES 52-56: MLP, loss, optimiser, training and eval loops")
    print("=" * 74)

    # slide 53/54 sanity checks of the layer and activation semantics
    import torch.nn as nn

    lin = nn.Linear(4, 3)
    probe = torch.randn(5, 4)
    print(f"nn.Linear(4,3): input {tuple(probe.shape)} -> output "
          f"{tuple(lin(probe).shape)}  (slide 53: y = Wx + b)")
    print(f"  weight shape {tuple(lin.weight.shape)}, bias shape {tuple(lin.bias.shape)}")
    relu, sig = nn.ReLU(), nn.Sigmoid()
    z = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
    print(f"  ReLU({z.tolist()})   = {relu(z).tolist()}")
    print(f"  Sigmoid({z.tolist()}) = "
          f"{[round(v,4) for v in sig(z).tolist()]}")
    out["activation_demo"] = {"input": z.tolist(),
                              "relu": relu(z).tolist(),
                              "sigmoid": [round(v, 4) for v in sig(z).tolist()]}

    EPOCHS = 5
    print(f"\n--- MLP (784-256-128-10, ReLU) trained for {EPOCHS} epochs ---")
    model = make_mlp((256, 128), "relu").to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")
    history = train_model(model, train_loader, test_loader, device,
                          epochs=EPOCHS, lr=0.01, optimizer_name="sgd",
                          momentum=0.0)
    out["mlp"] = {"architecture": "784-256-128-10", "activation": "relu",
                  "n_params": int(n_params), "optimizer": "SGD lr=0.01",
                  "epochs": EPOCHS, "history": history}
    write_csv(os.path.join(LAB7_RESULTS, "fashion_mnist_history.csv"), history,
              ["epoch", "train_loss", "train_acc", "test_loss", "test_acc",
               "elapsed_s"])

    # confusion matrix on the test set
    mlp_pred, mlp_true = predict_all(model, test_loader, device)
    from sklearn.metrics import confusion_matrix, f1_score

    cm = confusion_matrix(mlp_true, mlp_pred)
    mlp_test_acc = float((mlp_pred == mlp_true).mean())
    print(f"\n  final test accuracy = {mlp_test_acc:.4f}")
    print(f"  macro F1 = {f1_score(mlp_true, mlp_pred, average='macro'):.4f}")

    # per-class accuracy - which garment classes does the MLP confuse?
    print("\n  per-class accuracy:")
    per_class = []
    for c in range(10):
        s = mlp_true == c
        acc_c = float((mlp_pred[s] == c).mean())
        per_class.append({"class": CLASS_NAMES[c], "n": int(s.sum()),
                          "accuracy": acc_c})
        print(f"    {CLASS_NAMES[c]:>12}: {acc_c:.4f}  (n={int(s.sum())})")
    out["mlp"]["per_class_accuracy"] = per_class
    out["mlp"]["test_accuracy"] = mlp_test_acc
    out["mlp"]["macro_f1"] = float(f1_score(mlp_true, mlp_pred, average="macro"))
    out["mlp"]["confusion"] = cm.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"MLP confusion matrix (test acc = {mlp_test_acc:.4f})")
    axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true")
    axes[0].set_xticks(range(10)); axes[0].set_xticklabels(CLASS_NAMES, rotation=90,
                                                           fontsize=7)
    axes[0].set_yticks(range(10)); axes[0].set_yticklabels(CLASS_NAMES, fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    order = np.argsort([p["accuracy"] for p in per_class])
    axes[1].barh(range(10), [per_class[i]["accuracy"] for i in order],
                 color="steelblue")
    axes[1].set_yticks(range(10))
    axes[1].set_yticklabels([per_class[i]["class"] for i in order], fontsize=8)
    axes[1].set_xlabel("per-class accuracy")
    axes[1].set_title("Where the MLP errs")
    axes[1].grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB7_FIGURES, "lab7_confusion.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------- optimiser comparison
    print(f"\n--- slide 55: optimiser comparison ({EPOCHS} epochs each) ---")
    opt_rows = []
    torch.manual_seed(SEED)
    m_sgd = make_mlp().to(device)
    h = train_model(m_sgd, train_loader, test_loader, device, epochs=EPOCHS,
                    lr=0.01, optimizer_name="sgd", momentum=0.0)
    opt_rows.append({"optimizer": "SGD", "lr": 0.01, "momentum": 0.0,
                     "final_test_acc": h[-1]["test_acc"],
                     "final_test_loss": h[-1]["test_loss"],
                     "epochs_to_90pct": next((r["epoch"] for r in h
                                              if r["test_acc"] >= 0.90), None)})

    torch.manual_seed(SEED)
    m_mom = make_mlp().to(device)
    h = train_model(m_mom, train_loader, test_loader, device, epochs=EPOCHS,
                    lr=0.01, optimizer_name="sgd", momentum=0.9)
    opt_rows.append({"optimizer": "SGD + momentum 0.9", "lr": 0.01, "momentum": 0.9,
                     "final_test_acc": h[-1]["test_acc"],
                     "final_test_loss": h[-1]["test_loss"],
                     "epochs_to_90pct": next((r["epoch"] for r in h
                                              if r["test_acc"] >= 0.90), None)})

    torch.manual_seed(SEED)
    m_adam = make_mlp().to(device)
    h = train_model(m_adam, train_loader, test_loader, device, epochs=EPOCHS,
                    lr=0.001, optimizer_name="adam")
    opt_rows.append({"optimizer": "Adam", "lr": 0.001, "momentum": None,
                     "final_test_acc": h[-1]["test_acc"],
                     "final_test_loss": h[-1]["test_loss"],
                     "epochs_to_90pct": next((r["epoch"] for r in h
                                              if r["test_acc"] >= 0.90), None)})
    write_csv(os.path.join(LAB7_RESULTS, "optimizer_comparison.csv"), opt_rows,
              ["optimizer", "lr", "momentum", "final_test_acc", "final_test_loss",
               "epochs_to_90pct"])
    out["optimizers"] = opt_rows

    # ---------------------------------- slide 51 claim: does Normalize help?
    print(f"\n--- slide 51 claim: does Normalize 'speed up convergence'? ---")
    norm_rows = []
    for use_norm in (False, True):
        torch.manual_seed(SEED)
        tl, vl, _, _ = build_loaders(batch_size=32, normalize=use_norm)
        m = make_mlp().to(device)
        h = train_model(m, tl, vl, device, epochs=3, lr=0.01,
                        optimizer_name="sgd", verbose=False)
        label = "Normalize(0.5,0.5)" if use_norm else "ToTensor only"
        norm_rows.append({"preprocessing": label, "normalize": use_norm,
                          **{f"epoch{r['epoch']}_test_acc": r["test_acc"] for r in h},
                          **{f"epoch{r['epoch']}_train_loss": r["train_loss"] for r in h},
                          "final_test_acc": h[-1]["test_acc"]})
        print(f"  {label:>20}: 3-epoch test accuracy = "
              f"{[round(r['test_acc'],4) for r in h]}")
    write_csv(os.path.join(LAB7_RESULTS, "normalization_comparison.csv"), norm_rows,
              ["preprocessing", "normalize"] +
              [c for c in norm_rows[0] if c.startswith("epoch")] + ["final_test_acc"])
    out["normalization"] = norm_rows

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for row, colour in zip(norm_rows, ("C3", "C0")):
        accs = [row[f"epoch{e}_test_acc"] for e in (1, 2, 3)]
        losses = [row[f"epoch{e}_train_loss"] for e in (1, 2, 3)]
        axes[0].plot([1, 2, 3], accs, "o-", color=colour,
                     label=row["preprocessing"])
        axes[1].plot([1, 2, 3], losses, "o-", color=colour,
                     label=row["preprocessing"])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Slide 51 claim: effect of Normalize on convergence")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=9)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("training loss")
    axes[1].set_title("Training loss")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=9)
    fig.suptitle("Lab 7 - testing the deck's Normalize claim (same seed, same model)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB7_FIGURES, "lab7_normalization.png"), dpi=150)
    plt.close(fig)

    # ---------------------------------------------- classical vs MLP comparison
    print(f"\n--- MLP vs kNN vs decision tree on FashionMNIST ---")
    X_flat = np.stack([train_ds[i][0].numpy().reshape(-1)
                       for i in range(6000)])
    y_flat = np.array([train_ds[i][1] for i in range(6000)])
    X_te = np.stack([test_ds[i][0].numpy().reshape(-1)
                     for i in range(2000)])
    y_te = np.array([test_ds[i][1] for i in range(2000)])

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    cmp_rows = [{"model": "MLP (784-256-128-10)", "accuracy": mlp_test_acc,
                 "note": "trained on all 60000 images"}]
    for name, clf in (("Decision tree", DecisionTreeClassifier(
                           criterion="entropy", min_samples_leaf=2,
                           random_state=1)),
                      ("kNN k=1", KNeighborsClassifier(n_neighbors=1)),
                      ("kNN k=5", KNeighborsClassifier(n_neighbors=5))):
        t0 = time.time()
        clf.fit(X_flat, y_flat)
        acc = float(clf.score(X_te, y_te))
        cmp_rows.append({"model": name, "accuracy": acc,
                         "note": f"trained on 6000 images, {time.time()-t0:.1f}s"})
        print(f"  {name:>24}: {acc:.4f}   (trained on 6000 images)")
    print(f"  {'MLP':>24}: {mlp_test_acc:.4f}   (trained on 60000 images)")
    write_csv(os.path.join(LAB7_RESULTS, "classical_vs_mlp.csv"), cmp_rows,
              ["model", "accuracy", "note"])
    out["classical_vs_mlp"] = cmp_rows
    out["classical_subset"] = {"n_train": 6000, "n_test": 2000}

    # ------------------------------------------------------------- figures
    ep = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(ep, [r["train_loss"] for r in history], "o-", label="train loss")
    axes[0].plot(ep, [r["test_loss"] for r in history], "s-", label="test loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Loss per epoch"); axes[0].grid(alpha=0.3); axes[0].legend()
    axes[1].plot(ep, [r["train_acc"] for r in history], "o-", label="train accuracy")
    axes[1].plot(ep, [r["test_acc"] for r in history], "s-", label="test accuracy")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy per epoch"); axes[1].grid(alpha=0.3); axes[1].legend()
    fig.suptitle("Lab 7 - MLP training curves on FashionMNIST (slide 56 loop)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(LAB7_FIGURES, "lab7_training_curves.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------- machine-readable summary
    out["summary"] = {
        "architecture": "Flatten -> Linear(784,256) -> ReLU -> Linear(256,128) "
                        "-> ReLU -> Linear(128,10)",
        "n_parameters": int(n_params),
        "device": device.type,
        "epochs": EPOCHS,
        "mlp_test_accuracy": mlp_test_acc,
        "mlp_macro_f1": out["mlp"]["macro_f1"],
        "optimizer_winners": sorted(
            [{"optimizer": r["optimizer"], "test_acc": r["final_test_acc"]}
             for r in opt_rows], key=lambda r: -r["test_acc"]),
        "normalization": {
            "without_final_test_acc": [r for r in norm_rows if not r["normalize"]][0]["final_test_acc"],
            "with_final_test_acc": [r for r in norm_rows if r["normalize"]][0]["final_test_acc"],
            "without_epoch1_test_acc": [r for r in norm_rows if not r["normalize"]][0]["epoch1_test_acc"],
            "with_epoch1_test_acc": [r for r in norm_rows if r["normalize"]][0]["epoch1_test_acc"],
            "epoch1_gain_pp": 100 * (
                [r for r in norm_rows if r["normalize"]][0]["epoch1_test_acc"]
                - [r for r in norm_rows if not r["normalize"]][0]["epoch1_test_acc"]),
            "epoch3_gain_pp": 100 * (
                [r for r in norm_rows if r["normalize"]][0]["final_test_acc"]
                - [r for r in norm_rows if not r["normalize"]][0]["final_test_acc"]),
            "claim_supported": bool(
                [r for r in norm_rows if r["normalize"]][0]["final_test_acc"]
                > [r for r in norm_rows if not r["normalize"]][0]["final_test_acc"]),
        },
        "classical_comparison": cmp_rows,
        "hardest_classes": [p_["class"] for p_ in
                            sorted(per_class, key=lambda x: x["accuracy"])[:3]],
        "easiest_classes": [p_["class"] for p_ in
                            sorted(per_class, key=lambda x: -x["accuracy"])[:3]],
    }

    with open(os.path.join(LAB7_RESULTS, "lab7_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote lab7/results/lab7_summary.json and lab7/figures/*.png")


if __name__ == "__main__":
    main()
