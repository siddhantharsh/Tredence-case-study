import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = resolve_device()


# ----------custom linear layer with learnable gates

class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # init to 3.0 so sigmoid(gate_scores) ≈ 0.95 — all connections near-active
        # before the l1 penalty has had any epochs to accumulate gradient pressure
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.constant_(self.gate_scores, 3.0)

        # kaiming since we have relu activations downstream
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


class PrunableMLP(nn.Module):
    # cifar-10: 3x32x32 = 3072 input features

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            PrunableLinear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

    def prunable_layers(self) -> list[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


# ----------- loss + sparsity stuff

class SparsityEngine:
    # total loss = CE + lambda * sum(|gates|)
    # since gates = sigmoid(...) > 0, the abs is redundant but keeping it explicit

    def __init__(self, lam: float):
        self.lam = lam
        self._ce = nn.CrossEntropyLoss()

    def compute_penalty(self, model: PrunableMLP) -> torch.Tensor:
        # mean across all gates so l1 stays in [0, 1] — readable next to CE loss
        total_sum    = torch.tensor(0.0, device=DEVICE)
        total_params = 0
        for layer in model.prunable_layers():
            gates         = torch.sigmoid(layer.gate_scores)
            total_sum     = total_sum + gates.sum()
            total_params += gates.numel()
        return total_sum / total_params

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: PrunableMLP,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce_loss = self._ce(logits, targets)
        l1_loss = self.compute_penalty(model)
        total_loss = ce_loss + self.lam * l1_loss
        return total_loss, ce_loss, l1_loss


def compute_sparsity(model: PrunableMLP, threshold: float = 0.01) -> float:
    # fraction of gates below threshold = effectively dead connections
    all_gates = []
    with torch.no_grad():
        for layer in model.prunable_layers():
            all_gates.append(torch.sigmoid(layer.gate_scores).cpu().flatten())
    gate_tensor = torch.cat(all_gates)
    return (gate_tensor < threshold).float().mean().item()


# ------------- data loading

def build_loaders(batch_size: int = 256) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


# --------training loop

def train_epoch(
    model: PrunableMLP,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    engine: SparsityEngine,
) -> tuple[float, float, float]:
    model.train()
    total_loss = total_ce = total_l1 = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss, ce, l1 = engine(logits, y, model)
        loss.backward()
        optimizer.step()

        n = x.size(0)
        total_loss += loss.item() * n
        total_ce   += ce.item()   * n
        total_l1   += l1.item()   * n

    N = len(loader.dataset)
    return total_loss / N, total_ce / N, total_l1 / N


@torch.no_grad()
def evaluate(model: PrunableMLP, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += preds.eq(y).sum().item()
        total   += y.size(0)
    return correct / total


def run_experiment(
    lam: float,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
) -> tuple[float, float, "PrunableMLP"]:
    label = {0.1: "low", 0.5: "medium", 1.0: "high"}.get(lam, str(lam))
    print(f"\nlambda={lam}  [{label} sparsity pressure]  device={DEVICE}")
    print("-" * 50)

    model  = PrunableMLP().to(DEVICE)
    engine = SparsityEngine(lam)

    # gates need lr=0.1 to travel 3.0 → -4.6 in 30 epochs; weights stay at 1e-3
    gate_params   = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    weight_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    optimizer = optim.Adam([
        {'params': weight_params, 'lr': 1e-3},
        {'params': gate_params,   'lr': 1e-1},
    ])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc   = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, ce, l1 = train_epoch(model, train_loader, optimizer, engine)
        acc = evaluate(model, val_loader)
        scheduler.step()

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            sparsity = compute_sparsity(model)
            print(
                f"  epoch {epoch:02d}/{epochs}"
                f"  loss {train_loss:.4f} (ce {ce:.4f}, l1 {l1:.4f})"
                f"  acc {acc*100:.2f}%"
                f"  sparsity {sparsity*100:.1f}%"
            )

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    final_acc      = evaluate(model, val_loader)
    final_sparsity = compute_sparsity(model)

    print(f"\n  done — best acc: {final_acc*100:.2f}%  sparsity: {final_sparsity*100:.2f}%")
    return final_acc, final_sparsity, model


#-----------gate distribution plot

def plot_gate_distribution(model: PrunableMLP, save_path: str = "distribution.png"):
    layers = model.prunable_layers()
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4), sharey=False)
    fig.patch.set_facecolor("#0d1117")

    layer_labels = [f"L{i+1}  ({l.in_features}→{l.out_features})" for i, l in enumerate(layers)]
    cmap = plt.get_cmap("plasma")

    with torch.no_grad():
        for ax, layer, label, color_idx in zip(
            axes, layers, layer_labels, np.linspace(0.1, 0.9, len(layers))
        ):
            gates = torch.sigmoid(layer.gate_scores).cpu().numpy().flatten()
            ax.hist(gates, bins=80, color=cmap(color_idx), edgecolor="none", alpha=0.9)
            ax.axvline(0.01, color="#ff4757", linewidth=1.2, linestyle="--", label="threshold (0.01)")
            ax.set_title(label, color="#e6edf3", fontsize=11, pad=8)
            ax.set_xlabel("gate value", color="#8b949e", fontsize=9)
            ax.set_ylabel("count", color="#8b949e", fontsize=9)
            ax.set_facecolor("#161b22")
            ax.tick_params(colors="#8b949e")
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363d")
            ax.legend(fontsize=8, labelcolor="#8b949e", facecolor="#161b22", edgecolor="#30363d")

    fig.suptitle("gate value distribution — best model", color="#e6edf3", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nsaved gate distribution -> {save_path}")


#------------------main

if __name__ == "__main__":
    # scaled up because l1 is now mean-normalized (~0.9), not a sum of millions
    LAMBDAS = [0.1, 0.5, 1.0]
    EPOCHS  = 30

    train_loader, val_loader = build_loaders(batch_size=256)

    results: dict[float, tuple[float, float, PrunableMLP]] = {}
    for lam in LAMBDAS:
        acc, sparsity, model = run_experiment(lam, train_loader, val_loader, epochs=EPOCHS)
        results[lam] = (acc, sparsity, model)

    print("\n\nresults summary")
    print("=" * 55)
    print(f"  {'lambda':<12} {'pressure':<12} {'test acc':>10} {'sparsity':>12}")
    print(f"  {'-'*48}")
    labels = {0.1: "low", 0.5: "medium", 1.0: "high"}
    for lam, (acc, sparsity, _) in results.items():
        print(f"  {lam:<12.4f} {labels.get(lam, str(lam)):<12} {acc*100:>9.2f}%  {sparsity*100:>10.2f}%")
    print("=" * 55)

    best_lam   = max(results, key=lambda k: results[k][0])
    best_model = results[best_lam][2]
    print(f"\nbest lambda by accuracy: {best_lam}")
    plot_gate_distribution(best_model, save_path="distribution.png")
