import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = resolve_device()


# built a custom linear layer instead of using nn.linear. each weight has its own
# gate that decides if that connection lives or dies. so three params, weight, bias,
# and gate_scores.

class PrunableLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # tried 0 at first but sigmoid(0) = 0.5 so the network started half broken.
        # tried random too but results were all over the place. 3.0 works because
        # sigmoid(3) is about 0.95 so everything starts almost fully on. the l1
        # penalty then slowly shuts off the ones that don't matter.
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.constant_(self.gate_scores, 3.0)

        # kaiming init so the gradients don't blow up right away with relu.
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# the actual network. just a simple mlp, flatten the 3x32x32 image to 3072 and then
# shrink it down through a few layers to 10 classes. added batchnorm and dropout
# because without them accuracy was stuck around 45% and nothing interesting happened.

class PrunableMLP(nn.Module):

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


# this handles the loss. total loss = cross entropy + lambda * l1 penalty.
# the l1 part is just the average of all gate values after sigmoid.
# i was using .sum() before but that gave numbers in the millions which made
# everything confusing. using .mean() keeps it between 0 and 1.

class SparsityEngine:

    def __init__(self, lam: float):
        self.lam = lam
        self._ce = nn.CrossEntropyLoss()

    def compute_penalty(self, model: PrunableMLP) -> torch.Tensor:
        # .sum() gave like 3.8 million, ce was like 2.3. made no sense.
        # .mean() keeps it between 0 and 1 which is way easier to balance.
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


# counts how many gates are basically dead. if a gate is below 0.01 it's
# doing nothing useful so we count it as pruned.

def compute_sparsity(model: PrunableMLP, threshold: float = 0.01) -> float:
    all_gates = []
    with torch.no_grad():
        for layer in model.prunable_layers():
            all_gates.append(torch.sigmoid(layer.gate_scores).cpu().flatten())
    gate_tensor = torch.cat(all_gates)
    return (gate_tensor < threshold).float().mean().item()


# the idea is simple, don't apply any pruning pressure for the first 15 epochs
# so the network can actually learn something. then slowly ramp up the lambda
# over the next 20 epochs. this way the network has good weights before we
# start killing connections, so it can figure out which ones matter.

def get_current_lambda(epoch: int, max_lam: float, warmup_epochs: int = 15, ramp_epochs: int = 20) -> float:
    if epoch <= warmup_epochs:
        return 0.0
    ramp_end = warmup_epochs + ramp_epochs
    if epoch >= ramp_end:
        return max_lam
    # linear ramp from 0 to max_lam
    progress = (epoch - warmup_epochs) / ramp_epochs
    return max_lam * progress


# cifar 10 data loading. basic augmentation, random flips and crops.
# the mean and std values are the standard ones everyone uses for cifar.

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


# regular training loop. the sparsity engine adds the l1 penalty to the loss
# every step, so the gates are always being pushed to close. both the weights
# and gates get updated together.

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
    epochs: int = 50,
) -> tuple[float, float, "PrunableMLP"]:
    label = {0.1: "low", 1.0: "medium", 2.5: "high"}.get(lam, str(lam))
    print(f"\nmax_lambda={lam}  [{label} sparsity pressure]  device={DEVICE}")
    print("-" * 60)

    model  = PrunableMLP().to(DEVICE)
    engine = SparsityEngine(lam)

    # had to give gates a separate, faster learning rate. they start at 3.0 and
    # need to get to about 4.6 for sigmoid to go below 0.01. that's a distance
    # of 7.6. with lr=0.001 and about 5850 steps you can only move 5.85, not
    # enough. bumping gate lr to 0.1 fixes it. weights stay at 0.001 so they
    # don't go crazy.
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
        # update lambda based on warmup schedule. first 15 epochs = no pruning,
        # then it ramps up linearly so the network isn't caught off guard.
        current_lam = get_current_lambda(epoch, max_lam=lam)
        engine.lam  = current_lam

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
                f"  lam {current_lam:.4f}"
                f"  loss {train_loss:.4f} (ce {ce:.4f}, l1 {l1:.4f})"
                f"  acc {acc*100:.2f}%"
                f"  sparsity {sparsity*100:.1f}%"
            )

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    final_acc      = evaluate(model, val_loader)
    final_sparsity = compute_sparsity(model)

    print(f"\n  done, best acc: {final_acc*100:.2f}%  sparsity: {final_sparsity*100:.2f}%")
    return final_acc, final_sparsity, model


# testing three lambda values with warmup. we can push lambda way higher now
# (up to 2.5) because the network gets 15 clean epochs to learn features first
# before the l1 pressure kicks in.

if __name__ == "__main__":
    LAMBDAS = [0.1, 1.0, 2.5]
    EPOCHS  = 50

    train_loader, val_loader = build_loaders(batch_size=256)

    results: dict[float, tuple[float, float, PrunableMLP]] = {}
    for lam in LAMBDAS:
        acc, sparsity, model = run_experiment(lam, train_loader, val_loader, epochs=EPOCHS)
        results[lam] = (acc, sparsity, model)

    print("\n\nresults summary")
    print("=" * 55)
    print(f"  {'lambda':<12} {'pressure':<12} {'test acc':>10} {'sparsity':>12}")
    print(f"  {'-'*48}")
    labels = {0.1: "low", 1.0: "medium", 2.5: "high"}
    for lam, (acc, sparsity, _) in results.items():
        print(f"  {lam:<12.4f} {labels.get(lam, str(lam)):<12} {acc*100:>9.2f}%  {sparsity*100:>10.2f}%")
    print("=" * 55)

    best_lam = max(results, key=lambda k: results[k][0])
    print(f"\nbest lambda by accuracy: {best_lam}")
