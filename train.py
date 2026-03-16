import torch
from torch.utils.data import DataLoader

from config import LLMKGKANConfig
from data import DummyABSADataset, collate_fn
from model import LLMKGKAN
from utils import set_seed, token_accuracy


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += token_accuracy(out["logits"], batch["labels"])

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        total_loss += out["loss"].item()
        total_acc += token_accuracy(out["logits"], batch["labels"])

    n = max(len(loader), 1)
    return total_loss / n, total_acc / n


def main():
    cfg = LLMKGKANConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = DummyABSADataset(num_samples=32)
    dev_data = DummyABSADataset(num_samples=16)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LLMKGKAN(cfg).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, device)
        print(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"dev_loss={dev_loss:.4f} dev_acc={dev_acc:.4f}"
        )

    torch.save(model.state_dict(), "llm_kgkan.pt")
    print("Model saved to llm_kgkan.pt")


if __name__ == "__main__":
    main()
