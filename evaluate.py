import torch
from torch.utils.data import DataLoader

from config import LLMKGKANConfig
from data import DummyABSADataset, collate_fn
from model import LLMKGKAN
from utils import token_accuracy


@torch.no_grad()
def main():
    cfg = LLMKGKANConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = DummyABSADataset(num_samples=16)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = LLMKGKAN(cfg).to(device)
    model.load_state_dict(torch.load("llm_kgkan.pt", map_location=device))
    model.eval()

    total_acc = 0.0
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch)
        total_acc += token_accuracy(out["logits"], batch["labels"])

    print(f"Test token accuracy proxy: {total_acc / max(len(test_loader), 1):.4f}")


if __name__ == "__main__":
    main()
