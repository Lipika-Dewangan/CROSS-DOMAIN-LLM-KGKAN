import torch
from torch.utils.data import Dataset


class DummyABSADataset(Dataset):
    def __init__(self, num_samples=32, seq_len=32, num_labels=7, num_dep_relations=40, num_entities=10000, num_kg_relations=500, num_triples=8):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_labels = num_labels
        self.num_dep_relations = num_dep_relations
        self.num_entities = num_entities
        self.num_kg_relations = num_kg_relations
        self.num_triples = num_triples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = self.seq_len
        k = self.num_triples
        sample = {
            "input_ids": torch.randint(0, 2000, (t,)),
            "attention_mask": torch.ones(t).long(),
            "labels": torch.randint(0, self.num_labels, (t,)),
            "domain_ids": torch.tensor(0 if idx % 2 == 0 else 1).long(),
            "dep_rel_ids": torch.randint(0, self.num_dep_relations, (t, t)),
            "dep_adj": torch.randint(0, 2, (t, t)).bool(),
            "kg_heads": torch.randint(0, self.num_entities, (k,)),
            "kg_rels": torch.randint(0, self.num_kg_relations, (k,)),
            "kg_tails": torch.randint(0, self.num_entities, (k,)),
            "kg_mask": torch.ones(k).long(),
            "kg_token_map": torch.randint(0, 2, (t, k)).float(),
        }
        return sample


def collate_fn(batch):
    keys = batch[0].keys()
    output = {}
    for key in keys:
        output[key] = torch.stack([item[key] for item in batch], dim=0)
    return output
