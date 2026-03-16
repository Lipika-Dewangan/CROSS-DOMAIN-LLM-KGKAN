import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from peft import LoraConfig, get_peft_model

from config import LLMKGKANConfig
from utils import mmd_loss


class SemanticEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        base_cfg = AutoConfig.from_pretrained(cfg.llm_name)
        self.backbone = AutoModel.from_pretrained(cfg.llm_name, config=base_cfg)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        target_modules = ["query", "key", "value"] if "bert" in cfg.llm_name.lower() else ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="FEATURE_EXTRACTION",
        )
        self.backbone = get_peft_model(self.backbone, peft_cfg)
        self.proj = nn.Linear(self.backbone.config.hidden_size, cfg.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return self.proj(out.last_hidden_state)


class RGCNLayer(nn.Module):
    def __init__(self, hidden_size: int, num_relations: int, dropout: float):
        super().__init__()
        self.rel_weights = nn.Parameter(torch.randn(num_relations, hidden_size, hidden_size) * 0.02)
        self.self_weight = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.num_relations = num_relations

    def forward(self, h: torch.Tensor, dep_rel_ids: torch.Tensor, dep_adj: torch.Tensor) -> torch.Tensor:
        out = self.self_weight(h)
        for r in range(self.num_relations):
            rel_mask = (dep_rel_ids == r) & dep_adj.bool()
            if rel_mask.any():
                neigh = torch.matmul(rel_mask.float(), h)
                deg = rel_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
                neigh = neigh / deg
                out = out + torch.einsum("btd,df->btf", neigh, self.rel_weights[r])
        out = F.relu(out)
        out = self.dropout(out)
        return self.norm(h + out)


class SyntaxEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [RGCNLayer(cfg.hidden_size, cfg.num_dep_relations, cfg.dropout) for _ in range(cfg.rgcn_layers)]
        )

    def forward(self, token_features: torch.Tensor, dep_rel_ids: torch.Tensor, dep_adj: torch.Tensor) -> torch.Tensor:
        h = token_features
        for layer in self.layers:
            h = layer(h, dep_rel_ids, dep_adj)
        return h


class KGEncoder(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.cfg = cfg
        self.ent_emb = nn.Embedding(cfg.num_entities, cfg.kg_emb_dim)
        self.rel_emb = nn.Embedding(cfg.num_kg_relations, cfg.kg_emb_dim)
        self.proj = nn.Linear(cfg.kg_emb_dim, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size)

    def triple_encode(self, heads: torch.Tensor, rels: torch.Tensor, tails: torch.Tensor) -> torch.Tensor:
        h = self.ent_emb(heads)
        r = self.rel_emb(rels)
        t = self.ent_emb(tails)
        if self.cfg.use_distmult:
            return h * r * t
        return h + r + t

    def forward(
        self,
        kg_heads: torch.Tensor,
        kg_rels: torch.Tensor,
        kg_tails: torch.Tensor,
        kg_mask: torch.Tensor,
        kg_token_map: torch.Tensor,
    ) -> torch.Tensor:
        triple_repr = self.triple_encode(kg_heads, kg_rels, kg_tails)
        triple_repr = triple_repr * kg_mask.unsqueeze(-1).float()
        token_rel = torch.einsum("btk,bkd->btd", kg_token_map.float(), triple_repr)
        token_rel = self.proj(token_rel)
        return self.norm(token_rel)


class KANStyleFusion(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        in_dim = hidden_size * 3
        self.alpha = nn.Parameter(torch.randn(in_dim, 4) * 0.02)
        self.proj = nn.Linear(in_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, sem: torch.Tensor, syn: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sem, syn, rel], dim=-1)
        basis = torch.stack([x, torch.tanh(x), torch.sigmoid(x), x * x], dim=-1)
        x_kan = (basis * self.alpha.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        z = self.proj(x_kan)
        z = self.dropout(F.gelu(z))
        return self.norm(z)


class ARGModule(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.g_tr = nn.Linear(hidden_size, hidden_size)
        self.g_pr = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, z: torch.Tensor, sem: torch.Tensor, syn: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        g_tr = torch.sigmoid(self.g_tr(z))
        g_pr = torch.sigmoid(self.g_pr(z))
        out = g_tr * rel + g_pr * sem + (1.0 - g_tr) * syn
        return self.norm(out)


class LLMKGKAN(nn.Module):
    def __init__(self, cfg: LLMKGKANConfig):
        super().__init__()
        self.cfg = cfg
        self.semantic = SemanticEncoder(cfg)
        self.syntax = SyntaxEncoder(cfg)
        self.kg = KGEncoder(cfg)
        self.fusion = KANStyleFusion(cfg.hidden_size, cfg.dropout)
        self.arg = ARGModule(cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def encode(self, batch):
        sem = self.semantic(batch["input_ids"], batch["attention_mask"])
        syn = self.syntax(sem.detach(), batch["dep_rel_ids"], batch["dep_adj"])
        rel = self.kg(
            batch["kg_heads"],
            batch["kg_rels"],
            batch["kg_tails"],
            batch["kg_mask"],
            batch["kg_token_map"],
        )
        z = self.fusion(sem, syn, rel)
        z_prime = self.arg(z, sem, syn, rel)
        return sem, syn, rel, z, z_prime

    def forward(self, batch):
        sem, syn, rel, z, z_prime = self.encode(batch)
        logits = self.classifier(self.dropout(z_prime))

        output = {"logits": logits, "fused": z, "refined": z_prime}

        if "labels" in batch:
            loss_cls = F.cross_entropy(
                logits.view(-1, self.cfg.num_labels),
                batch["labels"].view(-1),
                ignore_index=self.cfg.ignore_index,
            )
            loss_align = logits.new_tensor(0.0)

            if "domain_ids" in batch:
                attn = batch["attention_mask"].bool()
                valid = attn & (batch["labels"] != self.cfg.ignore_index)

                src_mask = (batch["domain_ids"] == 0).unsqueeze(1) & valid
                tgt_mask = (batch["domain_ids"] == 1).unsqueeze(1) & attn

                src_tokens = z[src_mask]
                tgt_tokens = z[tgt_mask]

                if src_tokens.numel() > 0 and tgt_tokens.numel() > 0:
                    loss_align = mmd_loss(src_tokens, tgt_tokens)

            output["loss_cls"] = loss_cls
            output["loss_align"] = loss_align
            output["loss"] = loss_cls + self.cfg.mmd_lambda * loss_align

        return output
