import torch
import torch.nn as nn

from transformers import AutoModel

class BertClassifierLLM(nn.Module):
    def __init__(self, model_id="answerdotai/ModernBERT-base", num_labels=18, dropout=0.1, seed=99, attn_imp="sdpa"):

        super().__init__()
        torch.manual_seed(seed)

        self.encoder = AutoModel.from_pretrained(
            model_id,
            attn_implementation=attn_imp,
        )
        hidden_size = self.encoder.config.hidden_size

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        tokens = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
        mean_pool = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        cls = tokens[:, 0, :]
        pooled = torch.cat([cls, mean_pool], dim=-1)
        return self.classifier_head(pooled)
