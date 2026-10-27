import torch
import torch.nn as nn

from transformers import AutoModel

class BertClassifierLLM(nn.Module):
    def __init__(self,
        model_id="answerdotai/ModernBERT-base",
        num_labels=18,
        dropout=0.1,
        seed=99,
        attn_imp="sdpa",
        unfrozen_layers=8,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.encoder = AutoModel.from_pretrained(
            model_id,
            attn_implementation=attn_imp,
        )

        layers = self.encoder.layers
        if not 0 <= unfrozen_layers <= len(layers):
            raise ValueError(
                f"unfrozen_layers must be between 0 and {len(layers)}"
            )

        for p in self.encoder.parameters():
            p.requires_grad = False
        for layer in layers[-unfrozen_layers:] if unfrozen_layers else []:
            for p in layer.parameters():
                p.requires_grad = True

        hidden_size = self.encoder.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state

        cls = last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        mean = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.classifier_head(torch.cat([cls, mean], dim=-1))
