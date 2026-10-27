import torch
import torch.nn as nn

from transformers import AutoModel

class BertClassifierLLM(nn.Module):
    def __init__(self,
        model_id="answerdotai/ModernBERT-large",
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

        conv_dim = hidden_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_size, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        feat_dim = conv_dim * 2 + hidden_size
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)

        tokens = (last_hidden * mask).transpose(1, 2)
        conv_out = self.conv(tokens).transpose(1, 2)
        conv_out = conv_out * mask

        conv_mean = conv_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        conv_max = conv_out.masked_fill(mask == 0, torch.finfo(conv_out.dtype).min).max(dim=1).values

        raw_mean = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        feats = torch.cat([conv_mean, conv_max, raw_mean], dim=-1)
        return self.classifier_head(feats)
