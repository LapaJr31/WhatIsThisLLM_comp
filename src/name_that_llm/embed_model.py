import torch
import torch.nn as nn

from transformers import AutoModel

class EmbedModel(nn.Module):
    def __init__(
        self,
        model_id,
        num_labels=18,
        dropout=0.2,
        seed=99,
        attn_imp="sdpa",
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.encoder = AutoModel.from_pretrained(
            model_id,
            attn_implementation=attn_imp,
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden_size = self.encoder.config.hidden_size

        self.classifier_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)

        mean = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        huge = torch.finfo(last_hidden.dtype).min
        masked = last_hidden.masked_fill(mask == 0, huge)
        mx = masked.max(dim=1).values

        return torch.cat([mean, mx], dim=-1)

    def forward(self, input_ids, attention_mask):
        feats = self.encode(input_ids, attention_mask)
        return self.classifier_head(feats)

    def forward_emb(self, emb):
        return self.classifier_head(emb)
