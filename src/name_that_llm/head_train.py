import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from .embed_model import EmbedModel
from .embed_dataset import EmbedDataset
from .tracking import session as tracking_session, fold as tracking_fold


class HeadTrainer:
    def __init__(
        self,
        model: EmbedModel,
        dataset: EmbedDataset,
        checkpoint_dir="checkpoint",
        seed=99,
        batch_size=512,
        lr=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
    ):
        self.model = model
        self.dataset = dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.early_stop = 20
        self.log_every = 1

        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        self._session_run_id = ""

    def _loaders(self, train_index, val_index):
        train_loader = DataLoader(
            Subset(self.dataset, train_index),
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(self.dataset, val_index),
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader

    @staticmethod
    def train_epoch(model, optimizer, loader, loss_fn, device, scheduler=None):
        model.train()
        total_loss, correct = 0, 0
        steps = 0
        start = time.perf_counter()
        for emb, y in loader:
            emb, y = emb.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model.forward_emb(emb)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            steps += 1
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        elapsed = time.perf_counter() - start
        n = len(loader.dataset)
        stats = {
            "steps": steps,
            "sec_per_epoch": elapsed,
        }
        return total_loss / n, correct / n, stats

    @staticmethod
    def evaluate(model, loader, loss_fn, device):
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for emb, y in loader:
                emb, y = emb.to(device), y.to(device)
                logits = model.forward_emb(emb)
                total_loss += loss_fn(logits, y).item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    @staticmethod
    def eval_metrics(model, loader, device):
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for emb, y in loader:
                logits = model(emb.to(device))
                probs = torch.softmax(logits.float(), dim=-1).cpu()
                all_probs.append(probs)
                all_preds.append(logits.argmax(1).cpu())
                all_labels.append(y)
        labels = torch.cat(all_labels).numpy()
        preds = torch.cat(all_preds).numpy()
        probs = torch.cat(all_probs).numpy()
        report = classification_report(labels, preds, zero_division=0)
        cm = confusion_matrix(labels, preds)
        try:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        except ValueError:
            auc = float("nan")
        return report, cm, auc

    def train(self, epochs=100):
        label_ids = self.dataset.label_ids.numpy()
        fold_results, best_epochs = [], []
        global_best = 0.0

        with tracking_session() as sess:
            self._session_run_id = sess.run_id
            sess.log_params({
                "model_id": self.model.encoder.config._name_or_path,
                "trainer": "HeadTrainer",
                "batch_size": self.batch_size,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "label_smoothing": self.label_smoothing,
                "early_stop": self.early_stop,
                "seed": self.seed,
                "epochs": epochs,
                "dataset_rows": len(self.dataset),
            })

            splits = list(self.skf.split(np.zeros(len(label_ids)), label_ids))
            for i, (train_index, val_index) in enumerate(splits):
                acc, auc, best_epoch = self._train_fold(i, train_index, val_index, epochs)
                fold_results.append((i, acc, auc))
                best_epochs.append(best_epoch)
                global_best = max(global_best, acc)

            accs = [a for _, a, _ in fold_results]
            aucs = [a for _, _, a in fold_results if not np.isnan(a)]
            sess.log_summary({
                "mean_val_acc": float(np.mean(accs)),
                "std_val_acc": float(np.std(accs)),
                "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "global_best_acc": global_best,
            })

        self._print_summary(fold_results, best_epochs, global_best)
        return fold_results, best_epochs, global_best

    def _train_fold(self, i, train_index, val_index, epochs):
        with tracking_fold(self._session_run_id, i) as tr:
            return self._train_fold_inner(tr, i, train_index, val_index, epochs)

    def _train_fold_inner(self, tr, i, train_index, val_index, epochs):
        train_loader, val_loader = self._loaders(train_index, val_index)

        print(f"\n{'='*70}", flush=True)
        print(f"FOLD {i+1} -- train {len(train_index)} | val {len(val_index)}", flush=True)
        print(f"{'='*70}", flush=True)

        # reset head to init state each fold (encoder is frozen, no need to re-init it)
        for m in self.model.classifier_head.modules():
            reset = getattr(m, "reset_parameters", None)
            if callable(reset) and not isinstance(m, nn.LayerNorm):
                reset()

        loss_fn = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        optimizer = torch.optim.AdamW(
            self.model.classifier_head.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader)
        )

        best_val_acc, best_epoch, no_improve = 0.0, 0, 0
        fold_ckpt = self.checkpoint_dir / f"fold{i}_best.pt"
        torch.save(self.model.classifier_head.state_dict(), fold_ckpt)

        print(
            f"  {'Ep':>4} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Va Loss':>8} | {'Va Acc':>7} | {'s/ep':>6}",
            flush=True,
        )
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc, timing = self.train_epoch(
                self.model, optimizer, train_loader, loss_fn, self.device, scheduler
            )
            va_loss, va_acc = self.evaluate(self.model, val_loader, loss_fn, self.device)

            tr.log_epoch(epoch, {
                "tr_loss": tr_loss, "tr_acc": tr_acc,
                "va_loss": va_loss, "va_acc": va_acc,
            })

            if va_acc > best_val_acc:
                best_val_acc, best_epoch, no_improve = va_acc, epoch, 0
                torch.save(self.model.classifier_head.state_dict(), fold_ckpt)
            else:
                no_improve += 1

            if epoch % self.log_every == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"  {epoch:>4} | {tr_loss:>8.4f} | {tr_acc:>7.3%} | "
                    f"{va_loss:>8.4f} | {va_acc:>7.3%} | {timing['sec_per_epoch']:>6.2f}"
                    + ("  *" if no_improve == 0 else ""),
                    flush=True,
                )

            if no_improve >= self.early_stop:
                print(f"\n  Early stop at epoch {epoch}", flush=True)
                break

        self.model.classifier_head.load_state_dict(torch.load(fold_ckpt, weights_only=True))
        report, cm, auc = self.eval_metrics(self.model, val_loader, self.device)

        print(f"\n  Best val acc : {best_val_acc:.3%}  (epoch {best_epoch})", flush=True)
        print(f"  ROC-AUC (OvR macro) : {auc:.4f}\n", flush=True)

        tr.log_summary({"best_val_acc": best_val_acc, "auc": auc, "best_epoch": best_epoch})
        tr.log_artifact_text(
            "classification_report.txt",
            report + "\n\n" + "\n".join(
                f"{idx:>3} = {name}" for idx, name in enumerate(self.dataset.label_names)
            ),
        )
        tr.log_artifact_text(
            "confusion_matrix.csv",
            "true/pred," + ",".join(self.dataset.label_names) + "\n"
            + "\n".join(
                f"{self.dataset.label_names[r]}," + ",".join(str(int(v)) for v in row)
                for r, row in enumerate(cm)
            ),
        )
        return best_val_acc, auc, best_epoch

    def _print_summary(self, fold_results, best_epochs, global_best):
        accs = [a for _, a, _ in fold_results]
        aucs = [a for _, _, a in fold_results if not np.isnan(a)]
        print(f"\n{'='*55}", flush=True)
        print("CROSS-VALIDATION SUMMARY", flush=True)
        print(f"{'='*55}", flush=True)
        for (fold, acc, auc), epoch in zip(fold_results, best_epochs):
            print(f"  fold {fold}: acc {acc:.3%} | auc {auc:.4f} | best epoch {epoch}", flush=True)
        print(f"  mean acc : {np.mean(accs):.3%} ± {np.std(accs):.3%}", flush=True)
        if aucs:
            print(f"  mean auc : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
