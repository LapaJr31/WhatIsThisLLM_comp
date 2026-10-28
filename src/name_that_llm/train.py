import time

import torch
import numpy as np

from .model import BertClassifierLLM
from .dataset import LLMDataset
from .tracking import session as tracking_session, fold as tracking_fold

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold

from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path

from typing import Any, cast, Iterator


class BucketBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths, batch_size, shuffle=True, seed=99):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        order = np.argsort(lengths, kind="stable")
        self.megabatches = [
            order[i : i + batch_size * 50]
            for i in range(0, len(order), batch_size * 50)
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for mega in self.megabatches:
            mega = mega.copy()
            if self.shuffle:
                perm = rng.permutation(len(mega))
                mega = mega[perm]
            for i in range(0, len(mega), self.batch_size):
                batch = mega[i : i + self.batch_size]
                if self.shuffle:
                    batch = batch[rng.permutation(len(batch))]
                batches.append(batch.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

class Trainer:
    def __init__(
        self,
        model: BertClassifierLLM,
        dataset: LLMDataset,
        checkpoint_dir="checkpoint/best_model.pt",
        seed=99,
    ):

        self.checkpoint = checkpoint_dir
        self.seed = seed
        Path(checkpoint_dir).parent.mkdir(parents=True, exist_ok=True)
        self.log_every = 1
        self.early_stop = 20

        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.base_model = cast(Any, model.to(self.device))

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.dataset = dataset
        self._init_state = None

        self.skf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)

    @staticmethod
    def make_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def train_epoch(model, optimizer, loader, loss_fn, device, dtype, scheduler=None):
        model.train()
        total_loss, correct = 0, 0
        steps = 0
        start = time.perf_counter()
        for input_ids, attention_mask, y in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits = model(input_ids, attention_mask)
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
            "sec_per_step": elapsed / steps if steps else 0.0,
            "samples_per_sec": n / elapsed if elapsed else 0.0,
        }
        return total_loss / n, correct / n, stats

    @staticmethod
    def evaluate(model, loader, device, loss_fn, dtype):
        model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for input_ids, attention_mask, y in loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                y = y.to(device)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(input_ids, attention_mask)
                    loss = loss_fn(logits, y)
                total_loss += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
        n = len(loader.dataset)
        return total_loss / n, correct / n

    @staticmethod
    def eval_metrics(model, device, loader, dtype):
        model.eval()
        all_labels, all_preds, all_probs = [], [], []
        with torch.no_grad():
            for input_ids, attention_mask, y in loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(input_ids, attention_mask)
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

    def train(self, epochs):
        print("\nTraining!", flush=True)
        fold_results = []
        best_epochs = []
        global_best = 0.0

        with tracking_session() as sess:
            print("  tracking session ready", flush=True)
            sess.log_params({
                "model_id": self.base_model.encoder.config._name_or_path,
                "batch_size": 64,
                "encoder_lr": 2e-5,
                "head_lr": 1e-3,
                "label_smoothing": 0.1,
                "weight_decay": 0.1,
                "warmup_frac": 0.06,
                "early_stop": self.early_stop,
                "seed": self.seed,
                "epochs": epochs,
                "unfrozen_layers": self.base_model.unfrozen_layers,
            })

            print("  computing CV splits...", flush=True)
            splits = list(self.skf.split(self.dataset, self.dataset.label_ids))
            print(f"  {len(splits)} folds ready", flush=True)

            for i, (train_index, test_index) in enumerate(splits):
                fold_acc, fold_auc, fold_best_epoch = self._train_fold(
                    sess, i, train_index, test_index, epochs
                )
                fold_results.append((i, fold_acc, fold_auc))
                best_epochs.append(fold_best_epoch)
                if fold_acc > global_best:
                    global_best = fold_acc

            accs = [acc for _, acc, _ in fold_results]
            aucs = [a for _, _, a in fold_results if not np.isnan(a)]
            sess.log_summary({
                "mean_val_acc": float(np.mean(accs)),
                "std_val_acc": float(np.std(accs)),
                "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
                "global_best_acc": global_best,
            })

        self._print_summary(fold_results, best_epochs, global_best)
        return fold_results, best_epochs, global_best

    def _train_fold(self, sess, i, train_index, test_index, epochs):
        with tracking_fold(sess.run_id, i) as tr:
            return self._train_fold_inner(tr, i, train_index, test_index, epochs)

    def _train_fold_inner(self, tr, i, train_index, test_index, epochs):
        train_ds = Subset(self.dataset, train_index)
        test_ds = Subset (self.dataset, test_index)

        train_batch_sampler = BucketBatchSampler(
            self.dataset.lengths[train_index],
            batch_size=64,
            shuffle=True,
            seed=self.seed + i,
        )
        train_loader = DataLoader(
            dataset = train_ds,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )

        print(f"\n{'='*80}", flush=True)
        print(
            f"FOLD {i+1}/3 -- train {len(train_index)} | val {len(test_index)}",
            flush=True,
        )
        print(f"\n{'='*80}", flush=True)

        if self._init_state is None:
            self._init_state = {
                k: v.detach().cpu().clone() for k, v in self.base_model.state_dict().items()
            }
        model = self.base_model
        model.load_state_dict(self._init_state)

        compiled = None
        if self.device.type == "cuda":
            print("  compiling model for this fold...", flush=True)
            compiled = torch.compile(model)

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        head_params = list(model.classifier_head.parameters())
        head_ids = {id(p) for p in head_params}
        encoder_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": encoder_params, "lr": 2e-5},
                {"params": head_params, "lr": 1e-3},
            ],
            fused=self.device.type == "cuda",
            weight_decay=0.1,
        )

        steps_per_epoch = len(train_batch_sampler)
        total_steps = steps_per_epoch * epochs
        warmup_steps = max(1, int(0.06 * total_steps))
        scheduler = self.make_scheduler(optimizer, warmup_steps, total_steps)

        best_val_acc = 0.0
        best_epoch = 0
        fold_ckpt = Path(self.checkpoint).parent / f"fold{i}_best.pt"
        torch.save(model.state_dict(), fold_ckpt)
        no_improve = 0

        print(
        f"  {'Epoch':>5} | {'Tr Loss':>8} | {'Tr Acc':>7} | {'Va Loss':>8} | {'Va Acc':>7} | "
        f"{'Steps':>4} | {'s/step':>6} | {'samp/s':>7}",
        flush=True,
        )
        print(f"  {'-' * 78}", flush=True)

        for epoch in range(1, epochs + 1):
            train_batch_sampler.set_epoch(epoch)
            tr_loss, tr_acc, timing = self.train_epoch(
                model=compiled if compiled is not None else model,
                optimizer=optimizer,
                loader=train_loader,
                loss_fn=loss_fn,
                device=self.device,
                dtype=self.dtype,
                scheduler=scheduler,
            )
            va_loss, va_acc = self.evaluate(
                model=compiled if compiled is not None else model,
                loader=test_loader,
                loss_fn=loss_fn,
                device=self.device,
                dtype=self.dtype,
            )
            tr.log_epoch(epoch, {
                "tr_loss": tr_loss, "tr_acc": tr_acc,
                "va_loss": va_loss, "va_acc": va_acc,
                "sec_per_step": timing["sec_per_step"],
            })

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = epoch
                no_improve = 0

                torch.save(
                    {
                        "fold": i,
                        "epoch": epoch,
                        "val_acc": va_acc,
                        "model_state": model.state_dict(),  # raw module: no _orig_mod prefix
                    },
                    fold_ckpt,
                )
            else:
                no_improve += 1

            if epoch % self.log_every == 0 or epoch == 1 or epoch == epochs:
                print(
                    f"  {epoch:>5} | {tr_loss:>8.4f} | {tr_acc:>7.3%} | "
                    f"{va_loss:>8.4f} | {va_acc:>7.3%} | "
                    f"{timing['steps']:>4} steps | {timing['sec_per_step']:>6.3f} s/step | "
                    f"{timing['samples_per_sec']:>7.1f} samp/s"
                    + ("  *" if no_improve == 0 else ""),
                    flush=True,
                )

            if no_improve >= self.early_stop:
                print(
                    f"\n  Early stop at epoch {epoch} (no improvement for {self.early_stop} epochs)",
                    flush=True,
                )
                break

        ckpt = torch.load(fold_ckpt, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        report, cm, auc = self.eval_metrics(
            model=compiled if compiled is not None else model,
            loader=test_loader,
            device=self.device,
            dtype=self.dtype,
        )

        print(f"\n  Best val acc : {best_val_acc:.3%}  (epoch {best_epoch})", flush=True)
        print(f"  ROC-AUC (OvR macro) : {auc:.4f}", flush=True)
        print(f"\n  Classification report:\n", flush=True)
        for line in report.splitlines():
            print(f"    {line}", flush=True)

        print(f"\n  Label map:", flush=True)
        for idx, name in enumerate(self.dataset.label_names):
            print(f"    {idx:>3} = {name}", flush=True)

        print(f"\n  Confusion matrix (rows=true, cols=pred, ids from label map):", flush=True)
        header = "    " + " ".join(f"{idx:>4}" for idx in range(len(self.dataset.label_names)))
        print(header, flush=True)
        for idx, row in enumerate(cm):
            print(f"    {idx:>3} " + " ".join(f"{v:>4}" for v in row), flush=True)

        tr.log_summary({"best_val_acc": best_val_acc, "auc": auc, "best_epoch": best_epoch})
        tr.log_artifact_text(
            "classification_report.txt",
            report + "\n\n" + "\n".join(
                f"{idx:>3} = {name}"
                for idx, name in enumerate(self.dataset.label_names)
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
        del optimizer
        del compiled
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return best_val_acc, auc, best_epoch

    def _print_summary(self, fold_results, best_epochs, global_best):
        accs = [acc for _, acc, _ in fold_results]
        aucs = [a for _, _, a in fold_results if not np.isnan(a)]

        print(f"\n{'='*55}", flush=True)
        print("CROSS-VALIDATION SUMMARY", flush=True)
        print(f"{'='*55}", flush=True)
        for (fold, acc, auc), epoch in zip(fold_results, best_epochs):
            print(f"  fold {fold}: acc {acc:.3%} | auc {auc:.4f} | best epoch {epoch}", flush=True)
        print(f"  {'-' * 50}", flush=True)
        print(f"  mean acc : {np.mean(accs):.3%} ± {np.std(accs):.3%}", flush=True)
        if aucs:
            print(f"  mean auc : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}", flush=True)
        print(f"  global best acc : {global_best:.3%} (saved to {self.checkpoint})", flush=True)
