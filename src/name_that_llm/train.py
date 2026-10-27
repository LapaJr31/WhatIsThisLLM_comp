from click.core import batch
import time

import torch
import numpy as np
import copy

from .model import BertClassifierLLM
from .dataset import LLMDataset

from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold

from torch import nn, optim
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

            # test reduce_overhead afterwards
            self.model = torch.compile(self.base_model, mode="default")
            torch.cuda.synchronize()
        else:
            self.model = self.base_model

        self.dataset = dataset
        self._probe_opt = None
        self.batch_size = self.max_batch(self.probe_batch)
        torch.cuda.synchronize()
        print(f"\nBest batch size = {self.batch_size}\n")

        self.skf = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)

    def max_batch(self, probe, limit=4096, safety=0.75):
        limit = min(limit, len(self.dataset))
        if not self._probe_safe(probe, 1):
            raise RuntimeError("model cannot fit even a single sample")
        if self._probe_safe(probe, limit):
            return limit

        lo, hi = 1, limit
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._probe_safe(probe, mid):
                lo = mid
            else:
                hi = mid
        return max(1, int(lo * safety))

    @staticmethod
    def make_scheduler(optimizer, warmup_steps, total_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        return LambdaLR(optimizer, lr_lambda)

    def _probe_safe(self, probe, batch_size):
        try:
            probe(batch_size)
            torch.cuda.synchronize(self.device)
            return True
        except torch.cuda.OutOfMemoryError:
            return False
        finally:
            torch.cuda.empty_cache()

    def probe_batch(self, batch_size):
        probe_dl = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        batch = next(iter(probe_dl))
        input_ids, attention_mask, labels = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            batch[2].to(self.device),
        )

        if self._probe_opt is None:
            self._probe_opt = torch.optim.AdamW(
                (p for p in self.model.parameters() if p.requires_grad),
                lr=1e-4,
                fused=self.device.type == "cuda",
            )

        self.model.train()
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            logits = self.model(input_ids, attention_mask)
            loss = nn.functional.cross_entropy(logits, labels)

        loss.backward()
        self._probe_opt.step()
        self._probe_opt.zero_grad(set_to_none=True)

        del batch, input_ids, attention_mask, labels, logits, loss
        torch.cuda.empty_cache()

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
        fold_results = []
        best_epochs = []
        global_best = 0.0
        for i, (train_index, test_index) in enumerate(self.skf.split(self.dataset, self.dataset.label_ids)):
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
                # batch_size=self.batch_size,
                batch_size=64,
                shuffle=False,
                pin_memory=True,
                num_workers=4,
            )

            print(f"\n{'='*80}")
            print(
                f"FOLD {i+1}/3 -- train {len(train_index)} | val {len(test_index)}"
            )
            print(f"\n{'='*80}")

            model = copy.deepcopy(self.base_model).to(self.device)

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
            f"{'Steps':>4} | {'s/step':>6} | {'samp/s':>7}"
            )
            print(f"  {'-' * 78}")

            history = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}

            for epoch in range(1,epochs +1):
                train_batch_sampler.set_epoch(epoch)
                tr_loss, tr_acc, timing = self.train_epoch(
                    model=model,
                    optimizer=optimizer,
                    loader=train_loader,
                    loss_fn=loss_fn,
                    device=self.device,
                    dtype=self.dtype,
                    scheduler=scheduler,
                )
                va_loss, va_acc = self.evaluate(
                    model=model,
                    loader=test_loader,
                    loss_fn=loss_fn,
                    device=self.device,
                    dtype=self.dtype,
                )
                history["tr_loss"].append(tr_loss)
                history["tr_acc"].append(tr_acc)
                history["va_loss"].append(va_loss)
                history["va_acc"].append(va_acc)

                if va_acc > best_val_acc:
                    best_val_acc = va_acc
                    best_epoch = epoch
                    no_improve = 0

                    torch.save(
                        {
                            "fold": i,
                            "epoch": epoch,
                            "val_acc": va_acc,
                            "model_state": model.state_dict(),
                        },
                        fold_ckpt,
                    )
                    if va_acc > global_best:
                        global_best = va_acc
                        torch.save(
                            {
                                "fold": i,
                                "epoch": epoch,
                                "val_acc": va_acc,
                                "model_state": model.state_dict(),
                            },
                            self.checkpoint,
                        )
                else:
                    no_improve += 1

                if epoch % self.log_every == 0 or epoch == 1 or epoch == epochs:
                    print(
                        f"  {epoch:>5} | {tr_loss:>8.4f} | {tr_acc:>7.3%} | "
                        f"{va_loss:>8.4f} | {va_acc:>7.3%} | "
                        f"{timing['steps']:>4} steps | {timing['sec_per_step']:>6.3f} s/step | "
                        f"{timing['samples_per_sec']:>7.1f} samp/s"
                        + ("  *" if no_improve == 0 else "")
                    )

                if no_improve >= self.early_stop:
                    print(
                        f"\n  Early stop at epoch {epoch} (no improvement for {self.early_stop} epochs)"
                    )
                    break

            ckpt = torch.load(fold_ckpt, map_location=self.device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            report, cm, auc = self.eval_metrics(
                model=model,
                loader=test_loader,
                device=self.device,
                dtype=self.dtype,
            )

            print(f"\n  Best val acc : {best_val_acc:.3%}  (epoch {best_epoch})")
            print(f"  ROC-AUC (OvR macro) : {auc:.4f}")
            print(f"\n  Classification report:\n")
            for line in report.splitlines():
                print(f"    {line}")

            print(f"\n  Label map:")
            for idx, name in enumerate(self.dataset.label_names):
                print(f"    {idx:>3} = {name}")

            print(f"\n  Confusion matrix (rows=true, cols=pred, ids from label map):")
            header = "    " + " ".join(f"{idx:>4}" for idx in range(len(self.dataset.label_names)))
            print(header)
            for idx, row in enumerate(cm):
                print(f"    {idx:>3} " + " ".join(f"{v:>4}" for v in row))

            fold_results.append((i, best_val_acc, auc))
            best_epochs.append(best_epoch)

        accs = [acc for _, acc, _ in fold_results]
        aucs = [a for _, _, a in fold_results if not np.isnan(a)]

        print(f"\n{'='*55}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*55}")
        for (fold, acc, auc), epoch in zip(fold_results, best_epochs):
            print(f"  fold {fold}: acc {acc:.3%} | auc {auc:.4f} | best epoch {epoch}")
        print(f"  {'-' * 50}")
        print(f"  mean acc : {np.mean(accs):.3%} ± {np.std(accs):.3%}")
        if aucs:
            print(f"  mean auc : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  global best acc : {global_best:.3%} (saved to {self.checkpoint})")

        return fold_results, best_epochs, global_best
