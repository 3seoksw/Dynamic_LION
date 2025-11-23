from typing import Callable
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, log_path: str, metrics: dict[str, Callable] | None):
        self.log_path = log_path
        self.metrics = metrics
        metrics_str = ""
        if metrics is not None:
            for metric in metrics.keys():
                metrics_str += f",pos_{metric},neg_{metric}"

        log_dir = Path(log_path)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        if self.log_path is not None:
            with open(self.log_path, "w") as f:
                f.write(f"epoch,step,split,loss{metrics_str}\n")

    def log_step(self, history: dict, verbose: bool = False):
        if self.log_path is None:
            return

        epoch = history["epoch"]
        step = history["step"]
        split = history["split"]
        loss = history["loss"]
        if self.metrics is not None:
            pos_cos = history["pos_cosine_similarity"]
            neg_cos = history["neg_cosine_similarity"]
        else:
            pos_cos, neg_cos = 0, 0

        with open(self.log_path, "a") as f:
            if self.metrics is not None:
                f.write(f"{epoch},{step},{split},{loss},{pos_cos},{neg_cos}\n")
            else:
                f.write(f"{epoch},{step},{split},{loss}\n")
        if verbose:
            if self.metrics is not None:
                print(
                    f"Epoch {epoch}: {split} loss: {loss:.3f}, pos: {pos_cos:.3f}, neg: {neg_cos:.3f}"
                )
            else:
                print(f"Epoch {epoch}: {split} loss: {loss:.3f}")
