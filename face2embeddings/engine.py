import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from .model import FaceSwin
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List


def train_step(
    model: FaceSwin,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch_num: int,
    scheduler: torch.optim.lr_scheduler = None,
    writer: SummaryWriter | None = None,
    checkpoint_step: int = True,
    checkpoint_path: Path | None = None,
) -> float:
    model.train()

    train_loss = 0

    for batch_num, (anchor, pos_example, neg_example) in enumerate(
        tqdm(dataloader, desc="Train Step")
    ):
        anchor = anchor.to(device)
        pos_example = pos_example.to(device)
        neg_example = neg_example.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            anchor_embedding, pos_example_embedding, neg_example_embedding = (
                model.train_forward(
                    anchor=anchor, pos_example=pos_example, neg_example=neg_example
                )
            )
            loss = loss_fn(
                anchor_embedding, pos_example_embedding, neg_example_embedding
            )
        if not torch.isnan(loss):
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if writer is not None and batch_num % 100 == 0 and batch_num != 0:
            writer.add_scalar(
                tag="train_loss_epoch",
                scalar_value=train_loss / (batch_num + 1),
                global_step=(epoch_num * len(dataloader)) + (batch_num + 1),
            )
            writer.add_scalar(
                tag="train_loss_step",
                scalar_value=loss.item(),
                global_step=(epoch_num * len(dataloader)) + (batch_num + 1),
            )

        if scheduler is not None and batch_num % 100 == 0 and batch_num != 0:
            scheduler.step()
            if writer is not None:
                writer.add_scalar(
                    tag="learning_rate",
                    scalar_value=scheduler.get_last_lr()[0],
                    global_step=(epoch_num * len(dataloader)) + (batch_num + 1),
                )

        if checkpoint_step and batch_num % checkpoint_step == 0 and batch_num != 0:
            if checkpoint_path is not None:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(), checkpoint_path / f"checkpoint_{batch_num}.pt"
                )

    train_loss = train_loss / len(dataloader)
    return train_loss


def val_step(
    model: FaceSwin,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch_num: int,
    device: torch.device | str,
    writer: SummaryWriter | None = None,
) -> float:
    model.eval()

    test_loss = 0

    with torch.inference_mode():
        for batch_num, (anchor, pos_example, neg_example) in enumerate(
            tqdm(dataloader, desc="Validation Step")
        ):
            anchor = anchor.to(device)
            pos_example = pos_example.to(device)
            neg_example = neg_example.to(device)

            anchor_embedding, pos_example_embedding, neg_example_embedding = (
                model.train_forward(
                    anchor=anchor, pos_example=pos_example, neg_example=neg_example
                )
            )

            loss = loss_fn(
                anchor_embedding, pos_example_embedding, neg_example_embedding
            )
            test_loss += loss.item()

            if writer is not None and batch_num % 100 == 0:
                writer.add_scalar(
                    tag="val_loss_epoch",
                    scalar_value=test_loss / (batch_num + 1),
                    global_step=(epoch_num * len(dataloader)) + (batch_num + 1),
                )
    test_loss = test_loss / len(dataloader)
    return test_loss


def train(
    model: FaceSwin,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device | str,
    scheduler: torch.optim.lr_scheduler = None,
    writer: torch.utils.tensorboard.SummaryWriter | None = None,
    checkpoint_step: int = True,
    checkpoint_path: Path | None = None,
) -> Dict[str, List]:
    results = {"train_loss": [], "val_loss": []}

    for epoch_num in tqdm(range(epochs), desc="Epochs Loop"):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch_num,
            writer=writer,
            scheduler=scheduler,
            checkpoint_step=checkpoint_step,
            checkpoint_path=checkpoint_path,
        )
        val_loss = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            writer=writer,
            epoch_num=epoch_num,
        )

        print(
            f"Epoch: {epoch_num + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        if writer:
            writer.add_scalars(
                main_tag="Loss_per_epochs",
                tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
                global_step=epoch_num,
            )
            writer.close()
        else:
            pass
    return results
