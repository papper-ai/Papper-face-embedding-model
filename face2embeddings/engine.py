import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from .model import FaceSwin
from pathlib import Path
from typing import Dict, List, Tuple


def train_step(
    model: FaceSwin,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    epoch_num: int,
    scheduler: torch.optim.lr_scheduler = None,
    writer: torch.utils.tensorboard.writer.SummaryWriter | None = None,
    checkpoint_step: int = True,
    checkpoint_path: Path | None = None,
    save_limits: int | None = None,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (x1, x2, y) in enumerate(tqdm(dataloader, desc="Train Step")):
        x1: torch.Tensor = x1.to(device)
        x2: torch.Tensor = x2.to(device)
        y: torch.Tensor = y.to(device)

        y_pred = model.train_forward(x1=x1, x2=x2)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = y_pred.round()
        acc = (y_pred_class == y).sum().item() / len(y_pred)
        train_acc += acc

        if writer is not None and batch % 100 == 0:
            writer.add_scalar(
                tag="train_loss_step",
                scalar_value=train_loss / (batch + 1),
                global_step=(epoch_num + 1) * batch,
            )
            writer.add_scalar(
                tag="train_acc_step",
                scalar_value=train_acc / (batch + 1),
                global_step=(epoch_num + 1) * batch,
            )
            writer.close()

            if scheduler is not None:
                scheduler.step()

        if checkpoint_step and batch % checkpoint_step == 0:
            if checkpoint_path is not None:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(), checkpoint_path / f"checkpoint_{batch}.pt"
                )

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(
    model: FaceSwin,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
    writer: torch.utils.tensorboard.writer.SummaryWriter | None = None,
    epoch_num: int = 0,
) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (x1, x2, y) in enumerate(tqdm(dataloader, desc="Validation Step")):
            x1: torch.Tensor = x1.to(device)
            x2: torch.Tensor = x2.to(device)
            y: torch.Tensor = y.to(device)

            y_pred = model.train_forward(x1=x1, x2=x2)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            test_pred_labels = y_pred.round()
            acc = (test_pred_labels == y).sum().item() / len(test_pred_labels)
            test_acc += acc

            if writer is not None and batch % 100 == 0:
                writer.add_scalar(
                    tag="val_loss_step",
                    scalar_value=test_loss / (batch + 1),
                    global_step=(epoch_num + 1) * batch,
                )
                writer.add_scalar(
                    tag="val_acc_step",
                    scalar_value=test_acc / (batch + 1),
                    global_step=(epoch_num + 1) * batch,
                )
                writer.close()

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


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
    save_limits: int | None = None,
) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), desc="Epochs Loop"):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch,
            writer=writer,
            scheduler=scheduler,
            checkpoint_step=checkpoint_step,
            checkpoint_path=checkpoint_path,
            save_limits=save_limits,
        )
        val_loss, val_acc = val_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
            writer=writer,
            epoch_num=epoch,
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "val_loss": val_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "val_acc": val_acc},
                global_step=epoch,
            )
            writer.close()
        else:
            pass
        scheduler.step()
    return results
