import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .model import FaceSwin
from torch.utils import tensorboard
from typing import Dict, List, Tuple


def train_step(
    model: FaceSwin,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0

    for x1, x2, y in dataloader:
        x1: torch.Tensor = x1.to(device)
        x2: torch.Tensor = x2.to(device)
        y: torch.Tensor = y.to(device)

        y_pred = model(x1=x1, x2=x2)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = y_pred.round()
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(
    model: FaceSwin,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for x1, x2, y in dataloader:
            x1: torch.Tensor = x1.to(device)
            x2: torch.Tensor = x2.to(device)
            y: torch.Tensor = y.to(device)

            y_pred = model(x1=x1, x2=x2)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            test_pred_labels = y_pred.round()
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(
    model: FaceSwin,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device | str,
    writer: tensorboard,  # new parameter to take in a writer
) -> Dict[str, List]:
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer:
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )
            writer.close()
        else:
            pass
    return results
