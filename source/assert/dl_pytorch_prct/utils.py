import torch
from tqdm import tqdm


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    scaler=None,
    device="cuda",
    fp16=True,
):
    model.train()
    loop = tqdm(train_loader, desc="train", leave=False)
    total_loss = 0.0
    total_count = 0
    ateru = 0
    assert scaler is not None or not fp16

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(
            device_type=device,
            enabled=fp16,
            dtype=torch.float16 if fp16 else torch.float32,
        ):
            output = model(data)
            loss = criterion(output, target)
        if fp16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_count += len(data)
        # calc acc
        ateru += (output.argmax(dim=1) == target).sum().item()
        total_loss += loss.item()
        loop.set_postfix(
            {
                "acc": ateru / total_count,
                "loss": total_loss / (batch_idx + 1),
            }
        )
    scheduler.step()
    return ateru / total_count, total_loss / (batch_idx + 1)


@torch.no_grad()
def test_epoch(model, val_loader, criterion, device="cuda"):
    model.eval()
    loop = tqdm(val_loader, desc="val", leave=False)
    val_loss = 0.0
    total_num = 0
    total_correct = 0
    for i, (inputs, labels) in enumerate(loop):
        inputs = inputs.to(device)
        labels = labels.to(device)
        logist, _ = model(inputs)
        loss = criterion(logist, labels)
        val_loss += loss.item() * inputs.size(0)
        total_num += labels.size(0)
        total_correct += (logist.argmax(dim=1) == labels).sum().item()
        loop.set_postfix(loss=val_loss / (i + 1), acc=total_correct / total_num)
    val_loss = val_loss / len(val_loader.dataset)
    return total_correct / total_num, val_loss


def save(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return epoch
