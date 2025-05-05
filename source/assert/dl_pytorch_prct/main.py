import kagglehub
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from layers import Conv, C3k2, A2C2f, Classify
from utils import load, save, test_epoch, train_epoch
from tqdm import tqdm
import shutil


# ## build model
class MotionNet(nn.Module):
    def __init__(self, c1: int = 3, nc: int = 7):
        """
        Initialize classification model

        Args:
            c1 (int): Input channel size
            nc (int): Number of output classes
        """
        super().__init__()

        # Calculate scaling parameters from config

        # Build backbone
        self.backbone = nn.ModuleList(
            [
                # 0-P1/2
                Conv(c1=3, c2=16, k=3, s=2),
                # 1-P2/4
                Conv(c1=16, c2=32, k=3, s=2),
                # 2-C3k2 block
                C3k2(c1=32, c2=64, n=1, e=0.25),
                # 3-P3/8
                Conv(c1=64, c2=128, k=3, s=2),
                # 4-C3k2 block
                C3k2(c1=128, c2=128, n=1, e=0.25),
                # 5-P4/16
                Conv(c1=128, c2=128, k=3, s=2),
                # 6-A2C2f block
                A2C2f(c1=128, c2=128, n=2, a2=True, area=4, e=0.5),
                # 7-P5/32
                Conv(c1=128, c2=256, k=3, s=2),
                # 8-A2C2f block
                A2C2f(c1=256, c2=256, n=2, a2=True, area=1, e=0.5),
            ]
        )

        # Build classification head
        self.classify = Classify(256, nc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Pass through backbone
        for layer in self.backbone:
            x = layer(x)

        # Pass through classification head
        x = self.classify(x)

        return x


if __name__ == "__main__":
    # ## Hyper Parameters and Configs
    device = "cuda:1"
    lr = 1e-3
    step_size = 20
    epochs = 120
    start_epoch = -1
    train_patience = 20
    best_checkpoint = Path("best_model.pt")
    last_checkpoint = Path("last_model.pt")

    # ## prepare dataset
    # Download latest version
    data_id = "aadityasinghal/facial-expression-dataset"
    data_root = kagglehub.dataset_download(data_id)
    print("Path to dataset files:", data_root)
    data_root = Path(data_root)

    # ## build data transforms
    train_transformer = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),  # 224x224
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    val_transformer = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # ## build dataset
    train_dataset = ImageFolder(
        root=data_root / "train" / "train",
        transform=train_transformer,
    )
    val_dataset = ImageFolder(
        root=data_root / "test" / "test",
        transform=val_transformer,
    )

    # ## build dataloader
    batch_size = 128
    num_workers = 8
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = MotionNet(
        nc=len(train_dataset.classes),
    ).to(device)

    # ## build loss, optimizer and lr_scheduler

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=0.1,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    if last_checkpoint.is_file():
        start_epoch = load(
            model=model,
            optimizer=optimizer,
            path=last_checkpoint,
        )
        print(
            f"continue from {last_checkpoint.__str__()}, "
            + f"start at epoch {start_epoch+1}"
        )

    # ## train model
    loop = tqdm(range(start_epoch + 1, epochs))
    best_acc = 0
    patience = train_patience
    for epoch in loop:
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        train_acc, train_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device=device,
        )
        loop.set_postfix(
            train_acc=train_acc,
            train_loss=train_loss,
        )
        val_acc, val_loss = test_epoch(
            model,
            val_dataloader,
            criterion,
            device=device,
        )
        loop.set_postfix(
            train_acc=train_acc,
            train_loss=train_loss,
            val_acc=val_acc,
            val_loss=val_loss,
        )

        save(model, optimizer, epoch, last_checkpoint)

        if val_acc > best_acc:
            shutil.copyfile(last_checkpoint, best_checkpoint)
            best_acc = val_acc
            patience = train_patience
        else:
            patience -= 1
        if patience == 0:
            break
