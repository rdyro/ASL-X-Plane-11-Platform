from glob import glob
from PIL import Image
import os

import torch
from ray import tune
from ray.train import Checkpoint
from ray.air import session
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models.resnet import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)
from torch.utils.data import DataLoader

from .random_access_dataloader import RandomAccessXPlaneVideoDataset
from .utils import process_videos
from . import transform_eval, transform_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_collate_fn(batch, train: bool = True):
    frames, states = tuple(map(list, zip(*batch)))
    if train:
        frames = [transform_train(frame) for frame in frames]
    else:
        frames = [transform_eval(frame) for frame in frames]
    return torch.stack(frames), torch.tensor(states).to(torch.float32)


####################################################################################################


def train(config):
    files = list(glob(str(Path("~/datasets/*avi").expanduser())))
    process_videos(files)
    files = [Path(f).with_suffix(".mp4") for f in files]
    ds = RandomAccessXPlaneVideoDataset(
        files,
        transform=None,
        skip_start_frames=120,
        skip_end_frames=60,
        frame_skip_n=10,
    )
    dl = DataLoader(
        ds,
        batch_size=config["batch_size"],
        num_workers=16,
        shuffle=True,
        collate_fn=lambda x: x,  # need to collate in the master process using CUDA
    )
    model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.weight.shape[-1], 3)
    model.conv1 = nn.Sequential(nn.BatchNorm2d(3), model.conv1)
    model.to(device)
    print("#" * 40)
    print(f"CUDA is available: {torch.cuda.is_available()}")
    print("#" * 40)

    if config["opt"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["opt"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, 1, config["lr_step"])
    loss_fn = nn.MSELoss()
    norm_factor = torch.tensor([1e3, 1e3, 1e2], device=device)
    writer = SummaryWriter()

    # restore checkpoint
    start_epoch = 0
    if session.get_checkpoint() is not None:
        loaded_checkpoint = session.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_path:
            loaded_checkpoint_path = Path(loaded_checkpoint_path)
            optimizer.load_state_dict(torch.load(loaded_checkpoint_path / "optimizer.pt"))
            model.load_state_dict(torch.load(loaded_checkpoint_path / "model.pt"))
            start_epoch = int(torch.load(loaded_checkpoint_path / "epoch.pt"))
        print("#" * 40)
        print("Checkpoint found")
        print("#" * 40)
    else:
        print("#" * 40)
        print("Checkpoint NOT found")
        print("#" * 40)

    for epoch in range(start_epoch, 12):
        pbar = tqdm(dl, total=len(dl))
        for i, batch in enumerate(pbar):
            X, Y = custom_collate_fn(batch)
            if i == 0:
                for j, img in enumerate(X):
                    img = Image.fromarray(img.cpu().numpy().swapaxes(-1, -3))
                    path = Path(f"test_{j:02d}.png")
                    if path.exists():
                        os.remove(path)
                    img.save(path)
            optimizer.zero_grad()
            X, Y = X.to(device).to(torch.float32), Y.to(device)
            Y = Y[:, :3] / norm_factor
            loss = loss_fn(model(X), Y)
            loss.backward()
            pbar.set_description(f"Loss: {loss.item():.4e}")
            optimizer.step()
            writer.add_scalar("Loss", loss.item(), i * config["batch_size"] + epoch * len(ds))
        step_lr.step()
        torch.save(model.state_dict(), f"model_epoch{epoch:03d}.pt")
        checkpoint_dir = Path("checkpoint_dir")
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(epoch, checkpoint_dir / "epoch.pt")
        session.report(
            metrics=dict(epoch=epoch), checkpoint=Checkpoint.from_directory(str(checkpoint_dir))
        )

        # eval loss
        model.eval()
        with torch.no_grad():
            eval_loss, k = 0, 0
            for i, batch in enumerate(dl):
                X, Y = custom_collate_fn(batch)
                k += 1
                X, Y = X.to(device).to(torch.float32), Y.to(device)
                Y = Y[:, :3] / norm_factor
                eval_loss += loss_fn(model(X), Y)
                if i >= 100:
                    break
            eval_loss = eval_loss / k
        model.train()
        session.report(metrics=dict(objective=float(eval_loss.item())))

    torch.save(model.state_dict(), "model_final.pt")
    session.report(metrics=dict(objective=float(eval_loss.item())))
    return dict(objective=float(eval_loss.item()))


if __name__ == "__main__":
    best_config = {
        "dataset": "random_access",
        "batch_size": tune.choice([32]),
        "lr": tune.choice([3e-4]),
        "lr_step": tune.choice([1.0]),
        "opt": tune.choice(["Adam"]),
    }

    experiment = tune.Experiment(
        "training_xplane_new6",
        train,
        config=best_config,
        resources_per_trial={"gpu": 1, "cpu": 16},
        num_samples=50,
    )
    tune.run_experiments(experiment, resume="AUTO")
