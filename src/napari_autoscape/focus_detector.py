
import lightning as pl
import segmentation_models_pytorch as smp
from torch import nn, optim
import torch.nn.functional as F
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FocusLightningModel(pl.LightningModule):
    def __init__(self):
        super(FocusLightningModel, self).__init__()

        # Load the pre-trained ResNet18 model
        unet = smp.Unet(
            "resnet18",
            encoder_weights="imagenet",
            in_channels=1,
            classes=1,
            activation=None,
        )

        # Use the convolutional part as the encoder
        self.encoder = unet.encoder

        # Your custom layers for the regression task
        self.fc1 = nn.Linear(392000, 256)
        self.fc2 = nn.Linear(256, 1)

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.encoder(x)  # use only results from deepest layer
        flattened_features = [f.view(f.size(0), -1) for f in x[1:3]]
        x = torch.cat(flattened_features, dim=1)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels.float())
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels.float())

        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append(
            {
                "val_loss": loss,
            }
        )
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.val_outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)

        # Optionally clear the outputs to free up memory
        del self.val_outputs

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels.float())
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class FocusCNN(nn.Module):
    def __init__(self):
        super(FocusCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 17 * 17, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FocusModel(pl.LightningModule):
    def __init__(self):
        super(FocusModel, self).__init__()
        self.model = FocusCNN()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels.float())
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels.float())

        if not hasattr(self, "val_outputs"):
            self.val_outputs = []
        self.val_outputs.append(
            {
                "val_loss": loss,
            }
        )
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.val_outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)

        # Optionally clear the outputs to free up memory
        del self.val_outputs

    def test_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self(images)
        loss = self.criterion(outputs.squeeze(), labels.float())
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer


def preprocess_stack(stack):
    augmentations = A.Compose(
        [
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Normalize(mean=0, std=1, max_pixel_value=255.0),
            A.Resize(height=140, width=140),
            ToTensorV2(),
        ]
    )

    processed_stack = []
    for slice in stack:
        # If grayscale, repeat the single channel three times to simulate RGB
        processed_slice = augmentations(image=slice)["image"]
        processed_stack.append(processed_slice)

    return torch.stack(processed_stack)


def get_focus_plane(model, image_stack):
    import numpy as np

    image_stack = preprocess_stack(image_stack).to("cuda")
    with torch.no_grad():
        offsets = model(image_stack)
        offsets = offsets.squeeze().detach().cpu().numpy()

    # retrieve focus plane
    return np.argmin(abs(offsets))


def detect_focus_plane(
    image: "napari.types.ImageData",
    segmentation: "napari.types.LabelsData",
    width: float,
    height: float,
    scale_xy: float,
    crop_margin: int = 0,
    focus_detector: torch.nn.Module = None,
) -> "napari.types.PointsData":
    from .utilities import crop_well_from_image, get_well_centroids
    import tqdm
    import numpy as np

    well_centroids = get_well_centroids(segmentation) / scale_xy
    focus_planes = []
    for idx in tqdm.tqdm(range(len(well_centroids)), desc="Detecting focus planes"):
        try:
            well = crop_well_from_image(image, well_centroids[idx], width, height)
            well = well[:, crop_margin:-crop_margin, crop_margin:-crop_margin]

            focus_plane = get_focus_plane(focus_detector, well)
            focus_planes.append(focus_plane)
        except:
            focus_planes.append(np.nan)

    focus_planes = np.asarray(focus_planes)
    focus_points = np.concatenate((focus_planes[:, None], well_centroids), axis=1)

    return focus_points
