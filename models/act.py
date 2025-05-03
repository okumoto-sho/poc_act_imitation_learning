import torch
import torch.nn as nn

from typing import Dict
from dataclasses import dataclass
from models.loss import l2_mean_error, kl_divergence
from models.backbones import create_backbone


def calculate_sinusoidal_positional_encoding(
    seq_length: int, embed_dim: int, temperature: float = 10000.0
):
    position = torch.arange(seq_length).float()
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float()
        * -(torch.log(torch.tensor(temperature)) / embed_dim)
    )
    emb = torch.zeros(seq_length, embed_dim)
    emb[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
    emb[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
    return emb


def sample_detached(mu, logvar):
    eps = torch.randn_like(mu)
    std = torch.exp(0.5 * logvar)
    return (mu + eps * std).detach()


class ActEncoder(nn.Module):
    def __init__(
        self,
        action_chunk_size,
        action_dim,
        qpos_dim,
        emb_dim,
        n_enc_layers=4,
        n_heads=8,
        feedforward_dim=3200,
        z_dim=32,
    ):
        super(ActEncoder, self).__init__()
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.qpos_dim = qpos_dim
        self.qpos_emb = nn.Linear(qpos_dim, emb_dim)
        self.action_emb = nn.Linear(action_dim, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            emb_dim, n_heads, feedforward_dim, batch_first=True
        )
        self.z_dim = z_dim
        self.encoder = nn.TransformerEncoder(self.encoder_layer, n_enc_layers)
        self.emb_z = nn.Linear(emb_dim, z_dim * 2)
        self.pos_encoding = (
            calculate_sinusoidal_positional_encoding(action_chunk_size + 2, emb_dim)
            .cuda()
            .detach()
        )

    def forward(self, qpos, actions):
        """
        Args:
        qpos: [batch_size, qpos_dim]
        action [batch_size, action_sequence_size, qpos_dim]
        Returns:
        mu_z: [batch_size, z_dim]
        logvar_z: [batch_size, z_dim]
        """
        actions_chunk = actions[:, : self.action_chunk_size, :]

        qpos_embeded = self.qpos_emb(qpos)
        qpos_embeded = qpos_embeded.unsqueeze(1)
        action_embeded = self.action_emb(actions_chunk)

        cls_token = self.cls_token.repeat(qpos_embeded.size(0), 1, 1)
        concatenated_embeded = torch.cat(
            (cls_token, qpos_embeded, action_embeded), dim=1
        )
        enc_output = self.encoder(concatenated_embeded + self.pos_encoding)
        z = self.emb_z(enc_output[:, 0, :])

        mu_z = z[:, : self.z_dim]
        logvar_z = z[:, self.z_dim :]
        return mu_z, logvar_z


class ActDecoder(nn.Module):
    # Maximum number of image features
    MAX_IMAGE_FEAT_SEQ_LENGTH = 3000

    def __init__(
        self,
        action_chunk_size,
        qpos_dim,
        action_dim,
        emb_dim,
        z_dim=32,
        n_dec_layers=7,
        n_enc_layers=4,
        n_heads=8,
        feedforward_dim=3200,
    ):
        super(ActDecoder, self).__init__()
        self.action_chunk_size = action_chunk_size
        self.qpos_dim = qpos_dim
        self.action_dim = action_dim
        self.emb_dim = emb_dim

        self.qpos_emb = nn.Linear(qpos_dim, emb_dim)
        self.z_emb = nn.Linear(z_dim, emb_dim)
        self.transformer = nn.Transformer(
            emb_dim,
            n_heads,
            n_enc_layers,
            n_dec_layers,
            feedforward_dim,
            batch_first=True,
        )

        self.action_token = nn.Parameter(torch.randn(1, action_chunk_size, emb_dim))
        self.action_head = nn.Linear(emb_dim, action_dim)

        self.pos_encoding = (
            calculate_sinusoidal_positional_encoding(
                self.MAX_IMAGE_FEAT_SEQ_LENGTH + 2, emb_dim
            )
            .cuda()
            .detach()
        )

    def forward(self, image_embeded, qpos, z):
        z_embeded = self.z_emb(z)
        z_embeded = z_embeded.unsqueeze(1)
        qpos_embeded = self.qpos_emb(qpos)
        qpos_embeded = qpos_embeded.unsqueeze(1)

        concated_embeded = torch.cat((z_embeded, qpos_embeded, image_embeded), dim=1)
        action_tokens = self.action_token.repeat(concated_embeded.size(0), 1, 1)

        output = self.transformer(
            concated_embeded
            + self.pos_encoding[: concated_embeded.shape[1], :].to(
                qpos.device, qpos.dtype
            ),
            action_tokens,
        )
        output = self.action_head(output[:, :, :])
        return output


@dataclass
class ActOutput:
    action_pred: torch.Tensor
    mu_z: torch.Tensor
    logvar_z: torch.Tensor
    loss_l2: torch.Tensor
    loss_kl: torch.Tensor
    loss_total: torch.Tensor


class ActPolicy(nn.Module):

    def __init__(
        self,
        camera_names,
        action_chunk_size,
        action_dim,
        qpos_dim,
        emb_dim,
        z_dim=32,
        n_enc_layers=4,
        n_dec_layers=7,
        n_heads=8,
        feedforward_dim=3200,
        backbone_model_name="resnet18",
        kl_loss_weight=10.0,
    ):
        super(ActPolicy, self).__init__()
        self.z_dim = z_dim
        self.backbones = nn.ModuleDict(
            {
                camera_name: create_backbone(
                    model_name=backbone_model_name, emb_dim=emb_dim
                )
                for camera_name in camera_names
            }
        )
        self.encoder = ActEncoder(
            action_chunk_size,
            action_dim,
            qpos_dim,
            emb_dim,
            n_enc_layers,
            n_heads,
            feedforward_dim,
            z_dim,
        )
        self.decoder = ActDecoder(
            action_chunk_size,
            qpos_dim,
            action_dim,
            emb_dim,
            z_dim,
            n_dec_layers,
            n_enc_layers,
            n_heads,
            feedforward_dim,
        )

        self.kl_loss_weight = kl_loss_weight

    def forward(
        self, qpos: torch.Tensor, actions: torch.Tensor, images: Dict[str, torch.Tensor]
    ):
        r"""
        This function takes qpos, actions, images as input and return the action predictions and vae outputs,
        mu_z and logvar_z.

        Args:
            qpos: [batch_size, qpos_dim]
            actions: [batch_size, action_horizon, action_dim]
            images (dict[str, torch.Tensor]): images from different camera sources.
            Example: {"camera_name_1": Tensor[batch_size, h, w, c], "camera_name_2": Tensor[batch_size, h, w, c]}
        Returns:
            ActOutput: A dataclass containing the output of the model.
        """

        # 　Encode the qpos and actions into latent z
        mu_z, logvar_z = self.encoder(qpos, actions)

        # Sample z from the latent distribution using reparameterization trick
        z = sample_detached(mu_z, logvar_z)

        # Encode the images into a single feature vector
        images_embeded = []
        for camera_name in images:
            images_embeded.append(self.backbones[camera_name](images[camera_name]))
        images_embeded = torch.cat(images_embeded, dim=1)

        # Decode the latent z and qpos into action predictions
        action_pred = self.decoder(images_embeded, qpos, z)

        # Calculate the loss
        loss_l2 = l2_mean_error(action_pred, actions)
        loss_kl = kl_divergence(mu_z, logvar_z)
        loss_total = loss_l2 + self.kl_loss_weight * loss_kl

        return ActOutput(
            mu_z=mu_z,
            logvar_z=logvar_z,
            action_pred=action_pred,
            loss_l2=loss_l2,
            loss_kl=loss_kl,
            loss_total=loss_total,
        )

    def inference(self, qpos, images):
        """
        qpos: [batch_size, qpos_dim]
        images: Dict[str, torch.Tensor[batch_size, h, w, n_channels]]
        """

        # Fix the latent variable z to be zero when doing inference
        z = torch.zeros((qpos.shape[0], self.z_dim)).to(
            device=qpos.device, dtype=qpos.dtype
        )

        # Encode the images into a single feature vector
        images_embeded = []
        for camera_name in images:
            images_embeded.append(self.backbones[camera_name](images[camera_name]))
        images_embeded = torch.cat(images_embeded, dim=1)

        # Decode the latent z and qpos into action predictions
        action_pred = self.decoder(images_embeded, qpos, z)
        return action_pred
