import torch
import torch.nn as nn
import torchvision
import einops


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


def kl_divergence(mu_z: torch.Tensor, logvar_z: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)
    return kl.mean()


def l2_error(action_pred, action):
    diff = action - action_pred
    return torch.mean(diff**2, dim=(0, 1, 2))


class ActBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet18",
        output_layer_name="layer4",
        weights="DEFAULT",
    ):
        super(ActBackbone, self).__init__()
        self.model = getattr(torchvision.models, model_name)(weights=weights)
        self.modules = []
        for name, module in self.model.named_children():
            self.modules.append(module)
            if name == output_layer_name:
                break

        if self.modules is None:
            raise ValueError(
                f"output_layer_name {output_layer_name} not found in {model_name}"
            )

    def forward(self, x):
        x = einops.rearrange(x, "b k h w c -> (b k) c h w")
        for module in self.modules:
            x = module(x)
        x = einops.rearrange(x, "b c h w -> b (h w) c")
        return x


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
        qpos: [batch_size, qpos_dim]
        action [batch_size, action_sequence_size, qpos_dim]
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
        """
        image_embeded: [batch_size, n_channels, emb_dim]
        qpos: [batch_size, qpos_dim]
        z: [batch_size, z_dim]
        """
        z_embeded = self.z_emb(z)
        z_embeded = z_embeded.unsqueeze(1)
        qpos_embeded = self.qpos_emb(qpos)
        qpos_embeded = qpos_embeded.unsqueeze(1)

        concated_embeded = torch.cat((z_embeded, qpos_embeded, image_embeded), dim=1)
        action_tokens = self.action_token.repeat(concated_embeded.size(0), 1, 1)

        output = self.transformer(
            concated_embeded + self.pos_encoding[: concated_embeded.shape[1], :],
            action_tokens,
        )
        output = self.action_head(output[:, :, :])
        return output


class ActPolicy(nn.Module):

    def __init__(
        self,
        action_chunk_size,
        action_dim,
        qpos_dim,
        emb_dim,
        z_dim=32,
        n_enc_layers=4,
        n_dec_layers=7,
        n_heads=8,
        feedforward_dim=3200,
    ):
        super(ActPolicy, self).__init__()
        self.z_dim = z_dim
        self.backbone = ActBackbone()
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

    def forward(self, qpos, actions, images):
        mu_z, logvar_z = self.encoder(qpos, actions)
        z = sample_detached(mu_z, logvar_z)
        images_embeded = self.backbone(images)
        action_pred = self.decoder(images_embeded, qpos, z)
        return action_pred, mu_z, logvar_z

    def inference(self, qpos, images):
        """
        qpos: [batch_size, qpos_dim]
        images: [batch_size, 1, n_channels, h, w]
        """
        z = torch.zeros((qpos.shape[0], self.z_dim)).to(qpos.device)
        images_embeded = self.backbone(images)
        action_pred = self.decoder(images_embeded, qpos, z)
        return action_pred
