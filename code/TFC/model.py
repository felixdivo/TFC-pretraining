from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TFC(nn.Module):
    def __init__(self, configs):
        super().__init__()

        encoder_layers_t = TransformerEncoderLayer(
            configs.TSlength_aligned,
            dim_feedforward=2 * configs.TSlength_aligned,
            nhead=2,
            batch_first=True,
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        encoder_layers_f = TransformerEncoderLayer(
            configs.TSlength_aligned,
            dim_feedforward=2 * configs.TSlength_aligned,
            nhead=2,
            batch_first=True,
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x_in_t, x_in_f):
        # TODO: This implementation is porbably still iucorrect, since the the input is regarded as a single token,
        # in which case indeed, no positional embedding is needed but it also does not make sense to use a transformer
        # in the first place. The input should be regarded as a sequence of tokens, in which case the positional embedding
        # is needed. This is not implemented yet.
        # It should not be used.

        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class SingleConvBackbone(nn.Sequential):
    def __init__(self, configs):
        super().__init__(
            nn.Sequential(
                nn.Conv1d(
                    configs.input_channels,
                    32,
                    kernel_size=configs.kernel_size,
                    stride=configs.stride,
                    bias=False,
                    padding=(configs.kernel_size // 2),
                ),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.Dropout(configs.dropout),
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            ),
            nn.Sequential(
                nn.Conv1d(
                    64,
                    configs.final_out_channels,
                    kernel_size=8,
                    stride=1,
                    bias=False,
                    padding=4,
                ),
                nn.BatchNorm1d(configs.final_out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            ),
        )


class SingleProjector(nn.Sequential):
    def __init__(self, configs):
        super().__init__(
            nn.Linear(configs.CNNoutput_channel * configs.final_out_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )


class TFC_Original(nn.Module):  # Frequency domain encoder
    def __init__(self, configs):
        super().__init__()

        self.conv_backbone_time = SingleConvBackbone(configs)
        self.conv_backbone_freq = SingleConvBackbone(configs)
        self.projector_time = SingleProjector(configs)
        self.projector_freq = SingleProjector(configs)

    def forward(self, x_in_t, x_in_f):
        """Time-based Contrastive Encoder"""
        x = self.conv_backbone_time(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_time(h_time)

        """Frequency-based contrastive encoder"""
        f = self.conv_backbone_freq(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_freq(h_freq)

        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module):
    """Downstream classifier only used in finetuning"""

    def __init__(self, configs):
        super().__init__()
        self.logits = nn.Linear(2 * 128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
