import torch
import torch.nn as nn
from timm import create_model

class UTAE_Swin(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
        out_conv=[32, 20],
        swin_type="swin_tiny_patch4_window7_224",  # Type of pretrained Swin Transformer
        str_conv_k=4,
        str_conv_s=2,
        str_conv_p=1,
        agg_mode="att_group",
        encoder_norm="group",
        n_head=16,
        d_model=256,
        d_k=4,
        encoder=False,
        return_maps=False,
        pad_value=0,
        padding_mode="reflect",
    ):
        """
        Modified U-TAE architecture using Swin Transformer for both spatial and temporal encoding.
        """
        super(UTAE_Swin, self).__init__()
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths
        self.decoder_widths = decoder_widths
        self.pad_value = pad_value
        self.encoder = encoder
        self.swin_type = swin_type

        if encoder:
            self.return_maps = True

        if decoder_widths is not None:
            assert len(encoder_widths) == len(decoder_widths)
            assert encoder_widths[-1] == decoder_widths[-1]
        else:
            decoder_widths = encoder_widths

        # Spatial Encoder (Swin Transformer)
        self.spatial_encoder = create_model(swin_type, pretrained=True, in_chans=input_dim)

        # Temporal Transformer: Reusing Swin Transformer for temporal encoding
        self.temporal_transformer = nn.Transformer(
            d_model=d_model, nhead=n_head, num_encoder_layers=6, dim_feedforward=2048
        )

        # Decoder (Same as before)
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=decoder_widths[i],
                d_out=decoder_widths[i - 1],
                d_skip=encoder_widths[i - 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm="batch",
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        self.out_conv = ConvBlock(nkernels=[decoder_widths[0]] + out_conv, padding_mode=padding_mode)

    def forward(self, input, batch_positions=None, return_att=False):
        pad_mask = (input == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)  # BxT pad mask

        # SPATIAL ENCODER (Swin Transformer for each timestamp in the sequence)
        b, t, c, h, w = input.shape
        input = input.view(b * t, c, h, w)  # Flatten temporal dimension for spatial encoder

        spatial_features = self.spatial_encoder(input)
        spatial_features = spatial_features.view(b, t, -1, h // 32, w // 32)  # Reshape back

        # TEMPORAL TRANSFORMER
        temporal_features = self.temporal_transformer(spatial_features.flatten(2), spatial_features.flatten(2))

        temporal_features = temporal_features.view(b, t, -1, h // 32, w // 32)  # Reshape back

        # DECODER (same U-Net style decoder)
        out = temporal_features[:, -1]  # Use the last time step
        if self.return_maps:
            maps = [out]
        for i in range(self.n_stages - 1):
            out = self.up_blocks[i](out, spatial_features[:, -(i + 2)])
            if self.return_maps:
                maps.append(out)

        out = self.out_conv(out)
        if return_att:
            return out, maps
        return out

