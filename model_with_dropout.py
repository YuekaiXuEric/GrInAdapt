import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from tool import pyutils
import torch.sparse as sparse
import numpy as np


def get_indices_of_pairs(radius, size):
    """
    Compute pairs of indices (flattened) for pixels in a grid of shape `size`
    that are within a specified Euclidean distance (`radius`).

    Args:
      radius (float or int): Maximum distance to consider.
      size (tuple): (height, width) of the grid.

    Returns:
      ind_from (np.ndarray): 1D array of indices for the "from" pixels.
      ind_to (np.ndarray): 1D array of indices for the "to" pixels.
    """
    H, W = size
    ind_from = []
    ind_to = []
    for r in range(H):
        for c in range(W):
            current_index = r * W + c
            # Determine neighborhood bounds
            r_min = max(0, r - int(radius))
            r_max = min(H - 1, r + int(radius))
            c_min = max(0, c - int(radius))
            c_max = min(W - 1, c + int(radius))
            for nr in range(r_min, r_max + 1):
                for nc in range(c_min, c_max + 1):
                    # Skip the pixel itself
                    if nr == r and nc == c:
                        continue
                    # Check the Euclidean distance
                    if math.sqrt((nr - r) ** 2 + (nc - c) ** 2) <= radius:
                        neighbor_index = nr * W + nc
                        ind_from.append(current_index)
                        ind_to.append(neighbor_index)
    return np.array(ind_from), np.array(ind_to)


# IPN
class IPN(nn.Module):
    def __init__(self, in_channels, n_classes, channels=128):
        super(IPN, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes
        self.input = Double3DConv(in_channels, channels)
        self.PLM1 = PLM(4, channels)
        self.PLM2 = PLM(4, channels)
        self.PLM3 = PLM(4, channels)
        self.PLM4 = PLM(2, channels)
        self.output = Double3DConv(channels, n_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.PLM1(x)
        x = self.PLM2(x)
        x = self.PLM3(x)
        x = self.PLM4(x)
        logits = self.output(x)
        return logits


class PLM(nn.Module):
    def __init__(self, poolingsize, channels):
        super().__init__()
        self.unipool = nn.MaxPool3d(kernel_size=[poolingsize, 1, 1])
        self.conv = Double3DConv(channels, channels)

    def forward(self, x):
        x = self.unipool(x)
        x = self.conv(x)
        return x


class Double3DConv(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_3dconv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout3d(0.2),
                                            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout3d(0.2))

    def forward(self, x):
        return self.double_3dconv(x)


# IPN-V2
class IPNV2(nn.Module):
    def __init__(self, in_channels, n_classes, ava_classes=2, return_feature=False, dc_norms="NN"):
        super(IPNV2, self).__init__()

        self.return_feature = return_feature
        # Suppose input shape is (N, 2, 128, 100, 100) or (B, C, H, W, D)
        self.FPM1 = FPM(in_channels, 16, h=16)  # H: 128->8, C: 2->16
        self.FPM2 = FPM(16, 32, h=4)  # H: 8->2, C: 16->32
        self.FPM3 = FPM(32, 64, h=2)  # H: 2->1, C: 32->64

        # (N, 64, 100, 100) -> (N, 5, 100, 100)

        # if ava_classes is not None:
        self.SegNet2D = UNetAva(64, 128, n_classes, ava_classes, return_feature=return_feature, dc_norms=dc_norms)
        # else:
        #     raise NotImplementedError("Only supports UNetAva for now")
        #     self.SegNet2D = UNet(64, 128, n_classes, return_feature=return_feature)  # H is squeezed, C -> n_classes: 64 -> 5

    def forward(self, x):

        x = self.FPM1(x)
        x = self.FPM2(x)
        x = self.FPM3(x)

        x = torch.squeeze(x, 2)

        return self.apply_head(x)

    def apply_head(self, x):
        # if isinstance(self.SegNet2D, UNetAva):
        if self.return_feature:
            cavf, ava, feature = self.SegNet2D(x)
        else:
            cavf, ava = self.SegNet2D(x)
        # else:
        #     if self.return_feature:
        #         cavf, feature = self.SegNet2D(x)
        #     else:
        #         cavf = self.SegNet2D(x)
        #     ava = None

        # logits_cavf = torch.unsqueeze(cavf, 2)

        # logits_ava = torch.unsqueeze(ava, 2) if ava is not None else None
        logits_cavf = cavf
        logits_ava = ava

        if self.return_feature:
            return logits_cavf, logits_ava, feature
        else:
            return logits_cavf, logits_ava

class FPM(nn.Module):
    def __init__(self, in_channels, out_channels, h):
        """
        in_channels: number of input channels (for the raw input should be 2, for OCTA(1) + OCT(1))
        out_channels: number of output channels
        h: the downsampling factor in the height dimension (3rd dimension)
        """
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(h, 1, 1), padding=(0, 0, 0), stride=(h, 1, 1)),
            nn.ReLU(inplace=True),
        )  # (B, C, H, W, D)

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(h, 3, 3), padding=(0, 1, 1), stride=(h, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(h, 3, 3), padding=(0, 2, 2), stride=(h, 1, 1), dilation=(1, 2, 2)),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool3d(kernel_size=[h, 1, 1])

        self.conv4 = nn.Sequential(
            nn.Conv3d(out_channels * 3 + in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x1 = self.conv1(x)  # (N, 2, 128, 100, 100) -> (N, 16, 8, 100, 100)

        x2 = self.conv2(x)  # (N, 2, 128, 100, 100) -> (N, 16, 8, 100, 100)

        x3 = self.conv3(x)  # (N, 2, 128, 100, 100) -> (N, 16, 8, 100, 100)

        x4 = self.maxpool(x)  # (N, 2, 128, 100, 100) -> (N, 2, 8, 100, 100)

        x = torch.cat([x1, x2], dim=1)  # (N, 16, 8, 100, 100) -> (N, 32, 8, 100, 100)
        x = torch.cat([x, x3], dim=1)  # (N, 32, 8, 100, 100) -> (N, 48, 8, 100, 100)
        x = torch.cat([x, x4], dim=1)  # (N, 48, 8, 100, 100) -> (N, 50, 8, 100, 100)

        x = self.conv4(x)  # (N, 50, 8, 100, 100) -> (N, 16, 8, 100, 100)

        return x

class IPNV2_with_proj_map(IPNV2):
    def __init__(self, in_channels, n_classes, proj_map_in_channels, ava_classes=2,
                 get_2D_pred=False, proj_vol_ratio=1, return_feature=True, dc_norms="NN", feature_dim = 128):
        super(IPNV2_with_proj_map, self).__init__(in_channels, n_classes, ava_classes=ava_classes, return_feature=return_feature, dc_norms=dc_norms)

        if proj_vol_ratio != 1:
            assert proj_vol_ratio == 2, "Only supports 1 or 2"

            self.pm_downsize_conv = nn.Conv2d(proj_map_in_channels, 64, kernel_size=3, padding=1, stride=2)
            self.proj_map_unet = UNet(64, 128, 64, return_feature=return_feature, dc_norms=dc_norms)
        else:
            self.pm_downsize_conv = None
            self.proj_map_unet = UNet(proj_map_in_channels, 128, 64, return_feature=return_feature, dc_norms=dc_norms)


        self.conv = DoubleConv2D(128, 64, norms=dc_norms)

        self.get_2D_pred = get_2D_pred
        if get_2D_pred:
            # if ava_classes is not None:
            self.head2D = UNetAva(64, 128, n_classes, ava_classes, dc_norms=dc_norms)
            #     self.head2D = UNet(64, 128, n_classes)  # H is squeezed, C -> n_classes: 64 -> 5

        self.manufacturer_fc = nn.Linear(feature_dim, 3)  # 3 classes for manufacturer
        self.anatomical_fc = nn.Linear(feature_dim, 2)
        self.region_size_fc = nn.Linear(feature_dim, 2)
        self.laterality_fc = nn.Linear(feature_dim, 2)

        BatchNorm = torch.nn.BatchNorm2d

        self.aff_faz = nn.Conv2d(128, 100, kernel_size=1, bias=False)
        torch.nn.init.xavier_uniform_(self.aff_faz.weight, gain=4)
        self.bn_faz = BatchNorm(100)
        self.bn_faz.weight.data.fill_(1)
        self.bn_faz.bias.data.zero_()

        self.from_scratch_layers = [self.aff_faz, self.bn_faz]

        self.predefined_featuresize = 100  # for example, if image_res=512, 512/16=32
        self.ind_from, self.ind_to = get_indices_of_pairs(radius=4, size=(self.predefined_featuresize, self.predefined_featuresize))
        self.ind_from = torch.from_numpy(self.ind_from)
        self.ind_to = torch.from_numpy(self.ind_to)




    def apply_2D_head(self, x):
        # if isinstance(self.head2D, UNetAva):
        out = self.head2D(x)
        if self.return_feature:
            logits_cavf, logits_ava, features = out
            return logits_cavf, logits_ava, features
        else:
            logits_cavf, logits_ava = out
            return logits_cavf, logits_ava
        # else:
        #     cavf = self.head2D(x)
        #     ava = None

        # logits_cavf = torch.unsqueeze(cavf, 2)
        # logits_ava = torch.unsqueeze(ava, 2) if ava is not None else None

        return logits_cavf, logits_ava

    def forward(self, x, proj_map, to_dense=False):
        x = self.FPM1(x)
        x = self.FPM2(x)
        x = self.FPM3(x)
        # (B, 64, 1, H, W)

        x = torch.squeeze(x, 2)

        if self.pm_downsize_conv is not None:
            # print("downsizing proj_map:", proj_map.shape)
            proj_map = self.pm_downsize_conv(proj_map)
            # print(proj_map.shape)

        proj_map_logits, proj_map_features = self.proj_map_unet(proj_map)

        x = torch.cat([x, proj_map_logits], dim=1)
        x = self.conv(x)
        if self.get_2D_pred:
            if self.return_feature:
                cavf3D_logits, ava3D_logits, features3D = self.apply_head(x)
                cavf2D_logits, ava2D_logits, features2D = self.apply_2D_head(proj_map_logits)

                pooled_features = torch.mean(features2D, dim=[2,3])
                manufacturer_logits = self.manufacturer_fc(pooled_features)
                anatomical_logits = self.anatomical_fc(pooled_features)
                region_size_logits = self.region_size_fc(pooled_features)
                laterality_logits = self.laterality_fc(pooled_features)

                f_faz = F.relu(self.bn_faz(self.aff_faz(features2D)))

                # print(f_faz.shape)
                # exit()

                f_faz = F.interpolate(f_faz, size=(100,100), mode='bilinear', align_corners=True)

                # Compute pairwise affinities on f_faz.
                if f_faz.size(2) == self.predefined_featuresize and f_faz.size(3) == self.predefined_featuresize:
                    ind_from = self.ind_from
                    ind_to = self.ind_to
                else:
                    print('featuresize error')
                    sys.exit()
                f_faz = f_faz.view(f_faz.size(0), f_faz.size(1), -1)
                ff = torch.index_select(f_faz, dim=2, index=ind_from.cuda(non_blocking=True))
                ft = torch.index_select(f_faz, dim=2, index=ind_to.cuda(non_blocking=True))
                ff = torch.unsqueeze(ff, dim=2)
                ft = ft.view(ft.size(0), ft.size(1), -1, ff.size(3))
                aff_faz = torch.exp(-torch.mean(torch.abs(ft-ff), dim=1))

                # aff_faz = torch.nn.functional.interpolate(aff_faz, size=63250, mode='linear', align_corners=False)

                if to_dense:
                    aff_faz = aff_faz.view(-1).cpu()

                    ind_from_exp = torch.unsqueeze(ind_from, dim=0).expand(ft.size(2), -1).contiguous().view(-1)
                    indices = torch.stack([ind_from_exp, ind_to])
                    indices_tp = torch.stack([ind_to, ind_from_exp])

                    area = f_faz.size(2)
                    indices_id = torch.stack([torch.arange(0, area).long(), torch.arange(0, area).long()])

                    aff_faz = sparse.FloatTensor(torch.cat([indices, indices_id, indices_tp], dim=1),
                                            torch.cat([aff_faz, torch.ones([area]), aff_faz])).to_dense()

                return cavf3D_logits, cavf2D_logits, manufacturer_logits, anatomical_logits, region_size_logits, laterality_logits, features2D, aff_faz

            cavf3D_logits, ava3D_logits = self.apply_head(x)
            cavf2D_logits, ava2D_logits = self.apply_2D_head(proj_map_logits)
            return cavf3D_logits, ava3D_logits, cavf2D_logits, ava2D_logits
        else:
            if self.return_feature:
                logits_cavf, logits_ava, features = self.apply_head(x)
                pooled_features = torch.mean(features, dim=[2,3])
                manufacturer_logits = self.manufacturer_fc(pooled_features)
                anatomical_logits = self.anatomical_fc(pooled_features)
                region_size_logits = self.region_size_fc(pooled_features)
                laterality_logits = self.laterality_fc(pooled_features)
                return logits_cavf, logits_ava, manufacturer_logits, anatomical_logits, region_size_logits, laterality_logits, features
            else:
                return self.apply_head(x)


class UNet(nn.Module):
    def __init__(self, in_channels, channels, n_classes, return_feature=True, dc_norms="NN"):
        """
        in_channels: number of input channels
        channels: number of channels in the hidden layers
        n_classes: number of output classes
        """
        super(UNet, self).__init__()
        self.return_feature = return_feature

        # output dim change is just in_channels -> n_classes.
        # the H and W will be the same.

        self.in_channels = in_channels
        self.channels = channels
        self.n_classes = n_classes

        # convolution to increase channels while keeping h, w the same
        self.inc = DoubleConv2D(in_channels, channels, dc_norms)

        # each down layer is a convolutional one.
        # C -> C ; H -> floor(H/2) ; W -> floor(W/2)
        self.down1 = Down(channels, 2 * channels, dc_norms)
        self.down2 = Down(2 * channels, 2 * channels, dc_norms)
        self.down3 = Down(2 * channels, 4 * channels, dc_norms)
        self.down4 = Down(4 * channels, 4 * channels, dc_norms)


        # each up layer is either a upsampling bilinear layer or an ConvTranspose layer.
        # Channels kept the same, while H and W increase through same dims in down layers.
        self.up1 = Up( (4+4) * channels, 4 * channels, norms=dc_norms)
        self.up2 = Up( (4+2) * channels, 2 * channels, norms=dc_norms)
        self.up3 = Up( (2+2) * channels, 2 * channels, norms=dc_norms)
        self.up4 = Up( (2+1) * channels, channels, norms=dc_norms)

        # output has same dims as input except C -> n_classes.
        self.outc = nn.Conv2d(channels, n_classes, kernel_size=1)


    def get_feature(self, x):
        x1 = self.inc(x)  # (N, 64, 100, 100) -> (N, 128, 100, 100)

        x2 = self.down1(x1)  # (N, 128, 100, 100) -> (N, 128, 50, 50)
        x3 = self.down2(x2)  # (N, 128, 50, 50) -> (N, 128, 25, 25)
        x4 = self.down3(x3)  # (N, 128, 25, 25) -> (N, 128, 12, 12)
        x5 = self.down4(x4)  # (N, 128, 12, 12) -> (N, 128, 6, 6)

        x = self.up1(x5, x4)  # (N, 128, 6, 6) -> (N, 128, 12, 12)

        x = self.up2(x, x3)  # (N, 128, 12, 12) -> (N, 128, 25, 25)
        x = self.up3(x, x2)  # (N, 128, 25, 25) -> (N, 128, 50, 50)
        feature = self.up4(x, x1)  # (N, 128, 50, 50) -> (N, 128, 100, 100)

        return feature

    def forward(self, x):

        feature = self.get_feature(x)
        logits_cavf = self.outc(feature)  # (N, 128, 100, 100) -> (N, 5, 100, 100)

        if self.return_feature:
            return logits_cavf, feature
        else:
            return logits_cavf


class UNetAva(UNet):
    def __init__(self, in_channels, channels, n_classes, ava_classes, return_feature=True, dc_norms="NN"):
        super(UNetAva, self).__init__(in_channels, channels, n_classes, return_feature, dc_norms)
        self.ava_out = nn.Conv2d(channels, ava_classes, kernel_size=1)

    def forward(self, x):
        feature = self.get_feature(x)
        logits_cavf = self.outc(feature)
        logits_ava = self.ava_out(feature)

        if self.return_feature:
            return logits_cavf, logits_ava, feature
        else:
            return logits_cavf, logits_ava



class DoubleConv2D(nn.Module):
    """(convolution=> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, norms = "NN"):
        super().__init__()

        assert len(norms) == 2, "Double Conv should have exactly 2 norms"
        # in_channels -> out_channels. The H and W don't change.
        # self.double_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        #                                  nn.ReLU(inplace=True),
        #                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        #                                  nn.ReLU(inplace=True))

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        if norms[0] == "G":
            self.norm1 = nn.GroupNorm(16, out_channels)
        elif norms[0] == "L":
            self.norm1 = nn.LayerNorm(out_channels)
        elif norms[0] == "N":
            self.norm1 = nn.Identity()
        else:
            raise ValueError("Unsuppored normalization type")

        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.dropout1 = nn.Dropout2d(0.2)

        if norms[1] == "G":
            self.norm2 = nn.GroupNorm(16, out_channels)
        elif norms[1] == "L":
            self.norm2 = nn.LayerNorm(out_channels)
        elif norms[1] == "N":
            self.norm2 = nn.Identity()
        else:
            raise ValueError("Unsuppored normalization type")

        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(0.2)

    def forward(self, x):
        # First block
        x = self.conv1(x)


        # x = x.permute(0, 2, 3, 1)
        # x = self.ln1(x)
        # x = x.permute(0, 3, 1, 2)

        x = self.norm1(x)
        x = self.relu1(x)

        x = self.dropout1(x)

        # Second block
        x = self.conv2(x)

        x = self.norm2(x)
        # x = x.permute(0, 2, 3, 1)
        # x = self.ln2(x)
        # x = x.permute(0, 3, 1, 2)

        x = self.relu2(x)
        x = self.dropout2(x)

        return x
        # return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norms = "NN"):
        super().__init__()

        # H -> floor(H/2), W -> floor(W/2)
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(in_channels, out_channels, norms))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norms = "NN"):
        super().__init__()

        self.in_channels = in_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv2D(in_channels, out_channels, norms)

    def forward(self, x1, x2):

        assert x1.size()[1] + x2.size()[1] == self.in_channels, "Input channels mismatch"
        # x1 is input and x2 is for skip connection
        # for x1 = (N, C1, H1, W2) and x2 = (N, C2, H2, W2)...

        # doubles H1 and W1
        x1 = self.up(x1)  # H1 -> 2*H1 ; W1 -> 2*W1

        # calculates diff in height and width between x1 and x2 now.
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # pads x1 to make sure that H1 = H2 and W1 = W2.
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # skip connection along channels
        x = torch.cat([x2, x1], dim=1)  # C -> 2*C

        # conv to go back down to C channels.
        x = self.conv(x)

        return x