import torch
import torch.nn as nn
import torch.nn.functional as F
from BoTNet import BottleStack


def yolo_params(version):
    if version == "n":
        return 1 / 3, 1 / 4, 2.0
    if version == "s":
        return 1 / 3, 1 / 2, 2.0
    if version == "m":
        return 2 / 3, 3 / 4, 1.5
    if version == "l":
        return 1.0, 1.0, 1.0
    if version == "x":
        return 1.0, 1.25, 1.0
    raise ValueError(f"Unsupported YOLO version: {version}")


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        activation=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x if self.shortcut else y


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList(
            [
                Bottleneck(self.mid_channels, self.mid_channels, shortcut=shortcut)
                for _ in range(num_bottlenecks)
            ]
        )
        self.conv2 = Conv(
            (num_bottlenecks + 2) * self.mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        outputs = [x1, x2]

        for block in self.m:
            x1 = block(x1)
            outputs.insert(0, x1)

        return self.conv2(torch.cat(outputs, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = Conv(4 * hidden_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            dilation=1,
            ceil_mode=False,
        )

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x).squeeze(-1).squeeze(-1))
        mx = self.mlp(self.max_pool(x).squeeze(-1).squeeze(-1))
        weight = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * weight


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        weight = self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * weight


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial."""

    def __init__(self, channels, ratio=8, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class Backbone(nn.Module):
    ATTN_MODES = ("none", "cbam", "botnet", "cbam_botnet")

    def __init__(self, version, in_channels=3, shortcut=True, attn="cbam_botnet", img_size=640):
        super().__init__()
        if attn not in self.ATTN_MODES:
            raise ValueError(f"attn must be one of {self.ATTN_MODES}, got {attn!r}")
        use_cbam = attn in ("cbam", "cbam_botnet")
        use_botnet = attn in ("botnet", "cbam_botnet")

        d, w, r = yolo_params(version)

        self.conv_0 = Conv(in_channels, int(64 * w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1)

        self.c2f_2 = C2f(int(128 * w), int(128 * w), num_bottlenecks=int(3 * d), shortcut=shortcut)
        self.c2f_4 = C2f(int(256 * w), int(256 * w), num_bottlenecks=int(6 * d), shortcut=shortcut)
        self.c2f_6 = C2f(int(512 * w), int(512 * w), num_bottlenecks=int(6 * d), shortcut=shortcut)
        # P5 block: BoTNet stack (MHSA) when enabled, else the standard C2f (c2f_8).
        # Both are shape-preserving (p5_channels -> p5_channels) so neck/head are unchanged.
        # BoTNet's rel_pos_emb bakes the P5 fmap size in and asserts if the input
        # size changes, so it must track img_size: P5 = img_size / 32 (e.g. 640->20,
        # 1280->40). img_size must therefore be a multiple of 32.
        p5_fmap_size = img_size // 32
        p5_channels = int(512 * w * r)
        if use_botnet:
            self.p5_block = BottleStack(
                dim=p5_channels,
                dim_out=p5_channels,
                fmap_size=p5_fmap_size,
                num_layers=3,
                heads=4,
                dim_head=p5_channels // (4 * 4),
                proj_factor=4,
                downsample=False,
                rel_pos_emb=True,
                activation=nn.SiLU(),
            )
        else:
            self.p5_block = C2f(p5_channels, p5_channels, num_bottlenecks=int(3 * d), shortcut=shortcut)

        # CBAM on the three feature maps that feed the neck (Identity when disabled,
        # keeping forward() branch-free). Channels match each stage's output.
        self.cbam_4 = CBAM(int(256 * w)) if use_cbam else nn.Identity()
        self.cbam_6 = CBAM(int(512 * w)) if use_cbam else nn.Identity()
        self.cbam_8 = CBAM(int(512 * w * r)) if use_cbam else nn.Identity()

        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.cbam_4(self.c2f_4(x))
        x = self.conv_5(out1)
        out2 = self.cbam_6(self.c2f_6(x))
        x = self.conv_7(out2)
        x = self.cbam_8(self.p5_block(x))
        out3 = self.sppf(x)
        return out1, out2, out3


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)

        self.up = Upsample()
        self.c2f_1 = C2f(
            in_channels=int(512 * w * (1 + r)),
            out_channels=int(512 * w),
            num_bottlenecks=int(3 * d),
            shortcut=False,
        )
        self.c2f_2 = C2f(
            in_channels=int(768 * w),
            out_channels=int(256 * w),
            num_bottlenecks=int(3 * d),
            shortcut=False,
        )
        self.c2f_3 = C2f(
            in_channels=int(768 * w),
            out_channels=int(512 * w),
            num_bottlenecks=int(3 * d),
            shortcut=False,
        )
        self.c2f_4 = C2f(
            in_channels=int(512 * w * (1 + r)),
            out_channels=int(512 * w * r),
            num_bottlenecks=int(3 * d),
            shortcut=False,
        )
        self.cv_1 = Conv(int(256 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(int(512 * w), int(512 * w), kernel_size=3, stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x):
        res_1 = x

        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)
        res_2 = self.c2f_1(x)

        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)
        out_1 = self.c2f_2(x)

        x = self.cv_1(out_1)
        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)

        x = self.cv_2(out_2)
        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1, out_2, out_3


class DFL(nn.Module):
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(ch, 1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = x

    def forward(self, x):
        b, _, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        return x.view(b, 4, a)


class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch * 4
        self.nc = num_classes
        self.no = self.coordinates + self.nc
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))

        _, w, r = yolo_params(version)
        cls_channels = max(int(256 * w), min(self.nc, 100))

        self.box = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(int(256 * w), self.coordinates, kernel_size=3, stride=1, padding=1),
                    Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w), self.coordinates, kernel_size=3, stride=1, padding=1),
                    Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w * r), self.coordinates, kernel_size=3, stride=1, padding=1),
                    Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1),
                ),
            ]
        )
        self.cls = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(int(256 * w), cls_channels, kernel_size=3, stride=1, padding=1),
                    Conv(cls_channels, cls_channels, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(cls_channels, self.nc, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w), cls_channels, kernel_size=3, stride=1, padding=1),
                    Conv(cls_channels, cls_channels, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(cls_channels, self.nc, kernel_size=1, stride=1),
                ),
                nn.Sequential(
                    Conv(int(512 * w * r), cls_channels, kernel_size=3, stride=1, padding=1),
                    Conv(cls_channels, cls_channels, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(cls_channels, self.nc, kernel_size=1, stride=1),
                ),
            ]
        )
        self.dfl = DFL(ch)

    def forward(self, x):
        for i in range(len(self.box)):
            box = self.box[i](x[i])
            cls = self.cls[i](x[i])
            x[i] = torch.cat((box, cls), dim=1)

        if self.training:
            return x

        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)

    def make_anchors(self, x, strides, offset=0.5):
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device

        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


class MyYolo(nn.Module):
    def __init__(self, version="s", num_classes=2, attn="cbam_botnet", img_size=640):
        super().__init__()
        self.backbone = Backbone(version=version, attn=attn, img_size=img_size)
        self.neck = Neck(version=version)
        self.head = Head(version=version, num_classes=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))


if __name__ == "__main__":
    # Self-check: every attn mode builds, forwards 640x640, and the flag actually
    # gates modules (param counts differ as expected).
    def _params(m):
        return sum(p.numel() for p in m.parameters())

    counts = {}
    for mode in Backbone.ATTN_MODES:
        m = MyYolo(version="s", num_classes=2, attn=mode).eval()
        with torch.no_grad():
            out = m(torch.zeros(1, 3, 640, 640))
        expected = (1, 4 + m.head.nc, 8400)  # (box(4) + classes) x anchors(80^2+40^2+20^2)
        assert out.shape == expected, f"{mode}: got {tuple(out.shape)}, expected {expected}"
        counts[mode] = _params(m)
        print(f"{mode:12s} params={counts[mode]:>12,} out={tuple(out.shape)}")

    assert counts["cbam"] > counts["none"], "CBAM should add parameters over baseline"
    assert counts["cbam_botnet"] > counts["botnet"], "CBAM should add parameters over BoTNet"
    assert counts["botnet"] != counts["none"], "BoTNet should change the P5 block"
    print("OK: all attn modes build and the toggle gates modules.")
