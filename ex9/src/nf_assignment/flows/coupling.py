"""Affine coupling flow primitives."""

from __future__ import annotations

import torch
from nf_assignment.flows.transforms import Transform
from torch import nn


class AffineCouplingTransform(Transform):
    """Shared affine coupling transform for toy and sequence flows."""

    def __init__(self, conditioner: nn.Module):
        """Store the neural network that predicts shift and log-scale parameters."""

        super().__init__()
        self.conditioner = conditioner

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply affine coupling in the data-to-latent direction.

        Args:
            x: Input tensor shaped ``[batch, channels]`` for toy data or
                ``[batch, channels, frames]`` for speech features. The channel
                axis is split into identity and transformed halves.
            mask: Optional speech mask shaped ``[batch, 1, frames]`` where one
                marks valid frames and zero marks padding.
            condition: Optional conditioner input shaped ``[batch, cond_channels,
                frames]`` on the same frame grid as ``x``.

        Returns:
            A pair ``(y, log_det)`` where ``y`` has the same axes as ``x`` and
            ``log_det`` is shaped ``[batch]``.
        """

        # 全体のチャンネル数を得てその半分を変換に用いる
        # size(1)とすることで
        channels = x.size(1)
        c = channels // 2

        # identity / transformed に分ける
        # 前半(0:c)が固定、後半(c:)が変換に用いられる
        x_id = x[:, :c]
        x_tr = x[:, c:]

        # conditioner
        # 固定要素から導く変換パラメータを計算する
        # 話者情報や特徴量などconditionerに入力する場合もある
        if condition is None:
            h = self.conditioner(x_id)
        else:
            h = self.conditioner(x_id, condition=condition, mask=mask)

        # 連結されている2つの要素をチャンネル方向に分割
        # shiftは平行移動の関数
        # log_scaleは拡大率の対数(>0)
        shift, log_scale = torch.chunk(h, 2, dim=1)

        # affine変換
        y_tr = x_tr * torch.exp(log_scale) + shift

        # mask[batch, 1, frames]の処理
        # マスクする場所はmaskを掛けることで0になる
        # 2番目の要素が1であるがこれはbroadcastにより引き伸ばされ各チャンネルに掛けられる
        if mask is not None:
            y_tr = y_tr * mask
            y_id = x_id * mask
            log_scale = log_scale * mask
        else:
            y_id = x_id

        y = torch.cat([y_id, y_tr], dim=1)

        # log determinant(ヤコビアン)
        # 下三角行列のlog_detを求めるにあたり対角成分の積の対数を求める
        # 対角成分の積の対数を取るが、右下部分 ∂x_tr/∂y_trはdiag(exp(log_scale))である
        # そのため対角成分(今回のヤコビアンの値でもある)はexp(∑ᵢ log_scaleᵢ)
        # log|det(J)| = log( exp(∑ᵢ log_scaleᵢ) )
        #             = ∑ᵢ log_scaleᵢ   ← log と exp が打ち消し合う
        #             = log_scale.sum()
        if x.dim() == 2:
            log_det = log_scale.sum(dim=1)
        else:
            log_det = log_scale.sum(dim=(1, 2))

        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply affine coupling in the latent-to-data direction.

        Args:
            y: Latent tensor shaped ``[batch, channels]`` for toy data or
                ``[batch, channels, frames]`` for speech features.
            mask: Optional speech mask shaped ``[batch, 1, frames]``.
            condition: Optional conditioner input shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(x, log_det)`` where ``x`` has the same axes as ``y`` and
            ``log_det`` is shaped ``[batch]``.
        """

        # おおよそForwardと同じ
        # 逆方向の操作が必要
        channels = y.size(1)
        c = channels // 2

        y_id = y[:, :c]
        y_tr = y[:, c:]

        if condition is None:
            h = self.conditioner(y_id)
        else:
            h = self.conditioner(y_id, condition=condition, mask=mask)

        shift, log_scale = torch.chunk(h, 2, dim=1)

        # inverse affine transform
        x_tr = (y_tr - shift) * torch.exp(-log_scale)

        if mask is not None:
            x_tr = x_tr * mask
            x_id = y_id * mask
            log_scale = log_scale * mask
        else:
            x_id = y_id

        x = torch.cat([x_id, x_tr], dim=1)

        # inverse log determinant
        if y.dim() == 2:
            log_det = -log_scale.sum(dim=1)
        else:
            log_det = -log_scale.sum(dim=(1, 2))

        return x, log_det


class AffineCouplingBlock(Transform):
    """Toy affine coupling layer using the shared coupling transform."""

    def __init__(self, param_map: nn.Module):
        """Wrap a toy conditioner that maps ``[batch, channels / 2]`` to affine params."""

        super().__init__()
        self.transform = AffineCouplingTransform(param_map)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform toy samples from data to latent axes.

        Args:
            x: Tensor shaped ``[batch, 2]`` where axis 0 is sample index and
                axis 1 contains the two toy coordinates.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped ``[batch, 2]`` and
            ``log_det`` shaped ``[batch]``.
        """

        return self.transform(x)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform toy latent samples back to data axes.

        Args:
            y: Tensor shaped ``[batch, 2]`` where axis 0 is sample index and
                axis 1 contains latent coordinates.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped ``[batch, 2]`` and
            ``log_det`` shaped ``[batch]``.
        """

        return self.transform.inverse(y)


class SequenceAffineCoupling(Transform):
    """Speech affine coupling layer using the shared coupling transform."""

    def __init__(self, channels: int, conditioner: nn.Module):
        """Create a channel-split speech coupling transform."""

        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be even for channel-split coupling.")
        self.channels = int(channels)
        self.conditioner = conditioner
        self.transform = AffineCouplingTransform(conditioner)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speech features from data to latent axes.

        Args:
            x: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Optional condition tensor shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(z, log_det)`` with ``z`` shaped like ``x`` and ``log_det``
            shaped ``[batch]``.
        """

        return self.transform(x, mask=mask, condition=condition)

    def inverse(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform speech latent features back to data axes.

        Args:
            z: Tensor shaped ``[batch, coded_sp_channels, frames]``.
            mask: Optional mask shaped ``[batch, 1, frames]``.
            condition: Optional condition tensor shaped ``[batch, cond_channels,
                frames]``.

        Returns:
            A pair ``(x, log_det)`` with ``x`` shaped like ``z`` and ``log_det``
            shaped ``[batch]``.
        """

        return self.transform.inverse(z, mask=mask, condition=condition)
