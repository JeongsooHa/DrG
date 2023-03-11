import torch
from torch import nn
from torch.nn import functional as F

def spatial_softmax(features):
    """Compute softmax over the spatial dimensions

    Compute the softmax over heights and width

    Args
    ----
    features: tensor of shape [N, C, H, W]
    """
    features_reshape = features.reshape(features.shape[:-2] + (-1,)) # 64, 12, 25, 25 ==> 64, 12, 625
    output = F.softmax(features_reshape, dim=-1) # 64, 12, 625
    output = output.reshape(features.shape) # 64, 12, 625 ==> 64, 12, 25, 25
    return output


class FeatureEncoder(nn.Module):
    """Phi"""
    def __init__(self, encoder):
        super(FeatureEncoder, self).__init__()
        self.net = encoder


    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.

        Returns
        =======
        y: (N, C, H, K) tensor.
        """
        return self.net(x, keypoint=True)


class PoseRegressor(nn.Module):
    """Pose regressor"""

    # https://papers.nips.cc/paper/7657-unsupervised-learning-of-object-landmarks-through-conditional-image-generation.pdf

    def __init__(self, encoder, k=1):
        super(PoseRegressor, self).__init__()
        self.net = encoder
        net_channels = 256
        self.regressor = nn.Conv2d(net_channels, k, kernel_size=(1, 1))

    def forward(self, x):
        """
        Args
        ====
        x: (N, C, H, W) tensor.
        
        Returns
        =======
        y: (N, k, H', W') tensor.
        """
        x = self.net(x, keypoint=True) # 64, 128, 25, 25
        return self.regressor(x) # 64, 12, 25, 25


class RefineNet(nn.Module):
    """Network that generates images from feature maps and heatmaps."""

    def __init__(self, num_channels, hidden_size, embed_size):
        super(RefineNet, self).__init__()
        self.conv = nn.Conv2d(num_channels, 128, 4, stride=2)
        self.fc1 = nn.Linear(1024, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        """
        x: the transported feature map.
        """
        hidden = F.relu(self.conv(x))
        hidden = hidden.view(-1, 1024)
        hidden = F.relu(self.fc1(hidden))
        return self.fc2(hidden)

def compute_keypoint_location_mean(features):
    S_row = features.sum(-1)  # N, K, H # 64, 12, 25, 25 ==> 64, 12, 25
    S_col = features.sum(-2)  # N, K, W # 64, 12, 25, 25 ==> 64, 12, 25

    # N, K # 64, 12, 25 mul(25) ==> 64, 12
    u_row = S_row.mul(torch.linspace(-1, 1, S_row.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    # N, K # 64, 12, 25 mul(25) ==> 64, 12
    u_col = S_col.mul(torch.linspace(-1, 1, S_col.size(-1), dtype=features.dtype, device=features.device)).sum(-1)
    return torch.stack((u_row, u_col), -1) # N, K, 2 # 64, 12, 2


def gaussian_map(features, std=0.2):
    # features: (N, K, H, W)
    width, height = features.size(-1), features.size(-2) # 25, 25
    mu = compute_keypoint_location_mean(features)  # N, K, 2 # 64, 12, 2
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2] # 64, 12, 1 // 64, 12, 1
    y = torch.linspace(-1.0, 1.0, height, dtype=mu.dtype, device=mu.device) # 25 -1.0~1.0 25개로
    x = torch.linspace(-1.0, 1.0, width, dtype=mu.dtype, device=mu.device)  # 25 -1.0~1.0 25개로
    mu_y, mu_x = mu_y.unsqueeze(-1), mu_x.unsqueeze(-1) # 64, 12, 1 ==> 64, 12, 1, 1

    y = torch.reshape(y, [1, 1, height, 1])
    x = torch.reshape(x, [1, 1, 1, width])

    inv_std = 1 / std
    g_y = torch.pow(y - mu_y, 2)
    g_x = torch.pow(x - mu_x, 2)
    dist = (g_y + g_x) * inv_std**2
    g_yx = torch.exp(-dist)
    # g_yx = g_yx.permute([0, 2, 3, 1])
    return g_yx


def transport(source_keypoints, target_keypoints, source_features,
              target_features):
    """
    Args
    ====
    source_keypoints (N, K, H, W)
    target_keypoints (N, K, H, W)
    source_features (N, D, H, W)
    target_features (N, D, H, W)

    Returns
    =======
    """
    out = source_features   # 64, 128, 25, 25
    for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)): # 64, 25, 25 // 64, 25, 25
        out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * out + t.unsqueeze(1) * target_features
    return out              # 64, 128, 25, 25


class Transporter(nn.Module):

    def __init__(self, feature_encoder, point_net, refine_net, std=0.1):
        super(Transporter, self).__init__()
        self.feature_encoder = feature_encoder
        self.point_net = point_net
        self.refine_net = refine_net
        self.std = std
        
    def forward(self, source_images, target_images):
        source_features = self.feature_encoder(source_images)   # 64, 128, 25, 25
        target_features = self.feature_encoder(target_images)   # 64, 128, 25, 25

        source_keypoints = gaussian_map(
            spatial_softmax(self.point_net(source_images)), std=self.std)   # 64, 12, 25, 25

        target_keypoints = gaussian_map(
            spatial_softmax(self.point_net(target_images)), std=self.std)

        transported_features = transport(source_keypoints.detach(), # 64, 12, 25, 25
                                         target_keypoints,          # 64, 128, 25, 25
                                         source_features.detach(),  # 64, 12, 25, 25
                                         target_features)           # 64, 128, 25, 25

        assert transported_features.shape == target_features.shape  # transported_features 64, 128, 25, 25

        # reconstruction = self.refine_net(transported_features) # 64, 3, 100, 100
        return self.refine_net(transported_features)
