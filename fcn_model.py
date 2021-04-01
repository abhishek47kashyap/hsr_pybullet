import torchvision.models as models
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
import time

from self_attention_cv import ViT


class FCN(nn.Module):
    def __init__(self, num_rotations=16):
        super().__init__()

        self.num_rotations = num_rotations
        self.use_cuda = True

        modules = list(models.resnet18().children())[:-5]
        self.backbone = nn.Sequential(*modules)
        self.end = nn.Sequential(
            nn.Conv2d(66, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 1, 1, 1),
        )
        self.vit = ViT(img_dim=56, in_channels=66, patch_dim=4,
                #dim=64,
                blocks=2,
                #heads=1,
                #dim_linear_block=64,
                classification=False)
        #self.vit_fc = nn.Linear(512, 64)
        self.vit_upsample = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 66, 1, 1),
        )

    def cat_grid(self, input, affine_grid=None):
        x = torch.abs(torch.linspace(-0.5, 0.5, steps=input.shape[-2])).cuda() # side
        y = torch.tensor(torch.linspace(0, 1, steps=input.shape[-1])).cuda()  # forward
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)
        grid_x = grid_x.repeat(len(input), 1, 1, 1)
        grid_y = grid_y.repeat(len(input), 1, 1, 1)
        grid = torch.cat([grid_x, grid_y], 1)

        if affine_grid is not None:
            flow_grid = F.affine_grid(affine_grid, input.size())
            grid = F.grid_sample(grid, flow_grid, mode='nearest')

        x = torch.cat([input, grid], 1)

        return x

    def self_attention(self, x):
        z = self.vit(x)
        #z = self.vit_fc(z)
        z = z.view(-1, 14, 14, 512)
        z = z.permute(0, 3, 1, 2)
        z = self.vit_upsample(z)
        x = F.relu(x + z)

        return x

    def forward(self, x):
        bs = len(x)
        # y = self.end(self.backbone(x))

        # assert x.shape[-2:] == y.shape[-2:], 'input =/= output shape {} {}'.format(x.shape, y.shape)
        output_prob = []

        # x = x[:, 0:1]
        # x = self.cat_meshgrid(x)

        if self.num_rotations == 1:
            out = self.end(self.self_attention(self.cat_grid(self.backbone(x))))
            return out
        else:
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray(
                    [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2, 3, 1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                affine_mat_before = affine_mat_before.repeat(bs, 1, 1)

                #print(affine_mat_before.shape, x.shape)
                #print(affine_mat_before.is_cuda, x.is_cuda)

                if self.use_cuda:
                    affine_mat_before = affine_mat_before.cuda()
                    flow_grid_before = F.affine_grid(affine_mat_before, x.size())
                else:
                    affine_mat_before = affine_mat_before.detach()
                    flow_grid_before = F.affine_grid(affine_mat_before, x.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_depth = F.grid_sample(x.detach().cuda(), flow_grid_before, mode='nearest')
                else:
                    rotate_depth = F.grid_sample(x.detach(), flow_grid_before, mode='nearest')

                # Compute intermediate features
                output_map = self.end(self.self_attention(self.cat_grid(self.backbone(rotate_depth), affine_mat_before)))

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray(
                    [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2, 3, 1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                affine_mat_after = affine_mat_after.repeat(bs, 1, 1)

                if self.use_cuda:
                    flow_grid_after = F.affine_grid(affine_mat_after.detach().cuda(),
                                                    output_map.size())
                else:
                    flow_grid_after = F.affine_grid(affine_mat_after.detach(),
                                                    output_map.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append(F.grid_sample(output_map, flow_grid_after, mode='nearest'))

        return output_prob


if __name__ == '__main__':
    model = FCN()
    model.cuda()
    model.eval()

    while True:
        y = model(torch.rand((1, 3, 224, 224)))
        print(torch.stack(y).shape)
