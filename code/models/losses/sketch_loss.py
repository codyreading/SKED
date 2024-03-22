import torch
import torch.nn as nn

import os
import sys
sys.path.append('../../../code')
sys.path.append('../../models')
from utils import sketch_utils
import utils.rend_utils as rend_utils
from utils.general import ce_pq_loss
from utils.mesh_utils import MeshOBJ
import numpy as np
from PIL import Image

from render_wrapper import Renderer

import cv2
from igl import read_obj

from scipy.spatial import KDTree

class SketchLoss(nn.Module):
    def __init__(self, sketch_canvases, sketch_poses,
                 sketch_intrinsics, sketch_distances, proximal_surface,
                 use_kdtree = True, normalize = True, base_mesh = None,
                 nerf_gt = None, color_lambda = 5.):
        #sketch_canvases: (B, H, W) sketch canvases
        #sketch_poses: (B, 4, 4) camera2world poses for each sketch canvas
        #sketch_intrinsics: (B, 4) camera intrinsics for each sketch canvas
        #proximal_surface: sigma for weighted loss
        #occ_threshold: threshold for determininng g.t occupancy of each point based on distance
        super().__init__()
        self.sketch_canvases = sketch_canvases
        self.H, self.W = sketch_canvases.shape[1:]
        self.num_sketches = len(sketch_canvases)

        self.sketch_poses = sketch_poses
        self.sketch_intrinsics = sketch_intrinsics
        self.sketch_distances = sketch_distances

        self.normalize = normalize


        self.proximal_surface = proximal_surface

        self.delta = 0.07

        self.base_mesh = base_mesh

        self.nerf_gt = nerf_gt
        self.color_lambda = color_lambda


    def forward(self, xyzs, sigmas, colors = None, color_lambda = 5.):
        # xyzs: (N, 3) points in the 3D space
        # sigmas: (N) sigma values for each point
        # colors: (N, 3) colors for each point (optional). Can be None if we don't want to constrain the color
        # return: (N) sketch loss for each point
        # TODO: Update this to only put occupancy in sketch areas

        with torch.no_grad():
            projected_points = torch.round(rend_utils.batch_proj_points2image(xyzs, self.sketch_poses, self.sketch_intrinsics)).float()#(B, N, 2)
            if self.normalize:
                projected_points = projected_points / torch.tensor([self.H, self.W], dtype = torch.float32, device = projected_points.device).reshape(1, 1, 2)

            projected_points = projected_points.unsqueeze(2) # (B, N, 1, 2)
            min_dists = torch.nn.functional.grid_sample(input=self.sketch_distances, grid=projected_points, padding_mode="border")
            min_dists = min_dists.squeeze(-1).squeeze(1) # B, 1, N, 1 -> B, N
            D = min_dists.mean(dim = 0) #(N)


            if self.base_mesh is not None:
                occ = self.base_mesh.winding_number(xyzs)
                occ = (occ > 0.5).float()
            elif self.nerf_gt is not None:
                gt = self.nerf_gt.density(xyzs)
                color_gt, sigma_gt= gt['albedo'], gt['sigma']
                occ = (sigma_gt > self.nerf_gt.density_thresh).float()

            else:
                occ = (D < 0.001).float()


            weights = self.compute_weights(D) #(N)

        nerf_occ = 1 - torch.exp(-self.delta * sigmas)
        nerf_occ = nerf_occ.clamp(min=0, max=1.1)

        loss = ce_pq_loss(occ, nerf_occ, weight = weights)
        #loss = (((sigma_gt - sigmas) ** 2) * weights).sum()
        if colors is not None:
            #l2 loss between color_gt and colors
            color_loss = (torch.sum((colors - color_gt)**2, dim = -1) * weights).sum()
            loss = loss + color_loss * self.color_lambda
        return loss


    def compute_weights(self, D):
        #D: (N) distance to sketch points
        #return: (N) weight for each point
        return 1 - torch.exp(-(D ** 2) / (2 * (self.proximal_surface ** 2)))
