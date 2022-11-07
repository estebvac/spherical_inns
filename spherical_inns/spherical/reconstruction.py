import numpy as np
import torch
import torch.nn as nn
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix


class ReconstructionModel(nn.Module):
    """
        This class presents the solution to the ellipsoid in cartesian coordinates, the function is obtained with the
        symbolic solution to the symbolic script:

        syms x y z theta phi r a b c x y z t u v x0 y0 z0 real
        Rz = [cos(phi)  -sin(phi) 0 ; ...
            sin(phi) cos(phi) 0; ...
             0  0 1];
        Ry = [cos(theta) 0 sin(theta);
              0 1 0; ...
              -sin(theta) 0 cos(theta)];
        R =  Ry*Rz
        r = sqrt(2*(cos(2*theta)+1))/2;

        x_c = sin(2*theta)/2 * cos(phi);
        y_c = -sin(2*theta)/2 * sin(phi);
        z_c = (cos(2*theta)+1)/2;

        A = diag([(r*a)^-2,(r*b)^-2,(r*c)^-2])
        X = [x-x_c; y-y_c; z-z_c];

        eq = X'*R'*A*R*X -1 == 0
        Z = simplify(solve(eq,z))

        The value of Z is then used to select the coordinates that contain a valid solution in the sphere, then the
        response at a given angle is calculated using the distance of each valid coordinate to the centroid of the
        ellipsoid as:
        f = ((x - x_c) ^ 2 + (y - y_c)^ 2 + (z - z_c)^ 2) / r ^ 2
    """

    def __init__(self, l_side, thickness: float = 0.03):
        """

        Parameters
        ----------
        l_side
        thickness
        """
        super(ReconstructionModel, self).__init__()

        # Create SphereHealpix Grid
        g_h = SphereHealpix(subdivisions=l_side)
        self.phi, self.theta = torch.tensor(g_h.signals['lon']), torch.tensor(np.pi / 2 - g_h.signals['lat'])
        self.phi = self.phi.to(torch.float32)
        self.theta = self.theta.to(torch.float32)

        self.a, self.b, self.c = thickness, 1.0, 1.0

        self.all_responses = self.ellipse_valid(torch.float32, "cpu")

    def forward(self, x, mix='clip'):
        output = torch.zeros_like(x)
        x_valid = x[:, :, self.theta <= np.pi / 2]

        if x.device != self.all_responses.device or x.dtype != self.all_responses.dtype:
            self.all_responses = self.all_responses.to(x.dtype).to(x.device)
        if mix == 'max':
            all_mixture = x_valid[:, :, :, None] * self.all_responses[None, None, :, :]
            mixture, _ = torch.max(all_mixture, dim=2)
        if mix == 'mean':
            all_mixture = x_valid[:, :, :, None] * self.all_responses[None, None, :, :]
            mixture = torch.sum(all_mixture, dim=2) / torch.clip(torch.count_nonzero(all_mixture, dim=2), 1)
            mixture = mixture.to(x.dtype).to(x.device)
        if mix == 'clip':
            mixture = torch.tensordot(x_valid, self.all_responses, dims=([2], [0]))
            mixture = torch.clip(mixture, 0, 1)
        else:
            mixture = torch.tensordot(x_valid, self.all_responses, dims=([2], [0]))

        output[:, :, self.theta <= np.pi / 2] = mixture
        return output

    def ellipse_valid(self, dtype, device):
        phi = self.phi[self.theta <= np.pi / 2].to(device).to(dtype)
        theta = self.theta[self.theta <= np.pi / 2].to(device).to(dtype)
        a, b, c = self.a, self.b, self.c

        # Convert to the grid of coordinates:
        x = torch.cos(phi) * torch.sin(theta)
        y = torch.sin(phi) * torch.sin(theta)
        x = torch.flatten(x)
        y = torch.flatten(y)
        phi, x = torch.meshgrid(phi, x)
        theta, y = torch.meshgrid(theta, y)

        theta = (theta + np.pi / 2) % np.pi
        phi = (phi + np.pi / 2) % (2 * np.pi)

        # Calculate the parameters for all possible ellipsoides:
        x_c = torch.sin(2 * theta) / 2 * torch.cos(phi)
        y_c = -torch.sin(2 * theta) / 2 * torch.sin(phi)
        z_c = (torch.cos(2 * theta) + 1) / 2
        r = torch.sqrt(2 * (torch.cos(2 * theta) + 1)) / 2 + 1e-5
        a2, b2, c2 = a ** 2, b ** 2, c ** 2

        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        cos_p, sin_p = torch.cos(phi), torch.sin(phi)

        # Solved ellipsoid function:
        # This is calculated by resolving the rotated ellipsoid equation in the cartesian coordinates

        z = (a * c * torch.sqrt(
            -b2 * cos_t ** 2 + b2 * cos_t ** 4 - b2 * y ** 2 - c2 * x ** 2 + a2 * b2 * cos_t ** 4 +
            b2 * c2 * cos_t ** 2 - b2 * c2 * cos_t ** 4 - b2 * x ** 2 * cos_p ** 2 + b2 * y ** 2 * cos_p ** 2 +
            c2 * x ** 2 * cos_p ** 2 - c2 * y ** 2 * cos_p ** 2 - a2 * x ** 2 * cos_t ** 2 +
            c2 * x ** 2 * cos_t ** 2 + b2 * x * y * torch.sin(phi * 2.0) - c2 * x * y * torch.sin(phi * 2.0) +
            a2 * x ** 2 * cos_p ** 2 * cos_t ** 2 - a2 * y ** 2 * cos_p ** 2 * cos_t ** 2 -
            c2 * x ** 2 * cos_p ** 2 * cos_t ** 2 + c2 * y ** 2 * cos_p ** 2 * cos_t ** 2 +
            b2 * x * cos_p * cos_t * sin_t * 2.0 - b2 * y * cos_t * sin_p * sin_t * 2.0 +
            c2 * x * y * cos_p * cos_t ** 2 * sin_p * 2.0 - a2 * x * y * cos_p * cos_t ** 2 * sin_p * 2.0) -
             a2 * b * cos_t ** 2 + a2 * b * cos_t ** 4 * 2.0 + b * c2 * cos_t ** 2 * 2.0 - b * c2 * cos_t ** 4 * 2.0 +
             a2 * b * x * cos_p * cos_t * sin_t - b * c2 * x * cos_p * cos_t * sin_t -
             a2 * b * y * cos_t * sin_p * sin_t + b * c2 * y * cos_t * sin_p * sin_t) / \
            (b * (a2 * cos_t ** 2 - c2 * cos_t ** 2 + c2))

        f = ((x - x_c) ** 2 + (y - y_c) ** 2 + (z - z_c) ** 2) / r ** 2
        f[f != f] = 0
        return f
