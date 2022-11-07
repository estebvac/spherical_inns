import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pygsp.graphs.nngraphs.spherehealpix import SphereHealpix
from scipy.interpolate import griddata


class Scattering2HealpixInterpolation:
    def __init__(self, radius_pixel=0.18, distance_h=13.2, n_pix_interpolation=3, l_side=8, radial_norm: int = True):

        self.radius_pixel = radius_pixel
        self.distance_h = distance_h
        self.n_pix_interpolation = n_pix_interpolation
        self.radial_norm = radial_norm

        # Distances for the interpolation.
        self.angular_distances = AngularDistance(l_side=l_side)
        self.valid_values = self.angular_distances.valid

        # Create grid for the output space:
        g_h = SphereHealpix(subdivisions=l_side)
        self.phi_out, self.theta_out = g_h.signals['lon'], np.pi / 2 - g_h.signals['lat']  # phi, theta

        self.xyz_out = np.stack([np.cos(self.phi_out) * np.sin(self.theta_out),
                                 np.sin(self.phi_out) * np.sin(self.theta_out),
                                 np.cos(self.theta_out)], axis=1)

        self.theta_unique = np.unique(self.theta_out)

        self.theta_mask = list()
        for theta_val in self.theta_unique:
            self.theta_mask.append(np.nonzero(self.theta_out == theta_val))

    def __call__(self, image):
        image = image.astype(np.float32)
        img_min = image.min(axis=(0, 1))
        img_max = image.max(axis=(0, 1))

        image_norm = (image - img_min) / (img_max - img_min)  # * 4091
        img_size = image_norm.shape[0]
        # print(img_size)

        x0, y0 = get_center(image_norm)
        r_circle = np.min([image_norm.shape[1] - y0, y0, image_norm.shape[0] - x0, x0])
        border = 2
        angular_distances = self.angular_distances(img_size, r_circle + border)
        img_valid = image_norm[y0 - r_circle:y0 + r_circle, x0 - r_circle:x0 + r_circle, :]
        img_valid_border = cv2.copyMakeBorder(img_valid, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

        signal_interp = self.interpolate_sphere(img_valid_border, angular_distances)
        channels = 1 if len(image_norm.shape) == 2 else image_norm.shape[2]
        output_signal = np.zeros(self.phi_out.shape + (channels,))
        output_signal[self.valid_values, :] = signal_interp

        # theta max:
        distance_kernel = 256 / img_size * self.radius_pixel
        max_theta = np.arctan(distance_kernel / self.distance_h * r_circle)
        output_signal[self.theta_out > max_theta] = 0

        max_theta = max_theta + 10 * np.pi / 180

        norm_weight_max, norm_weight_min = radial_weights(output_signal, np.pi / 2, self.theta_unique, self.theta_mask)

        if self.radial_norm == 2:
            output_signal = (output_signal - norm_weight_min) / np.clip(norm_weight_max - norm_weight_min, 0.1, np.inf)
        elif self.radial_norm:
            output_signal = output_signal / np.clip(norm_weight_max, 0.1, np.inf)

        output_signal /= output_signal.max()  # This normalizes the scattering pattern to one
        mask = np.zeros_like(output_signal)
        mask[self.theta_out < max_theta] = 1

        # self.plot(image_norm, output_signal)

        return torch.tensor(output_signal), torch.tensor(norm_weight_max), torch.tensor(mask), x0, y0

    def interpolate_sphere(self, image, diff):
        # Reshape the image:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        signal = image.reshape(-1, image.shape[-1])

        # Get the pixel position for the interpolation:
        arg1 = np.argpartition(-diff, self.n_pix_interpolation)[:, :self.n_pix_interpolation]
        # np.argsort(diff)[:, -self.n_int_pixels:]
        arg2 = [np.arange(diff.shape[0])[:, np.newaxis], arg1]

        # Compute the interpolation in the sphere:
        weights = diff[arg2]
        threshold = 1 / ((5 * np.pi / 180) + 0.2)  # Pixels which are more than 5 deg away will not be considered
        weights_z = np.where(weights > threshold, weights, 0)
        signal_interp = np.sum(np.expand_dims(weights_z, axis=-1) * signal[arg1], axis=1) / np.expand_dims(
            np.sum(weights, axis=1), axis=-1)

        # self.plot(image, signal_interp)
        return signal_interp

    def plot(self, image, signal):
        plt.subplot(1, 2, 1)
        plt.imshow(image.sum(-1))
        plt.subplot(1, 2, 2)
        plt.scatter(-self.xyz_out[:, 1][self.valid_values], self.xyz_out[:, 0][self.valid_values],
                    c=signal[:, 0][self.valid_values], vmin=0, vmax=signal.max(),
                    cmap='inferno')
        plt.gca().set_aspect('equal')
        plt.show()


class Scattering2HealpixLinear:
    def __init__(self, radius_pixel=0.18, max_dim=100, l_s=8):
        self.radius_pixel = radius_pixel

        # Create grid for the input space:
        mesh_grid = np.meshgrid(np.arange(0, max_dim), np.arange(0, max_dim))
        self.coord_array = np.asarray(mesh_grid).transpose((1, 2, 0))

        # Create grid for the output space:
        g_h = SphereHealpix(subdivisions=l_s)
        self.phi, self.theta = g_h.signals['lon'], np.pi / 2 - g_h.signals['lat']  # phi, theta

    def __call__(self, image, n_pixels, distance_h=13.2):
        #     n_pixels = 2

        if n_pixels != 81:
            n = n_pixels

        distance_kernel = n_pixels * self.radius_pixel

        # Get centroid for the conversion:
        image = image.astype(np.float32)
        x0, y0 = get_center(image)
        r_circle = np.min([image.shape[1] - y0, y0, image.shape[0] - x0, x0])

        # theta max:
        theta_max = np.arctan(distance_kernel / distance_h * r_circle)
        # Select the north hemisphere of the sphere:
        self.valid_values = self.theta <= theta_max
        self.phi_valid = self.phi[self.valid_values]
        self.theta_valid = self.theta[self.valid_values]

        # Create an image grid
        grid_xy = np.concatenate(self.coord_array[:image.shape[0], :image.shape[1]])
        x_, y_ = grid_xy[:, 0], grid_xy[:, 1]

        # Compute the spherical coordinates
        r_input = np.sqrt((x_ - x0) ** 2 + (y_ - y0) ** 2)
        theta_input = np.arctan(distance_kernel / distance_h * r_input)
        phi_input = np.arctan2((x_ - x0), (y_ - y0)) % (2 * np.pi)
        data_flatten = image[:, :, 0].flatten()

        # CREATE AN EXTENDED COORDINATE SYSTEM:
        theta_input_ext = np.concatenate((theta_input, theta_input))
        phi_input_ext = np.concatenate((phi_input, phi_input + 2 * np.pi))
        polar_flat_ext = np.stack([phi_input_ext, theta_input_ext], axis=-1)
        data_flatten_ext = np.concatenate((data_flatten, data_flatten))

        new_image_flat = griddata(polar_flat_ext, data_flatten_ext,
                                  (self.phi_valid, self.theta_valid),
                                  method='linear', fill_value=0)

        result = np.zeros_like(self.phi)
        result[self.valid_values] = new_image_flat
        return new_image_flat, self.phi_valid, self.theta_valid


def get_center(image):
    kernel = max(image.shape[0] // 8, 3)
    kernel = kernel + 1 if kernel % 2 == 0 else kernel
    dst = filter_with_pad(image, kernel)

    if len(dst.shape) == 2:
        dst = dst.copy()[..., np.newaxis]

    channel_centers = np.empty((dst.shape[2], 2), dtype=int)
    for color_channel in range(dst.shape[2]):
        max_value = dst[:, :, color_channel].max()
        max_positions = np.argwhere(dst[:, :, color_channel] >= max_value)
        max_positions.sort()
        if len(max_positions) % 2 == 0:
            channel_centers[color_channel] = max_positions[len(max_positions) // 2]
        else:
            channel_centers[color_channel] = (max_positions[len(max_positions) // 2] + max_positions[
                len(max_positions) // 2]) / 2.0

    x_mean, y_mean = channel_centers.mean(axis=0)
    return int(np.rint(y_mean)), int(np.rint(x_mean))


def filter_with_pad(image, kernel):
    displacement = kernel // 2

    constant_val = (0,) * image.shape[-1]

    if displacement > 0:
        img_pad = cv2.copyMakeBorder(image, displacement, displacement, displacement, displacement, cv2.BORDER_CONSTANT,
                                     constant_val)
    else:
        img_pad = image

    dst = cv2.GaussianBlur(img_pad, (kernel, kernel), cv2.BORDER_DEFAULT)
    if displacement > 0:
        dst = dst[displacement:-displacement, displacement:-displacement]

    return dst


class AngularDistance:
    def __init__(self, l_side, distance_h=13.2, led_radius=0.18):
        self.num_pixels = [81, 50, 64, 32, 16]
        self.led_radius = led_radius
        self.distance_h = distance_h

        # Create grid for the output space:
        g_h = SphereHealpix(subdivisions=l_side)
        self.phi_out, self.theta_out = g_h.signals['lon'], np.pi / 2 - g_h.signals['lat']  # phi, theta

        self.xyz_out = np.stack([np.cos(self.phi_out) * np.sin(self.theta_out),
                                 np.sin(self.phi_out) * np.sin(self.theta_out),
                                 np.cos(self.theta_out)], axis=1)

        self.valid = self.theta_out <= np.pi / 2
        self.xyz_out = self.xyz_out[self.valid]

        self.list_distances = list()

        for n_pix in self.num_pixels:
            diff = self.create_angular_distances(n_pix)
            self.list_distances.append(diff)

    def create_angular_distances(self, n_pix):
        distance_kernel = 256 / n_pix * self.led_radius

        # Create an image grid with centroid in its center
        mesh_grid = np.meshgrid(np.arange(0, n_pix), np.arange(0, n_pix))
        coord_array = np.asarray(mesh_grid).transpose((1, 2, 0))
        x_, y_ = coord_array[:, :, 0], coord_array[:, :, 1]
        x0, y0 = n_pix // 2, n_pix // 2

        # Compute the spherical coordinates
        r_input = np.sqrt((x_ - x0) ** 2 + (y_ - y0) ** 2)
        phi_input = np.arctan2((x_ - x0), (y_ - y0)) % (2 * np.pi)

        theta_input = np.arctan(distance_kernel / self.distance_h * r_input)
        # Compute the spherical surface
        xyz_input = np.stack([np.cos(phi_input) * np.sin(theta_input),
                              np.sin(phi_input) * np.sin(theta_input),
                              np.cos(theta_input)], axis=-1)

        xyz_input = xyz_input.reshape(-1, xyz_input.shape[-1])

        angular_distance = np.arccos(np.clip(np.dot(self.xyz_out, xyz_input.T), -1, 1))
        diff = 1 / (np.abs(angular_distance) + 0.2)
        # plt.imshow(diff[-1].reshape(n_pix,n_pix)), plt.show()
        return diff

    def __call__(self, n_pix, r_circle):
        if n_pix in self.num_pixels:
            index_ = self.num_pixels.index(n_pix)
        else:
            diff = self.create_angular_distances(n_pix)
            self.list_distances.append(diff)
            index_ = -1

        diff_flat = self.list_distances[index_]
        diff_xy = diff_flat.reshape(-1, n_pix, n_pix)
        x_m = n_pix // 2
        diff = diff_xy[:, x_m - r_circle:x_m + r_circle, x_m - r_circle:x_m + r_circle]
        diff = diff.reshape(diff.shape[0], -1)
        # plt.imshow(diff[-1].reshape(r_circle*2, r_circle*2)), plt.show()
        return diff


def radial_weights(signal, max_theta, theta_unique, theta_mask):
    norm_weight_max = np.zeros_like(signal)
    norm_weight_min = np.zeros_like(signal)
    for index, theta_val in enumerate(theta_unique):
        if theta_val > max_theta:
            break
        coordinates = theta_mask[index]
        norm_weight_max[coordinates, :] = signal[coordinates, :].max()
        norm_weight_min[coordinates, :] = signal[coordinates, :].min()

    return norm_weight_max, norm_weight_min

# class Scattering2HealpixInterpolationDeprecated:
#     def __init__(self, radius_pixel=0.18, max_dim=100, distance_h=13.2, n_int_pixels=3, l_side=8, radial_norm=True):
#         self.radius_pixel = radius_pixel
#         self.distance_h = distance_h
#         self.n_int_pixels = n_int_pixels
#         self.radial_norm = radial_norm
#
#         # Create grid for the input space:
#         mesh_grid = np.meshgrid(np.arange(0, max_dim), np.arange(0, max_dim))
#         self.coord_array = np.asarray(mesh_grid).transpose((1, 2, 0))
#
#         # Create grid for the output space:
#         g_h = SphereHealpix(subdivisions=l_side)
#         self.phi_out, self.theta_out = g_h.signals['lon'], np.pi / 2 - g_h.signals['lat']  # phi, theta
#
#         self.xyz_out = np.stack([np.cos(self.phi_out) * np.sin(self.theta_out),
#                                  np.sin(self.phi_out) * np.sin(self.theta_out),
#                                  np.cos(self.theta_out)], axis=1)
#
#         self.theta_unique = np.unique(self.theta_out)
#
#         self.theta_mask = list()
#         for theta_val in self.theta_unique:
#             self.theta_mask.append(np.nonzero(self.theta_out == theta_val))
#
#     def __call__(self, image, n_pixels):
#         #     n_pixels = 2
#         distance_kernel = n_pixels * self.radius_pixel  # d in the paper
#
#         # Get centroid central pixels
#         image = image.astype(np.float32)
#         x0, y0 = get_center(image)
#         r_circle = np.min([image.shape[1] - y0, y0, image.shape[0] - x0, x0])
#
#         # theta max:
#         theta_max = np.arctan(distance_kernel / self.distance_h * r_circle)
#
#         # Select the north hemisphere of the sphere:
#         max_theta = theta_max + 10 * np.pi / 180
#         valid_values = self.theta_out <= max_theta  # it could be possible to use more of this
#         xyz_out = self.xyz_out[valid_values, :]
#
#         # Compute input grid:
#         xyz_input = self.calculate_input_grid(image.shape, x0, y0, distance_kernel)
#         signal_interp = self.interpolate_sphere(image, xyz_input, xyz_out)
#
#         channels = 1 if len(image.shape) == 2 else image.shape[2]
#         output_signal = np.zeros(self.phi_out.shape + (channels,))
#         output_signal[valid_values, :] = signal_interp
#
#         norm_weight = radial_weights(output_signal, max_theta, self.theta_unique, self.theta_mask)
#
#         if self.radial_norm:
#             return output_signal / torch.clip(norm_weight, 1)
#         else:
#             return output_signal
#
#     def calculate_input_grid(self, image_shape, x0, y0, distance_kernel):
#         # Create an image grid
#         grid_xy = np.concatenate(self.coord_array[:image_shape[0], :image_shape[1]])
#         x_, y_ = grid_xy[:, 0], grid_xy[:, 1]
#
#         # Compute the spherical coordinates
#         r_input = np.sqrt((x_ - x0) ** 2 + (y_ - y0) ** 2)
#         theta_input = np.arctan(distance_kernel / self.distance_h * r_input)
#         phi_input = np.arctan2((x_ - x0), (y_ - y0)) % (2 * np.pi)
#
#         # Compute the spherical surface
#         xyz_input = np.stack([np.cos(phi_input) * np.sin(theta_input),
#                               np.sin(phi_input) * np.sin(theta_input),
#                               np.cos(theta_input)], axis=1)
#
#         return xyz_input
#
#     def interpolate_sphere(self, image, coor_input, coor_heal):
#         # Reshape the image:
#         if len(image.shape) == 2:
#             image = np.expand_dims(image, axis=-1)
#
#         signal = image.reshape(-1, image.shape[-1])
#
#         # Calculate the angular distance between all points:
#         angular_distance = np.arccos(np.clip(np.dot(coor_heal, coor_input.T), -1, 1))
#         diff = 1 / (np.abs(angular_distance) + 0.2)
#
#         # Get the pixel position for the interpolation:
#         arg1 = np.argpartition(-diff, self.n_int_pixels)[:, :self.n_int_pixels]
#         # np.argsort(diff)[:, -self.n_int_pixels:]
#         arg2 = [np.arange(len(coor_heal))[:, np.newaxis], arg1]
#
#         # Compute the interpolation in the sphere:
#         signal_interp = np.sum(np.expand_dims(diff[arg2], axis=-1) * signal[arg1], axis=1) / np.expand_dims(
#             np.sum(diff[arg2], axis=1), axis=-1)
#
#         return signal_interp
