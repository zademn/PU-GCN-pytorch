import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from .pc_util import euler2mat
from typing import List
from open3d.visualization.draw_plotly import get_plotly_fig


def viz_pcd_graph(points: np.array, edge_list: np.array):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ls = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pcd, pcd, edge_list
    )
    ls.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd, ls])


def viz_many(clouds: List[np.ndarray], use_plotly=True):
    """Visualise a list of point clouds using Open3D
    Parameters
    ----------
    clouds : List[np.ndarray]
        The list of point clouds
    """
    pcds = []
    for i, p in enumerate(clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            p + i * np.array([3, 0, 0])
        )  # shift to see them side by side
        pcds.append(pcd)
    if use_plotly:
        return get_plotly_fig(pcds)
    else:
        o3d.visualization.draw_geometries(pcds)


def viz_many_mpl(clouds: List[np.ndarray], d=3, ax=None):
    """Visualize a list of point clouds using matplotlib

    Parameters
    ----------
    clouds : List[np.ndarray]
        The list of point clouds


    d : int, default=3
        distance between point clouds

    ax : _type_, default=None
        matplotlib ax to plot on
    """

    assert len(clouds) > 0
    points = None
    for i, p in enumerate(clouds):
        points = (
            p
            if points is None
            else np.concatenate([points, p + i * np.array([d, 0, 0])])
        )

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

    ax.scatter(*points.T)

    plt.show()


def draw_point_cloud(
    input_points,
    canvasSize=500,
    space=240,
    diameter=10,
    xrot=0,
    yrot=0,
    zrot=0,
    switch_xyz=[0, 1, 2],
    normalize=True,
):
    """Render point cloud to image with alpha channel.
    Input:
        points: Nx3 numpy array (+y is up direction)
    Output:
        gray image as numpy array of size canvasSizexcanvasSize
    """
    canvasSizeX = canvasSize
    canvasSizeY = canvasSize

    image = np.zeros((canvasSizeX, canvasSizeY))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (
                j - radius
            ) <= radius * radius:
                disk[i, j] = np.exp(
                    (-((i - radius) ** 2) - (j - radius) ** 2) / (radius**2)
                )
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (
        np.max(points[:, 2] - np.min(points[:, 2]))
    )
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSizeX / 2 + (x * space)
        yc = canvasSizeY / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc
        # image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3
        image[px, py] = image[px, py] * 0.7 + dv * 0.3

    val = np.max(image) + 1e-8
    val = np.percentile(image, 99.9)
    image = image / val
    mask = image == 0

    image[image > 1.0] = 1.0
    image = 1.0 - image
    # image = np.expand_dims(image, axis=-1)
    # image = np.concatenate((image*0.3+0.7,np.ones_like(image), np.ones_like(image)), axis=2)
    # image = colors.hsv_to_rgb(image)
    image[mask] = 1.0

    return image


def point_cloud_three_views(points, diameter=5):
    """input points Nx3 numpy array (+y is up direction).
    return an numpy array gray image of size 500x1500."""
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation
    # img1 = draw_point_cloud(points, xrot=90/180.0*np.pi,  yrot=0/180.0*np.pi, zrot=0/180.0*np.pi,diameter=diameter)
    # img2 = draw_point_cloud(points, xrot=180/180.0*np.pi, yrot=0/180.0*np.pi, zrot=0/180.0*np.pi,diameter=diameter)
    # img3 = draw_point_cloud(points, xrot=0/180.0*np.pi,  yrot=-90/180.0*np.pi, zrot=0/180.0*np.pi,diameter=diameter)
    # image_large = np.concatenate([img1, img2, img3], 1)

    img1 = draw_point_cloud(
        points,
        zrot=110 / 180.0 * np.pi,
        xrot=135 / 180.0 * np.pi,
        yrot=0 / 180.0 * np.pi,
        diameter=diameter,
    )
    img2 = draw_point_cloud(
        points,
        zrot=70 / 180.0 * np.pi,
        xrot=135 / 180.0 * np.pi,
        yrot=0 / 180.0 * np.pi,
        diameter=diameter,
    )
    img3 = draw_point_cloud(
        points,
        zrot=180.0 / 180.0 * np.pi,
        xrot=90 / 180.0 * np.pi,
        yrot=0 / 180.0 * np.pi,
        diameter=diameter,
    )
    image_large = np.concatenate([img1, img2, img3], 1)

    return image_large
