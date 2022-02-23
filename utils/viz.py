import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def viz_pcd_graph(points, edge_list):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ls = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
        pcd, pcd, edge_list
    )
    ls.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd, ls])


def viz_many(clouds: list, as_plot=False):
    pcds = []
    for i, p in enumerate(clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            p + i * np.array([3, 0, 0])
        )  # shift to see them side by side
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)


def viz_many_mpl(clouds: list[np.ndarray], d=3, ax=None):

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
