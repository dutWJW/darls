import copy
import numpy as np
import open3d as o3d

'''
vis only for pointcloud
'''
def vis_o3d(ptcloud_xyz, colors=[]):
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    if colors != []:
        ptcloud_xyz.colors = o3d.utility.Vector3dVector(colors)
    if isinstance(ptcloud_xyz, list):
        o3d.visualization.draw_geometries(ptcloud_xyz)
    else:
        o3d.visualization.draw_geometries([ptcloud_xyz])


def set_color_for_o3d(pt, r, g, b):
    colorNp = np.zeros([pt.points.__len__(), 3])
    colorNp[:, 0] = r
    colorNp[:, 1] = g
    colorNp[:, 2] = b
    pt.colors = o3d.utility.Vector3dVector(colorNp)
    return pt


def vis_np(ptcloud_xyz, colors=[]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
    if colors != []:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([FOR1, pcd])


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], xyz.shape[0], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def generate3DKdtreeByNp(np):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np)
    return o3d.geometry.KDTreeFlann(point_cloud), point_cloud

def get_matching_indices(source, target, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(
            point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append([i, j])
    return np.asarray(match_inds)

def downsample_point_cloud(xyzr, voxel_size=0.05):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzr[:, :3])
    pcd_ds, ds_trace, ds_ids = pcd.voxel_down_sample_and_trace(
        voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    inv_ids = [ids[0] for ids in ds_ids]
    ds_intensities = np.asarray(xyzr[:, 3])[inv_ids]
    return np.hstack((pcd_ds.points, ds_intensities.reshape(-1, 1)))


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def polygon(polygon_points):
    lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(polygon_points)
    points_pcd.paint_uniform_color([0, 0.3, 0])

    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    lines_pcd.points = o3d.utility.Vector3dVector(polygon_points)

    return lines_pcd, points_pcd

def read_pcd(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

def rm_g(origin_pcd):
    _, inliers = origin_pcd.segment_plane(1, 3, 100)
    return origin_pcd.select_by_index(inliers, invert=True)

def rigid_transform_3D(A, B):
    assert len(A) == len(B)
    N = A.shape[0];
    mu_A = np.mean(A, axis=0)
    mu_B = np.mean(B, axis=0)

    AA = A - np.tile(mu_A, (N, 1))
    BB = B - np.tile(mu_B, (N, 1))
    H = AA.T @ BB

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ A.T + B.T

    return R, t
def transform(pts, trans):
    """
    Applies the SE3 transformations, support torch.Tensor and np.ndarry.  Equation: trans_pts = R @ pts + t
    Input
        - pts: [num_pts, 3] or [bs, num_pts, 3], pts to be transformed
        - trans: [4, 4] or [bs, 4, 4], SE3 transformation matrix
    Output
        - pts: [num_pts, 3] or [bs, num_pts, 3] transformed pts
    """
    if len(pts.shape) == 3:
        trans_pts = trans[:, :3, :3] @ pts.permute(0,2,1) + trans[:, :3, 3:4]
        return trans_pts.permute(0,2,1)
    else:
        trans_pts = trans[:3, :3] @ pts.T + trans[:3, 3:4]
        return trans_pts.T

def plot_corres(sraw, traw, skpts, tkpts, trans, thr, align=False):
    '''
    Args:
        sraw:     array  [P, 3]
        traw:     array  [Q, 3]
        skpts:    array  [N, 3]
        tkpts:    array  [N, 3]
        trans:    array  [4, 4]

    Returns:

    '''
    len = skpts.shape[0]
    t_skpts = transform(skpts, trans)
    mask = (np.sum((t_skpts - tkpts) ** 2, axis=-1) < thr ** 2)
    inlier_rate = mask.sum() / len
    colors = np.zeros((len, 3))
    colors[mask] = [0, 1, 0]
    colors[~mask] = [1, 0, 0]

    # visulization
    offset = 0
    offset = np.array([0, 0, offset])[None]

    if align is True:
        sraw = transform(sraw, trans)
        skpts = transform(skpts, trans)
    sraw_pcd = make_open3d_point_cloud(sraw, [227/255, 207/255, 87/255])
    sraw_pcd.estimate_normals()
    traw_pcd = make_open3d_point_cloud(traw + offset, [0, 0.651, 0.929])
    traw_pcd.estimate_normals()

    vertice = np.concatenate([skpts, tkpts + offset], axis=0)
    line = np.concatenate([np.arange(0, len)[:, None], np.arange(0, len)[:, None] + len], axis=-1)
    lines_pcd = plot_correspondences(vertice, line, colors)

    o3d.visualization.draw_geometries([sraw_pcd, traw_pcd, lines_pcd])
    return inlier_rate


def plot_correspondences(points, lines, color):
    '''
    Args:
        points:  initial point sets [2N, 3]
        lines:   indices of points  [N, 2]
        color:

    Returns:
    '''
    lines_pcd = o3d.geometry.LineSet()
    lines_pcd.lines = o3d.utility.Vector2iVector(lines)
    if color is not None:
        if len(color) == 3:
            color = np.repeat(np.array(color)[np.newaxis, ...], lines.shape[0], axis=0)
        lines_pcd.colors = o3d.utility.Vector3dVector(color)
    lines_pcd.points = o3d.utility.Vector3dVector(points)

    return lines_pcd
