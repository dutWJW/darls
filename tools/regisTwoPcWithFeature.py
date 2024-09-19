import faiss, copy
from tools.o3d_tools import *
from tools.pointcloud import *
from tools.gridPointCloudDataStructure import GenerateSimilarPcBasedDistribution
from tools.uniformSamplePointCloud import UniformSamplePointCloud

def findGoodRegionByFeatureGpu(pc1, pc2, f1, f2):

    res = faiss.StandardGpuResources()
    d = 32
    index_flat = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(f1)
    k = 1
    _, idx = gpu_index_flat.search(f2, k)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(f2)
    k = 1
    _, idx_ = gpu_index_flat.search(f1, k)
    idx = idx.reshape(-1)
    idx_ = idx_.reshape(-1)
    equal_elements = np.equal(idx_[idx], np.arange(idx.shape[0]))
    equal_elements = equal_elements.astype(np.int8)

    delta_z = np.abs(pc2[:, 2] - pc1[idx][:, 2])
    delta_z_average = np.average(delta_z)
    w = np.exp(-delta_z_average * 0.01)
    diffPc2AndPc1_ = np.log(delta_z_average / (delta_z + 0.01) * w) > 0

    diffPc2AndPc1_ = diffPc2AndPc1_.astype(np.int8)
    diffPc2AndPc1 = diffPc2AndPc1_ + equal_elements
    equal_indices = np.nonzero(diffPc2AndPc1 > 1)[0]

    f1_index = idx[equal_indices]
    f2_index = equal_indices

    f1_index = np.unique(np.array(f1_index))
    pc1 = pc1[f1_index]
    pc2 = pc2[f2_index]
    f1 = f1[f1_index]
    f2 = f2[f2_index]
    return pc1, pc2, f1, f2

def Registration(pc1, f1, pc2, f2, new_pc1, new_pc2):
    pc1Index, pc2Index = UniformSamplePointCloud(pc1, pc2, 0.3)

    pc1, f1 = pc1[pc1Index], f1[pc1Index]
    pc2, f2 = pc2[pc2Index], f2[pc2Index]

    pc1, pc2, f1, f2 = findGoodRegionByFeatureGpu(pc1, pc2, f1, f2)

    pc1 = make_open3d_point_cloud(pc1)
    pc2 = make_open3d_point_cloud(pc2)

    ransac_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        pc1,
        pc2,
        make_open3d_feature_from_numpy(f1),
        make_open3d_feature_from_numpy(f2),
    )

    pc1_new_index = []
    pc2_new_index = []
    pc1_np = np.array(pc1.points).astype(np.float32)
    pc2_np = np.array(pc2.points).astype(np.float32)
    dim, measure = 3, faiss.METRIC_L2
    param = 'Flat'
    k = 50
    dis_sqrt = 9
    d = 3
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(new_pc1.astype(np.float32))
    _, idx = gpu_index_flat.search(pc1_np, k)

    idx = idx.reshape(-1)

    arr = np.arange(0, pc1_np.shape[0]).reshape(-1, 1)
    arr_copy = np.tile(arr, (1, k)).reshape(-1)

    diffMatrix = np.sum(np.square(pc1_np[arr_copy] - new_pc1[idx]), axis=1)
    pc1_new_index = np.nonzero(diffMatrix < dis_sqrt)[0]
    flatIndex = idx[pc1_new_index]
    flatIndex = np.unique(flatIndex)
    new_pc1 = new_pc1[flatIndex]

    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
    gpu_index_flat.add(new_pc2.astype(np.float32))
    _, idx = gpu_index_flat.search(pc2_np, k)

    idx = idx.reshape(-1)

    arr = np.arange(0, pc2_np.shape[0]).reshape(-1, 1)
    arr_copy = np.tile(arr, (1, k)).reshape(-1)

    diffMatrix = np.sum(np.square(pc2_np[arr_copy] - new_pc2[idx]), axis=1)
    pc2_new_index = np.nonzero(diffMatrix < dis_sqrt)[0]
    flatIndex = idx[pc2_new_index]
    flatIndex = np.unique(flatIndex)
    new_pc2 = new_pc2[flatIndex]

    ransac_result = o3d.pipelines.registration.registration_icp(
        make_open3d_point_cloud(new_pc1),
        make_open3d_point_cloud(new_pc2),
        0.7,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))

    new_pc1, new_pc2 = GenerateSimilarPcBasedDistribution(
        np.asarray(make_open3d_point_cloud(new_pc1).transform(ransac_result.transformation).points),
        new_pc2,
        1)

    ransac_result = o3d.pipelines.registration.registration_icp(
        make_open3d_point_cloud(np.asarray(make_open3d_point_cloud(new_pc1).transform(np.linalg.inv(ransac_result.transformation)).points)),
        make_open3d_point_cloud(new_pc2),
        0.3,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))

    return ransac_result.transformation
