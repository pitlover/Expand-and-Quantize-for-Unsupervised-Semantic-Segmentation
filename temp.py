import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

all_data = [ i for i in range(100) ]
tf_matrix = np.random.random((100, 100))

# set your own number of clusters
num_clusters = 2

m_km = KMeans(n_clusters=num_clusters).fit(tf_matrix)
m_clusters = m_km.labels_.tolist()

centers = np.array(m_km.cluster_centers_)

closest_data = []
for i in range(num_clusters):
    center_vec = centers[i]
    data_idx_within_i_cluster = [ idx for idx, clu_num in enumerate(m_clusters) if clu_num == i ]

    one_cluster_tf_matrix = np.zeros( (  len(data_idx_within_i_cluster) , centers.shape[1] ) )
    for row_num, data_idx in enumerate(data_idx_within_i_cluster):
        one_row = tf_matrix[data_idx]
        one_cluster_tf_matrix[row_num] = one_row

    closest, _ = pairwise_distances_argmin_min(center_vec, one_cluster_tf_matrix)
    closest_idx_in_one_cluster_tf_matrix = closest[0]
    closest_data_row_num = data_idx_within_i_cluster[closest_idx_in_one_cluster_tf_matrix]
    data_id = all_data[closest_data_row_num]

    closest_data.append(data_id)

closest_data = list(set(closest_data))

assert len(closest_data) == num_clusters
# import torch
#
# b = 32
# flat_x = torch.randint(100, (2, 3))
# split_x = torch.chunk(flat_x, chunks=b, dim=0)
# p_mu = torch.randint(100, (2, 3))
# p_logvar = torch.randint(100, (2, 3))
#
# print(f"flat_x : {flat_x}")
# print(f"p_mu : {p_mu}")
# print(f"p_logvar : {p_logvar}")
#
# positive = -0.5 * torch.sum(
#     torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1  # (bhw, d) -> (bhw)
# )
#
# print(f"minus : ", flat_x - p_mu)
# print(f"torch.square : ", torch.square(flat_x - p_mu))
# print(f"torch.exp : ", torch.exp(p_logvar))
#
# negative = -0.5 * torch.mean(
#     torch.sum(
#         torch.square(flat_x.unsqueeze(0) - p_mu.unsqueeze(1)) /
#         torch.exp(p_logvar.unsqueeze(1)),
#         dim=-1
#     ),
#     dim=-1
# )
#
# loss2 = positive - negative
# loss2 = torch.mean(loss2)
# print(f"__________# 1  {loss2}__________________________")
# exit()
# split_p_mu = torch.chunk(p_mu, chunks=1568, dim=0)  # split tensor for code optimization
# split_p_logvar = torch.chunk(p_logvar, chunks=1568, dim=0)  # split tensor for code optimization
# split_positive = torch.chunk(positive, chunks=1568, dim=0)
#
# loss = 0
# tmp = 0
# for iter in range(1568):
#     # flat_x_ = split_flat_x[iter]
#     p_mu_ = split_p_mu[iter]
#     p_logvar_ = split_p_logvar[iter]
#
#     negative = -0.5 * torch.mean(
#         torch.sum(
#             torch.square(flat_x.unsqueeze(0) - p_mu_.unsqueeze(1)) /
#             torch.exp(p_logvar_.unsqueeze(1)),
#             dim=-1
#         ),
#         dim=-1
#     )
#     loss_ = split_positive[iter] - negative  # bhw/28
#     loss += loss_.item()
#
# loss = loss / 1568
#
# print(f"__________# 2  {loss}__________________________")


split_p_mu = torch.chunk(p_mu, chunks=b, dim=0)  # split tensor for code optimization
split_p_logvar = torch.chunk(p_logvar, chunks=b, dim=0)  # split tensor for code optimization
split_positive = torch.chunk(positive, chunks=b, dim=0)

loss = 0
for iter in range(b):
    p_mu_ = split_p_mu[iter]
    p_logvar_ = split_p_logvar[iter]

    negative = -0.5 * torch.mean(
        torch.sum(
            torch.square(flat_x.unsqueeze(0) - p_mu_.unsqueeze(1)) /
            torch.exp(p_logvar_.unsqueeze(1)),
            dim=-1
        ),
        dim=-1
    )
    loss_ = split_positive[iter] - negative  # bhw/28
    loss += loss_

loss = loss / b
loss = torch.mean(loss)

print(f"__________# 3  {loss}__________________________")
