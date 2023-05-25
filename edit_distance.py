import torch
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch.functional as F

# total_result = np.load("./index/final/book_0.npy")
# # total_result = total_result.T  # (27, 256) -> (256, 27)
# np.seterr(invalid='ignore')
# total_result = total_result / total_result.sum(axis=1, keepdims=True)
#
# fig = plt.figure(figsize=(256, 27))
# # fig = plt.figure()
# ax = fig.gca()
# # hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
# sns.heatmap(total_result, ax=ax, cmap="Blues", linewidths=0.5, linecolor="gray", cbar=False)
# plt.savefig(f"./codebook2.png")
# exit()

num = 1000
dim = 1024
file_name = f"equss"
for i in range(6, 7):
    label = np.load(f"./index/{file_name}/label_{i}.npy")  # (240, 102400)
    data = np.load(f"./index/{file_name}/data_{i}.npy")  # (240, 102400, dim)
    print("*** Load npy ***")

    label = label.reshape(-1, 1)  # (240 * 102400, 1)
    data = data.reshape(-1, dim)  # (240 * 102400, dim)
    datas = np.concatenate((label, data), axis=-1)  # (240 * 102400, 1 + dim)
    print(datas.shape)
    np.random.shuffle(datas)
    total_result = [[[0 for col in range(dim)] for row in range(num)] for depth in range(27)]
    class_count = [0 for _ in range(27)]

    for idx in tqdm(datas):
        # idx (1+pq_num, )
        cls = idx[0].astype(np.int64)
        if cls == 255:
            continue
        if class_count[cls] < num:
            total_result[cls][class_count[cls]] = idx[1:]
            class_count[cls] += 1
    print(class_count)
    np.save(f"./index/{file_name}/final/quantized_features_{num}_{i}.npy", total_result)
    exit()

    # raw cosine similarity ver.
    results = np.zeros((27, 27))
    norm_results = np.zeros((27, 27))
    data = np.load(f"./index/{file_name}/final/quantized_features_{num}.npy")  # (27, 10000, dim)
    print(data.shape)
    for a in range(27):
        for b in range(27):
            print(f"---------------{a}-{b}------------------")
            comp_A = total_result[a]
            comp_B = total_result[b]

            num = np.dot(comp_A, comp_B.T)
            p1 = np.sqrt(np.sum(comp_A ** 2, axis=1))[:np.newaxis]
            p2 = np.sqrt(np.sum(comp_B ** 2, axis=1))[np.newaxis:]
            tmp = num / (p1 * p2)
            print(tmp, tmp.shape)
            mean_A_B = np.mean(tmp)
            print(mean_A_B, mean_A_B.shape)
            exit()

            norm_comp_A = F.normalize(comp_A, dim=-1)
            norm_comp_B = F.normalize(comp_B, dim=-1)
            print(norm_comp_A.shape, norm_comp_B.shape)

            norm_num = np.dot(norm_comp_A, norm_comp_B.T)
            norm_p1 = np.sqrt(np.sum(norm_comp_A ** 2, axis=1))[:np.newaxis]
            norm_p2 = np.sqrt(np.sum(norm_comp_B ** 2, axis=1))[np.newaxis:]
            norm_tmp = norm_num / (norm_p1 * norm_p2)
            print(norm_tmp, norm_tmp.shape)
            norm_mean_A_B = np.mean(norm_tmp)
            print(norm_mean_A_B, norm_mean_A_B.shape)
            exit()
            norm_results[a][b] = norm_mean_A_B

    np.save(f"./index/{file_name}/final/class_norm_mean_distance_{num}.npy", results)
    np.save(f"./index/{file_name}/final/class_norm_norm_mean_distance_{num}.npy", norm_results)
    results = results / results.sum(axis=1, keepdims=True)
    norm_results = norm_results / norm_results.sum(axis=1, keepdims=True)
    # results = np.square(results)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    # hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(results, ax=ax, vmin=0, vmax=1, cmap="YlGnBu", cbar=False)
    # sns.heatmap(results, ax=ax, vmin=0, vmax=1, cmap="Blues", cbar=False)
    # sns.heatmap(total_result, ax=ax, cmap="Blues", linewidths=0.5, linecolor="gray", cbar=False)
    plt.savefig(f"./index/{file_name}/final/norm_matrix_{i}.png")
    np.save(f"./index/{file_name}/final/class_mean_distance_{num}.npy", results)

    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.gca()
    # hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(norm_results, ax=ax2, vmin=0, vmax=1, cmap="YlGnBu", cbar=False)
    # sns.heatmap(results, ax=ax, vmin=0, vmax=1, cmap="Blues", cbar=False)
    # sns.heatmap(total_result, ax=ax, cmap="Blues", linewidths=0.5, linecolor="gray", cbar=False)
    plt.savefig(f"./index/{file_name}/final/norm_norm_matrix_{i}.png")
    np.save(f"./index/{file_name}/final/class_mean_norm_distance_{num}.npy", norm_results)

    # edit-distance ver.
    # results = np.zeros((27, 27))
    # # data = np.load(f"./index/{file_name}/final/quantized_features_{num}.npy")  # (27, 10000, dim)
    # # print(data.shape)
    # for a in range(27):
    #     for b in range(27):
    #         print(f"---------------{a}-{b}------------------")
    #         comp_A = np.expand_dims(total_result[a], axis=0)  # A-th class (1, 10000, pq_num)
    #         comp_B= np.expand_dims(total_result[b], axis=1)  # B-th class (10000, 1, pq_num)
    #
    #         # edit-distance
    #         tmp = comp_A == comp_B  # (10000, 10000, pq_num) equal
    #         print(tmp.shape)
    #         edit_matrix = np.sum(tmp, axis=2)  # (sum_score)
    #         # edit_matrix = np.divide(np.mean(tmp, axis=2), pq_num) # (mean_score)
    #         # print(edit_matrix, edit_matrix.shape)
    #         mean_A_B = np.mean(edit_matrix)
    #         print(mean_A_B)  # not equal => distance large -> low similiarty
    #         results[a][b] = mean_A_B
    # np.save(f"./index/{file_name}/final/class_norm_mean_distance_{num}.npy", results)
    # results = results / results.sum(axis=1, keepdims=True)
    # # results = np.square(results)
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.gca()
    # # hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    # sns.heatmap(results, ax=ax, vmin=0, vmax=1, cmap="YlGnBu", cbar=False)
    # # sns.heatmap(results, ax=ax, vmin=0, vmax=1, cmap="Blues", cbar=False)
    # # sns.heatmap(total_result, ax=ax, cmap="Blues", linewidths=0.5, linecolor="gray", cbar=False)
    # plt.savefig(f"./index/{file_name}/final/norm_matrix_{i}.png")
    # np.save(f"./index/{file_name}/final/class_mean_distance_{num}.npy", results)

# ------------------ (64, 27, 128) * 10 하기 ------------------ #
# pq_num = 4
# file_name = f"pq_{pq_num}"
# # '''
# # data : (240, 102400, 64)
# # '''
# for a in range(1, 8): # per-iteration
#     label_name = f"label_{a}"
#     data_name = f"data_{a}"
#
#     label = np.load(f"./index/{file_name}/{label_name}.npy").astype(np.uint8)  # (240, 102400)
#     data = np.load(f"./index/{file_name}/{data_name}.npy").astype(np.uint8)  # (240, 102400, 64)
#
#     labels = label.reshape(-1, 1)  # (240 * 102400, 1)
#     datas = data.reshape(-1, pq_num)  # (240 * 102400, pq_num)
#     datas = np.concatenate((labels, datas), axis=-1)  # (240 * 102400, 1 + pq_num)
#
#     total_result = [[[0 for col in range(256)] for row in range(27)] for depth in range(pq_num)]
#
#     for idx in tqdm(datas):
#         # idx : (1 + pq_num, )
#         cls = idx[0]
#         if cls == 255:
#             continue
#         for pq_indx in range(pq_num):
#             cw = idx[pq_indx + 1]
#             total_result[pq_indx][cls][cw] += 1
#
#     for i in range(pq_num):
#         os.makedirs(f"./index/{file_name}/{i}/", exist_ok=True)
#         np.save(f"./index/{file_name}/{i}/book{i}_{data_name}.npy", total_result[i])

# import pandas as pd
#
# for i in range(64):
#     d1 = np.load(f"./index/{i}/book{i}_data.npy")
#
#     for j in range(1, 10):
#         tp = np.load(f"./index/{i}/book{i}_data_{j}.npy")
#         d1 += tp
#     np.save(f'./index/{file_name}/final/book_{i}.npy', d1)
#     df = pd.DataFrame(d1)
#     df.to_csv(f'./index/{file_name}/final/csv/book_{i}.csv', index=False)

# ------------------ Visualize ------------------ #

# for a in range(27):
#     data = np.zeros((64, 256), dtype=int)
#     # for i in range(64):
#     data = np.load(f"./index/{file_name}/final/class/class_{a}.npy")
#     np.save(f"./index/{file_name}/final/class/csv/class_{a}.csv", data)

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.preprocessing import normalize
# import torch
# import torch.nn.functional as F
#
# for a in range(27):
#     depth_embeddings = np.load(f"./index/final/class/class_{a}.npy") # (64, 256)
#     depth_embeddings = torch.from_numpy(depth_embeddings).double()  # (27, 256)
#     depth_embeddings = (depth_embeddings - depth_embeddings.min(1, keepdim=True)[0]) / \
#                        depth_embeddings.max(1, keepdim=True)[0]
#     # depth_embeddings = F.normalize(depth_embeddings, dim=1)
#     depth_embeddings = depth_embeddings.numpy()
#
#     plt.figure()
#     # plt.matshow(depth_embeddings, cmap=plt.get_cmap('Blues'))
#     plt.matshow(depth_embeddings, cmap=plt.get_cmap('RdYlBu_r'))
#     plt.colorbar()
#     plt.savefig(f"./index/final/class/fig/class_{a}.png")

# ------------------ Calculate ------------------ #
# import numpy as np
# import torch
# d1 = np.load("./index/final/class/class_23.npy")
# d2 = np.load("./index/final/class/class_26.npy")
# d1 = torch.from_numpy(d1).double()
# d2 = torch.from_numpy(d2).double()
# cos = torch.nn.CosineSimilarity(dim=1)
# output = cos(d1, d2)
# print(output, output.shape, torch.mean(output))
