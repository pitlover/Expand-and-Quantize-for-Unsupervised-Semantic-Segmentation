import torch
import numpy as np
from tqdm import tqdm

num = 10000
pq_num = 64
file_name = f"pq_64"
#
# for it in range(1, 11):
#     label = np.load(f"./index/{file_name}/label_{it}.npy")  # (240, 102400)
#     data = np.load(f"./index/{file_name}/data_{it}.npy")  # (240, 102400, 64)
#     print("*** Load npy ***")
#
#     label, data = torch.from_numpy(label).float().to("cuda:5"), torch.from_numpy(data).float().to("cuda:5")
#     label = label.reshape(-1, 1)
#     data = data.reshape(-1, pq_num)
#
#     print(data.shape, label.shape)
#     datas = torch.cat((label, data), dim=-1)  # (, 1 + pq_num)
#     rand_idx = torch.randperm(len(datas))
#     datas = datas[rand_idx]
#
#     # total_result = [[[0 for col in range(pq_num)] for row in range(num)] for depth in range(27)]
#     total_result = torch.zeros(27, num, pq_num, device="cuda:5")
#     # class_count = [0 for _ in range(27)]
#     class_count = [0 for _ in range(27)]
#
#     for idx in tqdm(datas):
#         # idx (1+pq_num, )
#         cls = idx[0].to(torch.int64)
#         if cls == 255:
#             continue
#         print(cls)
#         if class_count[cls] < num:
#             total_result[cls][class_count[cls]] = idx[1:]
#             class_count[cls] += 1
#     print(class_count)
#     print(total_result.shape)  # (27, 10000, num_pq)
#
#     torch.save(total_result, f'{num}_class{it}.pt')
#
# ##### .pt 100,000 개 분석..
# total_dis = torch.zeros(27, 64)
# for it in range(1, 11):
#     total_result = torch.load(f'./num/{num}_class{it}.pt')  # (27, 10000, 64)
#     print(total_result.shape)
#
#     cls_codebook_entro = torch.zeros(27, 64)
#
#     for cls in range(27):
#         data = total_result[cls]  # (10000, 64)
#         data = data.transpose(2, 1)  # (64, 10000)
#         cls_entropy = 0
#         for i in range(64):
#             unique, counts = torch.unique(data[i], return_counts=True)
#             unique = unique.astype(int)
#             dic = dict(zip(unique, counts))
#             for j in range(64):
#                 if dic.get(j) == None:
#                     dic[j] = 0
#             dis = torch.tensor(list(dic.values()))
#             total_dis[cls][i] += dis
# total_report = torch.zeros(27)
# for cls in range(27):
#     # total_dis : (27, 64) total_dis[i] : (64)
#     cls_entropy = torch.zeros()
#     total_dis[cls] = total_dis[cls] / sum(total_dis[cls])
#     print(total_dis[cls], total_dis[cls].shape, sum(total_dis[cls]))  # (64), sum : 100,000
#     for i in range(64):
#         entropy = torch.sum(torch.multiply(-total_dis[cls][i], torch.log2(total_dis[cls][i] + 1e-8)))
#         cls_entropy += entropy
#     total_report[cls] = cls_entropy
# print(total_report)
# print(cls_codebook_entro)
# torch.save(total_report, "pq_total_report.pt")


### class 별 visualization
import matplotlib.pyplot as plt

data = np.load(f"./index/{file_name}/final/class_index_10000.npy")  # (27, 10000, 64)
data = np.transpose(data, (0, 2, 1))  # (27, 64, 100000)
print(data.shape)
label = [
    "electronic",
    "appliance",
    "food",
    "furniture",
    "indoor",
    "kitchen",
    "accessory",
    "animal",
    "outdoor",
    "person",
    "sports",
    "vehicle",
    "ceiling",
    "floor",
    "food",
    "furniture",
    "rawmaterial",
    "textile",
    "wall",
    "window",
    "building",
    "ground",
    "plant",
    "sky",
    "solid",
    "structural",
    "water",
]
entropy = [
    286.5117,
    166.5303,
    298.8867,
    310.1024,
    323.4314,
    345.1441,
    375.1106,
    109.3044,
    262.8013,
    97.9323,
    315.6374,
    262.2051,
    258.4734,
    288.37,
    236.5934,
    280.6807,
    351.9307,
    305.3531,
    294.6149,
    227.8626,
    313.5121,
    220.4806,
    141.4855,
    196.9115,
    143.4397,
    121.8939,
    126.6447,
]
for cls in range(27):  # (64, 10000)
    print(f"------------- {cls} --------------")
    cls_data = np.zeros(shape=(64, 256))
    cls_entropy = 0
    for codebook in range(64):
        # print(f"--{codebook} --")
        values, count = np.unique(data[cls][codebook], return_counts=True)
        values = values.astype(int)
        dic = dict(zip(values, count))
        for a in range(256):
            if dic.get(a) == None:
                dic[a] = 0
        dic = dict(sorted(dic.items(), key=lambda x: x[0]))
        dis = np.array(list(dic.values()))  # (256, )
        # print(np.sum(dis))
        dis = dis / np.sum(dis) # (256,)
        cls_data[codebook] = dis
        cls_entropy += np.sum(np.multiply(-dis, np.log2(dis + 1e-8)))

    plt.figure()
    plt.matshow(cls_data, cmap=plt.get_cmap('prism'))
    # plt.matshow(cls_data, cmap=plt.get_cmap('prism'), vmin=0, vmax=0.5)
    plt.savefig(f"./index/{file_name}/final/new_fig/class{cls}_{label[cls]}_{entropy[cls]}.png")
