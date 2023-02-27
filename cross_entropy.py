import torch
import numpy as np
from tqdm import tqdm

num = 10000
pq_num = 64
file_name = f"pq_{pq_num}"

# for it in range(1, 11):
#     label = np.load(f"./index/pq_{pq_num}/label_{it}.npy")  # (240, 102400)
#     data = np.load(f"./index/pq_{pq_num}/data_{it}.npy")  # (240, 102400, 64)
#     print("*** Load npy ***")
#
#     label, data = torch.from_numpy(label).float().to("cuda:7"), torch.from_numpy(data).float().to("cuda:7")
#     label = label.reshape(-1, 1)
#     data = data.reshape(-1, pq_num)
#
#     print(data.shape, label.shape)
#     datas = torch.cat((label, data), dim=-1)  # (, 1 + pq_num)
#     rand_idx = torch.randperm(len(datas))
#     datas = datas[rand_idx]
#
#     # total_result = [[[0 for col in range(pq_num)] for row in range(num)] for depth in range(27)]
#     total_result = torch.zeros(27, num, pq_num, device="cuda:7")
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
##### .pt 100,000 개 분석..
total_dis = torch.zeros(27, 64)
for it in range(1, 11):
    total_result = torch.load(f'{num}_class{it}.pt')  # (27, 10000, 64)
    print(total_result.shape)

    total_report = torch.zeros(27)
    cls_codebook_entro = torch.zeros(27, 64)

    for cls in range(27):
        data = total_result[cls]  # (10000, 64)
        data = data.transpose(2, 1)  # (64, 10000)
        cls_entropy = 0
        for i in range(64):
            unique, counts = torch.unique(data[i], return_counts=True)
            unique = unique.astype(int)
            dic = dict(zip(unique, counts))
            for j in range(64):
                if dic.get(j) == None:
                    dic[j] = 0
            dis = torch.tensor(list(dic.values()))
            total_dis[cls][i] += dis
total_report = torch.zeros(27)
for cls in range(27):
    # total_dis : (27, 64) total_dis[i] : (64)
    cls_entropy = torch.zeros()
    total_dis[cls] = total_dis[cls] / sum(total_dis[cls])
    print(total_dis, total_dis.shape, sum(total_dis[cls]))  # (64)
    for i in range(64):
        entropy = torch.sum(np.multiply(-total_dis[cls][i], np.log2(total_dis[cls][i] + 1e-8)))
        cls_entropy += entropy
    total_report[cls] = cls_entropy
print(total_report)
print(cls_codebook_entro)
torch.save(total_report, "pq_total_report.pt")
