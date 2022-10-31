import torch

corr_matrix = torch.load("./pq0_correlation_matrix/pq_mask_jsd0.1/512/0.01_1.pt")
# corr_matrix2000 = torch.load("./pq0_correlation_matrix/pq_mask_jsd0.1/512/0.01_2001.pt")
print(corr_matrix, corr_matrix.shape, torch.max(corr_matrix), torch.min(corr_matrix))
avg_p = torch.mean(corr_matrix, dim=-1)
print(avg_p, avg_p.shape)
# print(corr_matrix2000)

