import torch

corr_matrix = torch.load("./pq0_correlation_matrix/pq_mask_jsd0.1/512/0.01_1.pt")
# corr_matrix2000 = torch.load("./pq0_correlation_matrix/pq_mask_jsd0.1/512/0.01_2001.pt")
print(corr_matrix, corr_matrix.shape)
print(torch.mean(corr_matrix, dim=-1))
# print(corr_matrix2000)

