import torch
import torch.nn.functional as F

flat_x = torch.randint(100, (10, 10))
p_mu = torch.randint(100, (10, 10))
p_logvar = torch.randint(100, (10, 10))

# positive = -0.5 * torch.sum(
#     torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1  # (bhw, d) -> (bhw)
# )
# print(positive.shape)
#
# negative = -0.5 * torch.mean(
#             torch.sum(
#                 torch.square(flat_x.unsqueeze(0) - p_mu.unsqueeze(1)) /
#                 torch.exp(p_logvar.unsqueeze(1)),
#                 dim=-1
#             ),
#             dim=-1
#         )
#
# loss2 = positive - negative
# loss2 = torch.mean(loss2)
# print(f"__________# 1  {loss2}__________________________")
#
# # split_flat_x = torch.chunk(flat_x, chunks=28, dim=0)  # split tensor for code optimization
# split_p_mu = torch.chunk(p_mu, chunks=8, dim=0)  # split tensor for code optimization
# split_p_logvar = torch.chunk(p_logvar, chunks=8, dim=0)  # split tensor for code optimization
# split_positive = torch.chunk(positive, chunks=8, dim=0)
#
# loss = 0
# for iter in range(8):
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
#     loss += loss_
#
# loss = loss / 8
# loss = torch.mean(loss)
# print(f"__________# 2  {loss}__________________________")

b = 5
d = 10
bhw = 10
neg_samples = 3
split_x = torch.chunk(flat_x, chunks=b, dim=0)
rand_neg = torch.zeros((2, 3, 10))
for iter in range(b):
    similarity_ = torch.matmul(split_x[iter], flat_x.T)  # (bhw/b, d) * (d, bhw) -> (bhw/b, bhw)
    self_mask_ = torch.where(torch.eye(split_x[iter].shape[0], flat_x.shape[0]) == 0, 1, 10 ** 6)  # TODO check overflow
    similarity_ = similarity_ * self_mask_
    negative_index = torch.topk(similarity_, neg_samples, largest=False, dim=-1)  # (bhw/b, n)
    rand_neg_ = F.embedding(negative_index.indices, flat_x)  # ( bhw/b, n, d)
    if iter == 0:
        rand_neg = rand_neg_
    else:
        rand_neg = torch.cat([rand_neg, rand_neg_], dim=0)

