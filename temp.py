import torch
import torch.nn.functional as F


def jsd(p: torch.Tensor, q: torch.Tensor):
    '''

    :param p: (b, selected, dim)
    :param q: (b, hw, dim)
    :return:
    '''
    # broadcasting ...
    p = p[:, None, ...] # (b, 1, selected, dim)
    q = q[:, :, None, :] # (b, hw, 1, dim)

    m = torch.clamp((p + q) * 0.5, 1e-7, 1).log() # (b, hw, selected, dim)
    loss = (F.kl_div(m, p, reduction="batchmean") + F.kl_div(m, q, reduction="batchmean")) * 0.5

    return loss


# def jsd(p: torch.Tensor, q: torch.Tensor):
#     kldiv = torch.nn.KLDivLoss()
#
#     if torch.min(p) < 0.0 or torch.max(p) > 1.0 or torch.min(q) < 0 or torch.max(q) > 1.0:
#         raise ValueError(
#             f"min_p, max_p, min_q, max_q : {torch.min(p)}, {torch.max(p)}, {torch.min(q)}, {torch.max(q)}")
#
#     return (p[None, ...] * (torch.log(p[None, ...] - q[:, None, :])).mean(dim=-1))
# m = (0.5 * (p + q).add(1e-6)).log()
# return 0.5 * (kldiv(m, p.add(1e-6).log()) + kldiv(m, q.add(1e-6).log()))


# random_index = torch.tensor([i for i in range(2 * 3)]).reshape(2, 3)  # (2, 3)
random_index = torch.tensor([[5, 1, 0],
                             [2, 6, 3]])
print(random_index)
feature = torch.Tensor([i for i in range(2 * 9 * 5)]).reshape(2, -1, 5)  # (2, 9, 5)
flat_feature = feature.reshape(-1, 5)  # (2, 9, 5)
pos_feature = torch.Tensor([i for i in range(2 * 9 * 5, 2 * 2 * 9 * 5)]).reshape(2, 9, 5)  # (2, 9, 5)
dis_prob = torch.rand(2 * 9, 10)
pos_dis_prob = torch.rand(2 * 9, 10)
# dis_prob = torch.Tensor([i for i in range(2 * 9 * 10)]).reshape(-1, 10)  # (2, 9, 10) -> (18, 10)
# pos_dis_prob = torch.Tensor([i for i in range(2 * 9 * 10, 2 * 2 * 9 * 10)]).reshape(-1, 10)  # (2, 9, 10) -> (18, 10)

for a in range(random_index.shape[0]):
    random_index[a] += (a * 9)

# feature = torch.tensor([i for i in range(2 * 9 * 5)]).reshape(2, 9, 5)  # (2, 9, 5)

print(feature, "feautre", feature.shape)
print(dis_prob, "dis_prob", dis_prob.shape)
print(pos_dis_prob, "pos_dis", pos_dis_prob.shape)

sample_feature = F.embedding(random_index, flat_feature)  # (2, 3, 5)
sample_feature_dis = F.embedding(random_index, dis_prob)  # (2, 3, 10)
print(sample_feature.shape, sample_feature_dis.shape)

attn = torch.einsum("bsc,bdc->bsd", sample_feature, pos_feature)  # pos_feature : (2, 3, 9)
print(attn, attn.shape)
attn -= attn.mean([-1], keepdim=True)
print(attn)
attn = attn.clamp(0)  # (b, 11, 11, h, w)
mask = attn > 0
print(mask, mask.shape)

loss = jsd(sample_feature_dis, pos_dis_prob.reshape(2, -1, 10))  # (6, 10) * (18, 10)
print(loss, loss.shape)
# result = torch.tensor([
#     [[25, 26, 27, 28, 29], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
#     [[55, 56, 57, 58, 59], [75, 76, 77, 78, 79], [60, 61, 62, 63, 64]]
# ])
