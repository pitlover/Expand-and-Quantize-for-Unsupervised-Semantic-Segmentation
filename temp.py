import torch

b = 28

flat_x = torch.randint(100, (56, 32))
p_mu = torch.randint(100, (56, 32))
p_logvar = torch.randint(100, (56, 32))

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

positive = -0.5 * torch.sum(
    torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1  # (bhw, d) -> (bhw)
)
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
    loss_ = split_positive[iter] - negative  # bhw/b
    loss += loss_

loss = loss / b

del positive, negative
del split_positive, split_p_mu, split_p_logvar

print(f"loss1 = {torch.mean(loss)}")

positive = -0.5 * torch.sum(
    torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1
)

negative = -0.5 * torch.mean(
    torch.sum(
        torch.square(flat_x.unsqueeze(0) - p_mu.unsqueeze(1)) /
        torch.exp(p_logvar.unsqueeze(1)),
        dim=-1
    ),
    dim=-1
)

loss = positive - negative

print(f"loss2 = {torch.mean(loss)}")
