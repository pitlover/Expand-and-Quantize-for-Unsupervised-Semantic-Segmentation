import torch

b = 32
flat_x = torch.rand(1568, 512)
split_x = torch.chunk(flat_x, chunks=b, dim=0)


p_mu = torch.randint(100, (1568, 512))
p_logvar = torch.randint(100, (1568, 512))

positive = -0.5 * torch.sum(
    torch.square(flat_x - p_mu) / torch.exp(p_logvar), dim=-1  # (bhw, d) -> (bhw)
)
negative = -0.5 * torch.mean(
    torch.sum(
        torch.square(flat_x.unsqueeze(0) - p_mu.unsqueeze(1)) /
        torch.exp(p_logvar.unsqueeze(1)),
        dim=-1
    ),
    dim=-1
)

loss2 = positive - negative
loss2 = torch.mean(loss2)
print(f"__________# 1  {loss2}__________________________")

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

# split_p_mu = torch.chunk(p_mu, chunks=2 * 28, dim=0)  # split tensor for code optimization
# split_p_logvar = torch.chunk(p_logvar, chunks=2 * 28, dim=0)  # split tensor for code optimization
# split_positive = torch.chunk(positive, chunks=2 * 28, dim=0)
#
# loss = 0
# for iter in range(2 * 28):
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
#     loss += torch.mean(loss_).item()
#
# loss = loss / (2 * 28)
#
# print(f"__________# 3  {loss}__________________________")
