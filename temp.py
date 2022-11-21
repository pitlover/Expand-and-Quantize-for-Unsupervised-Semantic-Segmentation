import torch
import torch.nn.functional as F
z = torch.arange(2*3*5).reshape(2, 3, 5)

z_pos_dis = torch.arange(2 * 10 * 5).reshape(2, 10, 5)  # (2, 10, 5)
print(z_pos_dis)
attn = torch.tensor([[[7, 0],
                      [1, 8],
                      [3, 0]],

                     [[7, 0],
                      [4, 1],
                      [6, 8]]])
# (2, 3, 2)
print(attn, attn.shape)
batch = torch.arange(start=0, end=2 * 10, step=10).unsqueeze(-1).unsqueeze(-1)
flat_z_pos_dis = z_pos_dis.reshape(-1, z_pos_dis.shape[-1])

batch_attn = batch + attn
my_result = torch.index_select(flat_z_pos_dis, 0, batch_attn.reshape(-1))
print(my_result)
result = torch.tensor([
    [
        [[35, 36, 37, 38, 39], [0, 1, 2, 3, 4]],
        [[5, 6, 7, 8, 9], [40, 41, 42, 43, 44]],
        [[15, 16, 17, 18, 19], [0, 1, 2, 3, 4]]
    ],
    [
        [[85, 86, 87, 88, 89], [50, 51, 52, 53, 54]],
        [[70, 71, 72, 73, 74], [55, 56, 57, 58, 59]],
        [[80, 81, 82, 83, 84], [90, 91, 92, 93, 94]]

    ]
])  # (2, 3, 2, 5)
