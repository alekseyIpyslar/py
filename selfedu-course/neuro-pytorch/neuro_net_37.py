import torch
import torch.nn as nn

x = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.float32)
w = torch.tensor([[[[1, 2], [3, 4]]]], dtype=torch.float32)
alg = nn.ConvTranspose2d(1, 1, 2, 2, bias=False)

st = alg.state_dict()
st['weight'] = w
alg.load_state_dict(st)

y = alg(x)
print(y)