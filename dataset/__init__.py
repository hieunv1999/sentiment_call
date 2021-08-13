import torch
from torch.functional import F
input = torch.randn(1, 5, requires_grad=True)
target = torch.randint(5, (1,), dtype=torch.int64)
print(input)
print(target)
loss = F.cross_entropy(input, target)
loss.backward()