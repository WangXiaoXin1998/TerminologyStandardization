
import torch

xx1 = torch.rand(1, 10)
print(xx1)

yy1 = torch.rand(4, 10)
print(yy1)

dis1 = torch.pairwise_distance(xx1, yy1, p=1)
print(dis1)

ang = torch.cosine_similarity(xx1, yy1, dim=1)
print(ang)

xx = torch.rand(1, 5)
print(xx)
xx = torch.argmax(xx, dim=1)
print(xx)

