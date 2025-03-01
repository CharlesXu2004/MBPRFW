import numpy as np
import torch

a = np.eye(640,dtype=float)
a = torch.from_numpy(np.float32(a))
print(a)


