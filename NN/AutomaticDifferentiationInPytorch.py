import torch
from torch.autograd import grad
import torch.nn.functional as F

#Define the input features x, weights w and bias b

x = torch.tensor([3.])
#by setting require_grad=True, pytorch will create a
# computation graph behind the scene,
# it basically means that we'll need the gradient for
# this function
w = torch.tensor([2.], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)
a = F.relu(x*w + b)

#Here, if we dont provide retain_graph = True, at every iteration the existing
# graph will be descarded and recreated -- this is good for memory
grad(a, w, retain_graph=True)