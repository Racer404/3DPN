import torch
import torch.nn as nn
import torch.optim as optim


class GrowableModel(nn.Module):
    def __init__(self, init_size=4, device='cuda'):
        super().__init__()
        data = torch.randn(init_size, 3, device=device)
        self.x = nn.Parameter(data)  # trainable tensor

    def extend(self, target_idx, optimizer):
        target_idx = target_idx + 1
        needed = target_idx - self.x.shape[0]
        if needed <= 0:
            return

        # create new rows
        new_rows = torch.randn(needed, 3, device=self.x.device)

        # create a NEW parameter by concatenating old + new
        new_param = nn.Parameter(torch.cat([self.x.data, new_rows], dim=0))

        # replace the parameter inside the model
        self.x = new_param

        # replace inside optimizer (NO LOOP, very fast)
        optimizer.param_groups[0]['params'] = [self.x]

        print(f"Extended parameter to size {self.x.shape[0]}")

    def forward(self, indices):
        # indices is a tensor selecting rows of x
        return torch.index_select(self.x, 0, indices)


# ----------------------------
# Training loop
# ----------------------------

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = GrowableModel(init_size=4, device=device)
optimizer = optim.Adam([model.x], lr=0.01)
loss_fn = nn.MSELoss()

for step in range(30):
    # simulate some random "access pattern"
    idx = torch.randint(low=0, high=model.x.shape[0] + 3, size=(5,), device=device)

    # If idx exceeds current size, extend parameter
    max_idx = idx.max().item()
    if max_idx >= model.x.shape[0]:
        model.extend(max_idx, optimizer)

    # forward: pick rows
    out = model(idx)

    # target is just zeros for demo
    target = torch.zeros_like(out)

    # compute loss
    loss = loss_fn(out, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Step {step:02d}, loss = {loss.item():.6f}, size = {model.x.shape[0]}")
