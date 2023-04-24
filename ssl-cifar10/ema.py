# Maintain a running average of the model parameters, 
# which is computed using an exponential moving average (EMA). 
# The EMA gives more weight to recent values (if decay <0.5), 
# so the final parameters are influenced more by the recent updates. 
# When reporting the performance of the model, 
# they use this EMA version of the model's parameters rather than the latest parameters.

import copy
import torch

class EMA:
    def __init__(self, model, decay, device):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.ema_model.to(device)
        # All the following code is from, my previous EMA was doing weird thing?
        # https://github.com/kekmodel/FixMatch-pytorch/blob/98cedd6ffca4813fe6d744f695bee52beaf0faf7/models/ema.py#L6
        self.ema_has_module = hasattr(self.ema_model, 'module')
        self.param_keys = [k for k, _ in self.ema_model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema_model.named_buffers()]
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])

