# Maintain a running average of the model parameters, 
# which is computed using an exponential moving average (EMA). 
# The EMA gives more weight to recent values, 
# so the final parameters are influenced more by the recent updates. 
# When reporting the performance of the model, 
# they use this EMA version of the model's parameters rather than the latest parameters.

import copy

class EMA:
    def __init__(self, model, decay, device):
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.ema_model.to(device)

    def update(self, model):
        for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
            ema_param.data.copy_(self.decay * ema_param.data + (1.0 - self.decay) * model_param.data)
