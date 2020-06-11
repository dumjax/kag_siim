import torch
import torch.nn as nn
from torch.nn import functional as F
import timm

USE_GENDER = False
USE_AGE = False


class efficientnet_b3_mix_1(nn.Module):
    def __init__(self, finetuning):
        """ Based on timm
        """
        super(efficientnet_b3_mix_1, self).__init__()

        self.finetuning = finetuning
        self.base_model = timm.create_model('efficientnet_b3', pretrained=True)

        if not finetuning:
            # disable fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.l0 = nn.Linear(2048, 1)
        self.l0 = nn.Linear(1536, 1)

    def trainable_params(self):
        if self.finetuning:
            return self.parameters()
        else:
            return list(self.l0.parameters())  # + list(self.l1.parameters())

    def forward(self, image, gender, age):
        batch_size, _, _, _ = image.shape

        x = self.base_model.forward_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        if USE_GENDER:
            x = torch.cat((x, gender.unsqueeze(1)), 1)
        if USE_AGE:
            x = torch.cat((x, age.unsqueeze(1)), 1)

        x = self.l0(x)

        return x
