import torch
import torch.nn as nn
from torch.nn import functional as F
import timm

model_name_to_fc_dim = {
    'efficientnet_b3': 1536
}


class TimmModel(nn.Module):
    def __init__(self, config):
        """ Based on timm
        """
        super(TimmModel, self).__init__()

        self.use_gender = config['USE_GENDER']
        self.use_age = config['USE_AGE']
        self.model_name = config['PRETRAINED_MODEL']
        self.finetuning = config['FINETUNING']

        self.base_model = timm.create_model(self.model_name, pretrained=True)

        if not self.finetuning:
            # disable fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.l0 = nn.Linear(2048, 1)
        self.l0 = nn.Linear((model_name_to_fc_dim[self.model_name] 
                             + (1 if self.use_gender else 0) + (1 if self.use_age else 0)), 
                            1)

    def trainable_params(self):
        if self.finetuning:
            return self.parameters()
        else:
            return list(self.l0.parameters())  # + list(self.l1.parameters())

    def forward(self, image, gender, age):
        batch_size, _, _, _ = image.shape

        x = self.base_model.forward_features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        if self.use_gender:
            x = torch.cat((x, gender.unsqueeze(1)), 1)
        if self.use_age:
            x = torch.cat((x, age.unsqueeze(1)), 1)

        x = self.l0(x)

        return x
