import torch
import torch.nn as nn
from torch.nn import functional as F
import timm

model_name_to_fc_dim = {
    'efficientnet_b3': 1536,
    'mixnet_m': 1536,
    'resnext50_32x4d': 2048,
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
        self.nonlinearity = config['NONLINEARITY']

        self.base_model = timm.create_model(self.model_name, pretrained=True)

        if not self.finetuning:
            # disable fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False

        # FC layers:
        input_size = model_name_to_fc_dim[self.model_name] + (1 if self.use_gender else 0) + (1 if self.use_age else 0)
        hidden_sizes = config['HIDDEN_SIZES']
        all_sizes = [input_size] + hidden_sizes + [1]
        self.fc_layers = nn.ModuleList([nn.Linear(all_sizes[i], all_sizes[i+1]) for i in range(len(all_sizes)-1)])

    def trainable_params(self):
        if self.finetuning:
            return self.parameters()
        else:
            params = []
            for fc_layer in self.fc_layers:
                params += list(fc_layer.parameters())
            return params

    def forward(self, image, gender, age):
        batch_size, _, _, _ = image.shape

        x = self.base_model.forward_features(image)

        # TODO: adapt this
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        if self.use_gender:
            x = torch.cat((x, gender.unsqueeze(1)), 1)
        if self.use_age:
            x = torch.cat((x, age.unsqueeze(1)), 1)

        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                x = self.nonlinearity(x)

        # Output a probability
        x = torch.sigmoid(x)

        return x
