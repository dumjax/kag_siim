import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
from efficientnet_pytorch import EfficientNet

model_name_to_fc_dim = {
    'efficientnet_b3': 1536,
    'tf_efficientnet_b3': 1536,
    'mixnet_m': 1536,
    'mixnet_xl': 1536,
    'resnext50_32x4d': 2048,
    'efficientnet_b2a': 1408,
    'tf_efficientnet_b4_ns': 1792,
    'efficientnet-b4': 1792
}

def get_dim(name):
    if name in model_name_to_fc_dim:
        return model_name_to_fc_dim[name]
    else:
        return 2048


class TimmModel(nn.Module):
    def __init__(self, config):
        """ Based on timm
        """
        super(TimmModel, self).__init__()

        self.use_gender = config['USE_GENDER']
        self.use_age = config['USE_AGE']
        self.use_sites = config['USE_SITES']
        self.model_name = config['PRETRAINED_MODEL']
        self.finetuning = config['FINETUNING']
        self.nonlinearity = config['NONLINEARITY']

        self.base_model = timm.create_model(self.model_name, pretrained=True)

        if not self.finetuning:
            # disable fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False

        # FC layers:
        input_size = (get_dim(self.model_name) 
                      + (1 if self.use_gender else 0) 
                      + (1 if self.use_age else 0)
                      + (6 if self.use_sites else 0))

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

    def forward(self, image, gender, age, sites):
        batch_size, _, _, _ = image.shape

        x = self.base_model.forward_features(image)

        # TODO: adapt this
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        if self.use_gender:
            x = torch.cat((x, gender.unsqueeze(1)), 1)
        if self.use_age:
            x = torch.cat((x, age.unsqueeze(1)), 1)
        if self.use_sites:
            x = torch.cat((x, sites), 1)

        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                x = self.nonlinearity(x)

        # Output a probability
        x = torch.sigmoid(x)

        return x


class TimmModel2(nn.Module):
    def __init__(self, config):
        """ Based on timm
        """
        super(TimmModel2, self).__init__()

        self.use_gender = config['USE_GENDER']
        self.use_age = config['USE_AGE']
        self.use_sites = config['USE_SITES']
        self.model_name = config['PRETRAINED_MODEL']
        self.finetuning = config['FINETUNING']
        self.nonlinearity = config['NONLINEARITY']

        self.base_model = timm.create_model(self.model_name, pretrained=True)

        if not self.finetuning:
            # disable fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False

        # FC layers:
        input_size = (get_dim(self.model_name)
                      + (1 if self.use_gender else 0)
                      + (1 if self.use_age else 0)
                      + (6 if self.use_sites else 0))

        hidden_sizes = config['HIDDEN_SIZES']
        all_sizes = [input_size] + hidden_sizes + [1]
        #self.fc_layers = nn.ModuleList([nn.Linear(all_sizes[i], all_sizes[i+1]) for i in range(len(all_sizes)-1)])
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            #nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_sizes[0], 10),
            #nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(10, 1)
        )

    def trainable_params(self):
        if self.finetuning:
            return self.parameters()
        else:
            params = []
            for fc_layer in self.fc_layers:
                params += list(fc_layer.parameters())
            return params

    def forward(self, image, gender, age, sites):
        batch_size, _, _, _ = image.shape

        x = self.base_model.forward_features(image)

        # TODO: adapt this
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        if self.use_gender:
            x = torch.cat((x, gender.unsqueeze(1)), 1)
        if self.use_age:
            x = torch.cat((x, age.unsqueeze(1)), 1)
        if self.use_sites:
            x = torch.cat((x, sites), 1)

        x = self.fc_layers(x)
        # for i, fc_layer in enumerate(self.fc_layers):
        #     x = fc_layer(x)
        #     if i < len(self.fc_layers) - 1:
        #         x = self.nonlinearity(x)

        # Output a probability
        x = torch.sigmoid(x)

        return x


class EfficientNetMix(nn.Module):
    def __init__(self, config):
        """ Based on efficientnet_pytorch
        """
        super(EfficientNetMix, self).__init__()

        self.use_gender = config['USE_GENDER']
        self.use_age = config['USE_AGE']
        self.use_sites = config['USE_SITES']
        self.arch_name = config['PRETRAINED_MODEL']
        self.finetuning = config['FINETUNING']

        self.arch = EfficientNet.from_pretrained(self.arch_name)

        if not self.finetuning:
            # disable fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False

        # FC layers:
        arch_out_size = get_dim(self.arch_name)
        # self.arch._fc = nn.Linear(in_features=arch_out_size, out_features=500, bias=True)

        meta_in_size = (0
                        + (1 if self.use_gender else 0)
                        + (1 if self.use_age else 0)
                        + (6 if self.use_sites else 0))

        self.fc_meta = nn.Sequential(nn.Linear(meta_in_size, 20),
                                       nn.BatchNorm1d(20),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(20, 10),
                                       nn.BatchNorm1d(10),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2))
        self.output = nn.Linear(arch_out_size + 10, 1)

    def trainable_params(self):
        if self.finetuning:
            return self.parameters()
        else:
            params = []
            for fc_layer in self.fc_layers:
                params += list(fc_layer.parameters())
            return params

    def forward(self, image, gender, age, sites):
        batch_size, _, _, _ = image.shape

        meta = []
        if self.use_gender:
            meta.append(gender.unsqueeze(1))
        if self.use_age:
            meta.append(age.unsqueeze(1))
        if self.use_sites:
            meta.append(sites)

        meta = torch.cat(meta, dim=1)

        meta_features = self.fc_meta(meta)
        cnn_features = self.arch.extract_features(image)
        cnn_features = F.adaptive_avg_pool2d(cnn_features, 1).reshape(batch_size, -1)
        features = torch.cat((cnn_features, meta_features), dim=1)

        x = self.output(features)
        # Output a probability
        x = torch.sigmoid(x)

        return x
