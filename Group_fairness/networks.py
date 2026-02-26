import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class ClassNNet(nn.Module):
    def __init__(self, num_features, num_classes, device, use_dropout=False):
        super(ClassNNet, self).__init__()

        self.device = device

        self.use_dropout = use_dropout

        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, num_classes)

        self.z_dim = 256 + 128

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)

    def forward(self, x, extract_features=False):
        x = self.layer_1(x)
        x = self.relu(x)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm1(x)

        z2 = self.layer_2(x)
        x = self.relu(z2)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm2(x)

        z3 = self.layer_3(x)
        x = self.relu(z3)
        if self.use_dropout:
          x = self.dropout(x)
          x = self.batchnorm3(x)
        x = self.layer_4(x)
        x = self.relu(x)

        if self.use_dropout:
            x = self.dropout(x)
            x = self.batchnorm4(x)

        x = self.layer_5(x)

        if extract_features:
          return x, torch.cat([z2,z3],1)
        else:
          return x

    def predict_prob(self, inputs):
        """
        Predict probabilities given any input data
        """
        self.eval()
        get_prob = nn.Softmax(dim = 1)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            x = self(inputs)
            prob = get_prob(x).cpu().numpy()
        return prob


class FaReG(nn.Module):
    def __init__(self, num_features, delta, device, hidden_dim=64, latent_dim=32):
        super(FaReG, self).__init__()
        self.device = device
        self.delta = delta
        self.v = -1

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mu_layer = nn.Linear(latent_dim, 4)
        self.logvar_layer = nn.Linear(latent_dim, 4)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = self.mu_layer(x), self.logvar_layer(x)
        return mu, logvar

    def reparameterization(self, mu, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mu + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def update_v(self, prob): 

        if torch.mean(prob) >= self.delta:
          self.v = 0
          return

        p_sum = torch.sum(prob)
        pi = prob.argsort(dim=0, descending=True)
        p_sort = torch.gather(prob, dim=0, index=pi)
        p_cumsum = p_sort.cumsum(dim=0)

        k_max = 0  
        n = prob.shape[0]
        v_final = 2 * (self.delta * n - p_sum) / n

        for i in range(n - 1):
          k = i + 1
          v_i = 2 * (self.delta * n - k - (p_sum - p_cumsum[i])) / (n - k)

          if v_i >= 2 * (1 - p_sort[i]) and v_i < 2 * (1 - p_sort[i + 1]): 
            k_max = max(k, k_max) 
            v_final = v_i

        # print('k_max: ', k_max)
        self.v = v_final
        return

    def project(self, prob):
        if torch.mean(prob) >= self.delta:
            return prob
        else:
            return torch.min(prob + self.v / 2, torch.ones_like(prob))    


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        prob = self.decode(z)

        # Project
        self.update_v(prob)
        prob = self.project(prob)

        return prob, mu, logvar
    
    def get_z(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        return z
    
    def get_decoder(self, z):
        prob = self.decoder(z)

        # Project
        self.update_v(prob)
        prob = self.project(prob)
        
        return prob
        
   
class FaReG_Rec(nn.Module):
    def __init__(self, num_features, device, hidden_dim=64, latent_dim=32):
        super(FaReG_Rec, self).__init__()
        self.device = device

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_features)
            )

    def decode(self, x):
        return self.decoder(x)


    def forward(self, x):
        hat_x = self.decode(x)

        return hat_x
    






# class FaReG(nn.Module):
#     def __init__(self, num_features, delta, device, use_dropout=False):
#         super(FaReG, self).__init__()

#         self.device = device
#         self.use_dropout = use_dropout

#         # self.input_layer = nn.Parameter(torch.ones(size=(1, num_features)), requires_grad=True)
#         self.input_layer = nn.Linear(num_features, 128)
#         self.hidden_layer_1 = nn.Linear(128, 64)
#         self.hidden_layer_2 = nn.Linear(64, 1)
#         # self.hidden_layer_3 = nn.Linear(64, 1)

#         # self.z_dim = 256 + 128

#         self.relu = nn.ReLU()
#         # self.dropout = nn.Dropout(p=0.2)
#         # self.batchnorm1 = nn.BatchNorm1d(128)
#         # self.batchnorm2 = nn.BatchNorm1d(64)

#         self.delta = delta
#         self.v = -1


#     def update_v(self, prob): 

#         if torch.mean(prob) >= self.delta:
#           self.v = 0
#           return

#         p_sum = torch.sum(prob)
#         pi = prob.argsort(dim=0, descending=True)
#         p_sort = torch.gather(prob, dim=0, index=pi)
#         p_cumsum = p_sort.cumsum(dim=0)

#         k_max = 0  
#         n = prob.shape[0]
#         v_final = 2 * (self.delta * n - p_sum) / n

#         for i in range(n - 1):
#           k = i + 1
#           v_i = 2 * (self.delta * n - k - (p_sum - p_cumsum[i])) / (n - k)

#           if v_i >= 2 * (1 - p_sort[i]) and v_i < 2 * (1 - p_sort[i + 1]): 
#             k_max = max(k, k_max) 
#             v_final = v_i

#         # print('k_max: ', k_max)
#         self.v = v_final
#         return

#     def project(self, prob):
#         if torch.mean(prob) >= self.delta:
#             return prob
#         else:
#             return torch.min(prob + self.v / 2, torch.ones_like(prob))


#     def forward(self, x):
#         # x = x * torch.tile(self.input_layer, (x.shape[0], 1))
#         # print(self.input_layer.detach().cpu().data)
#         x = self.input_layer(x)

#         z1 = self.hidden_layer_1(x)
#         x = self.relu(z1)

#         x = self.hidden_layer_2(x)
#         # x = self.relu(z2)

#         # x = self.hidden_layer_3(x)
#         prob = torch.sigmoid(x)
        
#         # Project
#         self.update_v(prob)
#         prob = self.project(prob)

#         return prob