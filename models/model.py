
import torch
import torch.nn as nn
from torchvision import transforms
# from vit_pytorch import ViT
# from pytorch_pretrained_vit import ViT


# class vision_model_256(nn.Module):
#     def __init__(self): # embsize -> 196
#         super().__init__()
#         self.transforms = transforms.Resize((256,256))
#         self.v = ViT(
#             image_size=256,
#             patch_size=32,
#             num_classes=1000,
#             dim=1024,
#             depth=6,
#             heads=16,
#             mlp_dim=2048,
#             dropout=0.1,
#             emb_dropout=0.1
#         )
#     def forward(self, x):
#         x = self.transforms(x)
#         x = self.v(x)
#
#         return x

class vision_model_224(nn.Module):
    def __init__(self): # embsize -> 196
        super().__init__()
        self.transforms = transforms.Compose([transforms.Resize((224,224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.v = ViT('B_16_imagenet1k', pretrained=True)
    def forward(self, x):
        x = self.transforms(x)
        x = self.v(x)

        return x


# img = torch.randn(1, 3, 114, 500)
# model = vision_model_224()
# print(model(img))