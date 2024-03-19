from utils.dataset import *
import torch
from models.NTU_Fi_model import *
from models.model import vision_model_224


def load_data_n_model(dataset_name, root):
    classes = {'UT_HAR_data':7,'NTU-Fi-HumanID':14,'NTU-Fi_HAR':6,'Widar':22}

    if dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/train_amp/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi-HumanID/test_amp/'), batch_size=64, shuffle=False)

        # print("using model: ViT")
        # # model = NTU_Fi_ViT(num_classes=num_classes)
        # model = vision_model()

        return train_loader, test_loader

    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/train_amp/'), batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=CSI_Dataset(root + 'NTU-Fi_HAR/test_amp/'), batch_size=64, shuffle=False)

        # print("using model: ViT")
        # model = NTU_Fi_ViT(num_classes=num_classes)

    elif dataset_name == 'UT_HAR_data':
        print('using dataset: UT-HAR DATA')
        data = UT_HAR_dataset(root)
        train_set = torch.utils.data.TensorDataset(data['X_train'],data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'],data['X_test']),0),torch.cat((data['y_val'],data['y_test']),0))
        train_loader = torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True, drop_last=True) # drop_last=True
        test_loader = torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False)

        
    return train_loader, test_loader


