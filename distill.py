import numpy as np
import torch
import torch.nn as nn
import argparse
from util import load_data_n_model
import torch.backends.cudnn as cudnn
from easydict import EasyDict
from models import vitmodel as vit_image
from timm.models.layers import trunc_normal_
from models.LeNet import *
from models.loss import *
from models.models import *

def train_all(teacher_model, student_model,tensor_loader_1, tensor_loader_2, num_epochs, learning_rate, criterion, device, RKD_loss=None, RKD_para=0):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        iter_list = [iter(tensor_loader_1), iter(tensor_loader_2)]
        for loader in iter_list:
            for data in loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.type(torch.LongTensor)
                optimizer.zero_grad()
                outputs_stu = student_model(inputs)
                outputs_stu = outputs_stu.to(device)
                outputs_stu = outputs_stu.type(torch.FloatTensor)

                outputs_tea = teacher_model(inputs)
                outputs_tea = outputs_tea.to(device)
                outputs_tea = outputs_tea.type(torch.FloatTensor)
                loss = criterion(outputs_stu, outputs_tea)
                if RKD_para:
                    loss += RKD_loss(outputs_stu,outputs_tea, RKD_para)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                predict_y = torch.argmax(outputs_stu, dim=1).to(device)
                epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / (len(iter_list[0])+len(iter_list[1]))
        epoch_accuracy = epoch_accuracy / (len(iter_list[0])+len(iter_list[1]))
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
    return


def train(teacher_model, student_model,tensor_loader, num_epochs, learning_rate, criterion, device, RKD_loss=None, RKD_para=0):
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        student_model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tensor_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()
            outputs_stu = student_model(inputs)
            outputs_stu = outputs_stu.to(device)
            outputs_stu = outputs_stu.type(torch.FloatTensor)

            outputs_tea = teacher_model(inputs)
            outputs_tea = outputs_tea.to(device)
            outputs_tea = outputs_tea.type(torch.FloatTensor)
            loss = criterion(outputs_stu, outputs_tea)
            if RKD_para:
                loss += RKD_loss(outputs_stu,outputs_tea, RKD_para)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)
            predict_y = torch.argmax(outputs_stu, dim=1).to(device)
            epoch_accuracy += (predict_y == labels.to(device)).sum().item() / labels.size(0)
        epoch_loss = epoch_loss / len(tensor_loader.dataset)
        epoch_accuracy = epoch_accuracy / len(tensor_loader)
        print('Epoch:{}, Accuracy:{:.4f},Loss:{:.9f}'.format(epoch + 1, float(epoch_accuracy), float(epoch_loss)))
    return


def test(model, tensor_loader, criterion, device):
    model.eval()
    test_acc = 0
    test_loss = 0
    for data in tensor_loader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.LongTensor)

        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)

        loss = criterion(outputs, labels)
        predict_y = torch.argmax(outputs, dim=1).to(device)
        accuracy = (predict_y == labels.to(device)).sum().item() / labels.size(0)
        test_acc += accuracy
        test_loss += loss.item() * inputs.size(0)
    test_acc = test_acc / len(tensor_loader)
    test_loss = test_loss / len(tensor_loader.dataset)
    print("validation accuracy:{:.4f}, loss:{:.5f}".format(float(test_acc), float(test_loss)))
    return



def load_vit_model(args):
    tuning_config = EasyDict(
        # AdaptFormer
        ffn_adapt=args.ffn_adapt,
        ffn_option="parallel",
        ffn_adapter_layernorm_option="none",
        ffn_adapter_init_option="lora",
        ffn_adapter_scalar="0.1",  # 什么意思
        ffn_num=args.ffn_num,
        d_model=768,
        # VPT related
        vpt_on=args.vpt,
        vpt_num=args.vpt_num,
    )
    model = vit_image.__dict__[args.model](
            num_classes=args.nb_classes,
            global_pool=args.global_pool,
            drop_path_rate=args.drop_path,
            tuning_config=tuning_config,
            dataset=args.dataset,
        )
    if args.distill:
        checkpoint = torch.load(args.distill, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(768, affine=False, eps=1e-6),
                                         model.head)

        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


    # freeze all but the head, remaining the adapter to update also.
    for name, p in model.named_parameters():
        p.requires_grad = False

    return model



def main():
    # 设置数据集的地址
    root = '/home/sun/zhoujg/csi_mixup/csimix/Data/'

    # 超参数以及额外信息的添加
    parser = argparse.ArgumentParser('Vision Transformer Adapter for CSI')
    parser.add_argument('--dataset', choices=['NTU-Fi-HumanID', 'NTU-Fi_HAR', 'UT_HAR_data'], default='NTU-Fi-HumanID')
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--seed', default=53, type=int)
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0)')
    # AdaptFormer related parameters
    parser.add_argument('--ffn_adapt', default=True, action='store_true', help='whether activate AdaptFormer')
    parser.add_argument('--ffn_num', default=64, type=int, help='bottleneck middle dimension')
    parser.add_argument('--vpt', default=False, action='store_true', help='whether activate VPT')
    parser.add_argument('--vpt_num', default=1, type=int, help='number of VPT prompts')
    parser.add_argument('--fulltune', default=False, action='store_true', help='full finetune model')
    parser.add_argument('--finetune', default='./pretrain_weights/mae_pretrain_vit_base.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--distill', default='./weights/teacher.pth',
                        help='distill teacher')
    parser.add_argument('--rkd_loss', default=0.5,
                        help='relational knowledge distill loss parameter')
    parser.add_argument('--global_pool', default= False, action='store_true')
    parser.add_argument('--train_all', default=False)
    # read_informatino
    args = parser.parse_args()
    train_epoch = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # random settings
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load model
    model = load_vit_model(args)
    model.to(device)
    if args.dataset=='UT_HAR_data':
        stu_model = UT_HAR_LeNet()
    else:
        # stu_model = NTU_Fi_LeNet(args.nb_classes)
        stu_model = NTU_Fi_ResNet18(args.nb_classes)
    stu_model.to(device)

    # load data
    cudnn.benchmark = True
    train_loader, test_loader = load_data_n_model(args.dataset, root)



    # config optimizer
    criterion_1 = nn.MSELoss()
    criterion_2 = nn.CrossEntropyLoss()
    if args.rkd_loss:
        RKD_loss = RKdAngle()
        print('RKD_loss: {}'.format(args.rkd_loss))

    if not args.train_all:
        # train_loader = zip(train_loader, test_loader)

        train(
            teacher_model=model,
            student_model=stu_model,
            tensor_loader=train_loader,
            num_epochs=train_epoch,
            learning_rate=1e-3,
            criterion=criterion_1,
            device=device,
            RKD_loss=RKD_loss if args.rkd_loss else None,
            RKD_para=args.rkd_loss
        )

        test(
            model=model,
            tensor_loader=test_loader,
            criterion=criterion_2,
            device=device
        )

    else:
        train_all(
            teacher_model=model,
            student_model=stu_model,
            tensor_loader_1=train_loader,
            tensor_loader_2=test_loader,
            num_epochs=train_epoch,
            learning_rate=1e-3,
            criterion=criterion_1,
            device=device,
            RKD_loss=RKD_loss if args.rkd_loss else None,
            RKD_para=args.rkd_loss
        )

    torch.save(stu_model.state_dict(),'./weights/student.pth')
    return

if __name__ == "__main__":
    main()
