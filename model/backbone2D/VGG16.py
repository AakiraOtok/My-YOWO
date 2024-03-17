import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

class VGG16Base(nn.Module):
    """
    Lấy VGG16 làm base network, tuy nhiên cần có một vài thay đổi:
    - Đầu vào ảnh là 300x300 thay vì 224x224, các comment bên dưới sẽ áp dụng cho đầu vào 300x300
    - Lớp pooling thứ 3 sử dụng ceiling mode thay vì floor mode
    - Lớp pooling thứ 5 kernel size (2, 2) -> (3, 3) và stride 2 -> 1, và padding = 1
    - Ta downsample (decimate) parameter fc6 và fc7 để tạo thành conv6 và conv7, loại bỏ hoàn toàn fc8
    """

    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(in_channels=  3, out_channels= 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(in_channels= 64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pool5   = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        # Không còn fc layers nữa, thay vào đó là conv6 và conv7
        # atrous
        self.conv6   = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
        self.conv7   = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

    def decimate(self, tensor, steps):
        assert(len(steps) == tensor.dim())
        
        for i in range(tensor.dim()):
            if steps[i] is not None:
                tensor = tensor.index_select(dim=i, index=torch.arange(start=0, end=tensor.shape[i], step=steps[i]))

        return tensor

    
    def load_pretrain(self):
        """
        load pretrain từ thư viện pytorch, decimate param lại để phù hợp với conv6 và conv7
        """

        state_dict  = self.state_dict() 
        param_names = list(state_dict.keys())

        # old version : torch.vision.models.vgg16(pretrain=True)
        # Load model theo API mới của pytorch, cụ thể hơn tại : https://pytorch.org/vision/stable/models.html
        pretrain_state_dict  = torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').state_dict()
        pretrain_param_names = list(pretrain_state_dict.keys())

        # Pretrain param name và custom param name không giống nhau, các param chỉ cùng thứ tự như trong architecture
        for idx, param_name in enumerate(param_names[:-4]): # 4 param cuối là weight và bias của conv6 và conv7, sẽ xử lí sau
            state_dict[param_name] = pretrain_state_dict[pretrain_param_names[idx]]

        # fc -> conv
        fc6_weight = pretrain_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        fc6_bias   = pretrain_state_dict['classifier.0.bias'].view(4096)

        fc7_weight = pretrain_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        fc7_bias   = pretrain_state_dict['classifier.3.bias'].view(4096)

        # downsample parameter
        state_dict['conv6.weight'] = self.decimate(fc6_weight, steps=[4, None, 3, 3])
        state_dict['conv6.bias']   = self.decimate(fc6_bias, steps=[4])

        state_dict['conv7.weight'] = self.decimate(fc7_weight, steps=[4, 4, None, None])
        state_dict['conv7.bias']   = self.decimate(fc7_bias, steps=[4])

        self.load_state_dict(state_dict)


    def forward(self, images):
        """
        :param images, tensor [N, 3, 300, 300]

        return:
        """
        out = F.relu(self.conv1_1(images)) # [N, 64, 300, 300]
        out = F.relu(self.conv1_2(out))    # [N, 64, 300, 300]
        out = self.pool1(out)              # [N, 64, 150, 150]

        out = F.relu(self.conv2_1(out))    # [N, 128, 150, 150]
        out = F.relu(self.conv2_2(out))    # [N, 128, 150, 150]
        out = self.pool2(out)              # [N, 128, 75, 75]

        out = F.relu(self.conv3_1(out))    # [N, 256, 75, 75]
        out = F.relu(self.conv3_2(out))    # [N, 256, 75, 75]
        out = F.relu(self.conv3_3(out))    # [N, 256, 75, 75]
        out = self.pool3(out)              # [N, 256, 38, 38] không phải [N, 256, 37, 37] do ceiling mode = True

        out = F.relu(self.conv4_1(out))    # [N, 512, 38, 38]
        out = F.relu(self.conv4_2(out))    # [N, 512, 38, 38]
        out = F.relu(self.conv4_3(out))    # [N, 512, 38, 38]
        conv4_3_feats = out                # [N, 512, 38, 38]
        out = self.pool4(out)              # [N, 512, 19, 19]

        out = F.relu(self.conv5_1(out))    # [N, 512, 19, 19]
        out = F.relu(self.conv5_2(out))    # [N, 512, 19, 19]
        out = F.relu(self.conv5_3(out))    # [N, 512, 19, 19]
        out = self.pool5(out)              # [N, 512, 19, 19], layer pooling này không làm thay đổi size features map

        out = F.relu(self.conv6(out))      # [N, 1024, 19, 19]

        conv7_feats = F.relu(self.conv7(out)) # [N, 1024, 19, 19]

        return conv4_3_feats, conv7_feats