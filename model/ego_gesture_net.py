import torch.nn as nn
import torch
import torchvision.models.resnet as resnet


class EgoGestureNet(nn.Module):

    def __init__(self):
        super(EgoGestureNet, self).__init__()
        rnet = resnet.resnet18(True).cuda()
        self.conv1 = rnet.conv1
        self.bn1 = rnet.bn1
        self.relu = rnet.relu
        self.maxpool = rnet.maxpool
        self.layer1 = rnet.layer1
        self.layer2 = rnet.layer2
        self.conv2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv3 = nn.Conv2d(128, 64, 3, 2, 1)
        self.avg_pool = nn.AvgPool2d((4, 7))
        self.deconv0 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv1 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(16, 8, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(8, 4, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(4, 2, 4, 2, (2, 1))
        self.lstm_feature_size = 64
        self.hidden_size = self.lstm_feature_size * 2
        self.batch_size = 1
        self.lstm = nn.LSTM(self.lstm_feature_size, self.hidden_size, 3, batch_first=True, dropout=0)
        self.linear1 = nn.Linear(self.hidden_size, 11)
        self.count = 0
        self.smax = nn.Softmax(dim=1)
        self.smax2d = nn.Softmax2d()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)

        encoded_features = self.avg_pool(x)
        encoded_features = torch.squeeze(encoded_features)
        encoded_features = torch.unsqueeze(encoded_features, 0)
        label, _ = self.lstm(encoded_features)
        label = label[:, -1, :]
        label = self.linear1(label)
        label = self.smax(label)

        x = self.deconv0(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        mask = self.smax2d(x)

        return mask, label

    def recognise_gesture(self, img_tensor):

        mask, label = self.forward(img_tensor)
        value, gesture_generated = torch.max(label, 1)
        return gesture_generated.cpu().data.numpy()
