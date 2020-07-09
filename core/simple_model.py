import torch as torch
import torch.nn as nn
from . import resnet_backbone

class IERNet(nn.Module):

    """Image Entropy Reduction Network"""

    def __init__(self,num_classes):

        super(IERNet,self).__init__()

        # feature extraction backbone
        self.backbone_extractor = resnet_backbone.resnet18(pretrained=True)
        self.feature_channels = 512

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # regress component
        self.entropy_regression = EntRegressComponent(self.feature_channels)

        # attention weights
        self.attention1_weights_generation = AttentionWeightGenerationComponent(in_channels=self.feature_channels)

        # classification
        self.backbone_classification = ClassificationComponent(in_features=self.feature_channels,num_classes=num_classes)
        self.attention1_classification = ClassificationComponent(in_features=self.feature_channels,num_classes=num_classes)
        self.fusion_classification = ClassificationComponent(in_features= self.feature_channels * 2, num_classes=num_classes)

    def forward(self, x, training):

        # extract feature
        backbone_extracted_feature = self.backbone_extractor(x)

        # get attention weights
        attention1_weights = self.attention1_weights_generation(backbone_extracted_feature)

        # # attention feature maps
        attention1_maps = backbone_extracted_feature * attention1_weights

        # average pool
        attention1_feature_vector = torch.flatten(self.avgpool(attention1_maps),1)

        # calculate
        extracted_feature_entropy = 0
        attention1_maps_entropy = 0
        feature_raw_cls_logits = 0
        attention1_cls_logits = 0

        extracted_feature_vector = torch.flatten(self.avgpool(backbone_extracted_feature), 1)

        if training:
            extracted_feature_entropy = self.entropy_regression(backbone_extracted_feature)
            attention1_maps_entropy = self.entropy_regression(attention1_maps)
            feature_raw_cls_logits = self.backbone_classification(extracted_feature_vector)
            attention1_cls_logits = self.attention1_classification(attention1_feature_vector)

        # fusion_vector = torch.flatten(self.avgpool(backbone_extracted_feature + attention1_maps), 1)
        fusion_vector = torch.cat((extracted_feature_vector,attention1_feature_vector),dim=1)
        fusion_cls_logits = self.fusion_classification(fusion_vector)
        # fusion_cls_logits = self.attention1_classification(attention1_feature_vector)
        return fusion_cls_logits,feature_raw_cls_logits,attention1_cls_logits,extracted_feature_entropy,attention1_maps_entropy

class EntRegressComponent(nn.Module):

    def __init__(self,in_channels):
        super(EntRegressComponent, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=9,kernel_size=3,stride=1,padding=1,bias=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(in_features = 9 * 7 * 7, out_features=3)

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(torch.flatten(x, 1))
        regression_value = self.relu(self.fc(x))
        return regression_value

class AttentionWeightGenerationComponent(nn.Module):

    def __init__(self,in_channels):
        super(AttentionWeightGenerationComponent, self).__init__()
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels//16)
        self.fc2 = nn.Linear(in_features=in_channels//16, out_features=in_channels)

    def forward(self,x):
        batch,channels = x.size()[0],x.size()[1]
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.relu(self.fc1(x))
        weights = self.relu(self.fc2(x))
        weights = weights.view(batch,channels,1,1)
        return weights

class ClassificationComponent(nn.Module):
    """
    num_classes is huge so it needs to be compressed
    """
    def __init__(self,in_features, num_classes):
        super(ClassificationComponent, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features, out_features=num_classes)

    def forward(self,x):
        x = self.dropout(x)
        output_digits = self.fc(x)
        return output_digits

if __name__ == "__main__":

    net = IERNet(num_classes=200)
    print(net)
