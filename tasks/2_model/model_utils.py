from torchvision.models.detection import fasterrcnn_resnet50_fpn        

def build_model(num_classes: int, pretrained: bool = True, backbone_pretrained: bool = True):
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=backbone_pretrained, num_classes=num_classes)
    return model