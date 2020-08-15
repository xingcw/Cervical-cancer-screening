import torch
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utils import utils
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from utils.engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from Dataloader import ISBI14Dataset, get_transform, Cervix93Dataset, cer_aug_dataset


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


# # load a model pre-trained on COCO
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
#
# # replace the classifier with a new one, that has num_classes which is user-defined
# # 3 classes('N', 'L', 'H') + 1 background
# # 2 classes('Cytoplasm', 'Nuclei') + 1 background
# num_classes = 2
#
# # get number of input features for the classifier
# in_features = model.roi_heads.box_predictor.cls_score.in_features
#
# # replace the pre-trained head with a new one
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#
# # load a pre-trained model for classification and return only the features
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
#
# # FasterRCNN needs to know the number of output channels in a backbone.
# # For mobilenet_v2, it's 1280. So we need to add it here
# backbone.out_channels = 1280
#
# # let's make the RPN generate 5 x 3 anchors per spatial
# # location, with 5 different sizes and 3 different aspect
# # ratios. We have a Tuple[Tuple[int]] because each feature
# # map could potentially have different sizes and aspect ratios
# anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
#                                    aspect_ratios=((0.5, 1.0, 2.0),))
#
# # let's define what are the feature maps that we will use to perform the region of
# # interest cropping, as well as the size of the crop after rescaling.
# # if your backbone returns a Tensor, featmap_names is expected to
# # be [0]. More generally, the backbone should return an OrderedDict[Tensor],
# # and in featmap_names you can choose which feature maps to use.
# roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
#                                                 output_size=7,
#                                                 sampling_ratio=2)
#
# # put the pieces together inside a FasterRCNN model
# model = FasterRCNN(backbone,
#                    num_classes,
#                    rpn_anchor_generator=anchor_generator,
#                    box_roi_pool=roi_pooler)

def train():
    # use isbi14 dataset or the fused dataset
    dataset = ISBI14Dataset('dataset', get_transform(train=True))
    dataset_test = ISBI14Dataset('dataset', get_transform(train=False))

    # use the CERVIX93 dataset
    # dataset = cer_aug_dataset(r'E:\大四下\毕设\数据集\CERVIX93_Seg.&Clss\cytology_dataset-master\avaliable_imgs',
    # #                           get_transform(train=True))
    # # dataset_test = cer_aug_dataset(r'E:\大四下\毕设\数据集\CERVIX93_Seg.&Clss\cytology_dataset-master\avaliable_imgs',
    # #                                get_transform(train=False))

    # split the dataset into train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    # the codes following are used to detect the illegal data, so the inputs are sorted
    # indices = torch.arange(0, len(dataset)).tolist()
    # train_indices = indices[0:101]
    # train_indices.extend(indices[191:])
    dataset = torch.utils.data.Subset(dataset, indices[:800])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[800:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True,  # num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False,  # num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # the dataset has two classes only - background and cytoplasm
    # or 3 classes - background, cytoplasm and nuclei
    num_classes = 2

    # get the model using the helper function
    model = get_instance_segmentation_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.008,
                                momentum=0.9, weight_decay=0.0005)

    # the learning rate scheduler decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 5
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # save the trained parameters of the model
    path = r'E:\大四下\毕设\第四周_MaskRCNN\trained_model\model_data_seg_fraction_6_2.pt'
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    train()