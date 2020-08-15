import os
import skimage
import h5py
import scipy
import json
import uuid
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from PIL import Image
from skimage import io, measure, color
from labelme import utils


def showImgInfo(img_path):
    img = io.imread(img_path)
    plt.imshow(img)
    plt.show()
    plt.imshow(img, plt.cm.gray)
    plt.show()

    # show image's information
    print('the type of the picture:', type(img))
    print('the shape of the picture:', img.shape)
    print('the height of the picture:', img.shape[0])
    print('the width of the picture:', img.shape[1])
    print('the number of channels of the picture:', img.shape[1])
    print('total pixels of the picture:', img.size)
    print('the maximal pixel value of the picture:', img.max())
    print('the minimal pixel value of the picture:', img.min())
    print('the mean pixel value of the picture:', img.mean())


def showMultipleImgs(imgs, num_rows, num_cols, scale=5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    if num_rows == 1:
        for j in range(num_cols):
            axes[j].imshow(imgs[j])
            axes[j].axes.get_xaxis().set_visible(False)
            axes[j].axes.get_yaxis().set_visible(False)
    elif num_cols == 1:
        for i in range(num_rows):
            axes[i].imshow(imgs[i])
            axes[i].axes.get_xaxis().set_visible(False)
            axes[i].axes.get_yaxis().set_visible(False)
    else:
        for i in range(num_rows):
            for j in range(num_cols):
                axes[i][j].imshow(imgs[i * num_cols + j])
                axes[i][j].axes.get_xaxis().set_visible(False)
                axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


# for the dataset ISBI14, the five mat files are of different version,
# trainset.mat is opened with loadmat, and the other three with h5py.

def h5py_mask(root, data_mat, data_type='train', cell_part='Cytoplasm'):
    global cytoPath, nucleiPath
    GT = data_mat[data_type + '_' + cell_part]
    if cell_part == 'Cytoplasm':
        for i in range(GT.size):
            cytoPath = os.path.join(root, data_type + 'Masks', data_type + 'Masks_' + str(i))
            os.makedirs(cytoPath, exist_ok=True)

        for i in range(GT.size):
            for j in range(data_mat[GT[0][i]].size):
                skimage.io.imsave(os.path.join(cytoPath, data_type + cell_part + '_GT_' + str(j) + '.png'),
                                  data_mat[data_mat[GT[0][i]][0][j]][:])
    elif cell_part == 'Nuclei':
        nucleiPath = os.path.join(root, data_type + cell_part)
        os.makedirs(nucleiPath, exist_ok=True)
        for i in range(GT.size):
            skimage.io.imsave(os.path.join(nucleiPath, data_type + cell_part + '_' + str(i) + '.png'),
                              data_mat[GT[0][i]][:])
    return


def loadmat_mask(root, data_mat, data_type='train', cell_part='Cytoplasm'):
    GT = data_mat[data_type + '_' + cell_part]
    if cell_part == 'Cytoplasm':
        for i in range(len(GT)):
            os.makedirs(root + '\\' + data_type + 'Masks\\' + data_type + 'Masks_' + str(i), exist_ok=True)
        for i in range(len(GT)):
            for j in range(len(GT[i][0])):
                skimage.io.imsave(root + '\\' + data_type + 'Masks\\' + data_type + 'Masks_' + str(i) +
                                  '\\' + data_type + cell_part + '_GT_' + str(j) + '.png', GT[i][0][j][0])
    elif cell_part == 'Nuclei':
        os.makedirs(root + '\\' + data_type + cell_part, exist_ok=True)
        for i in range(len(GT)):
            skimage.io.imsave(
                root + '\\' + data_type + cell_part + '\\' + data_type + cell_part + '_' + str(i) + '.png', GT[i][0])
    return


def h5py_images(root, data_mat, data_type='train'):
    img_set = data_mat[data_type + 'set']
    os.makedirs(root + '\\' + data_type + 'Images', exist_ok=True)
    for i in range(img_set.size):
        skimage.io.imsave(root + '\\' + data_type + 'Images\\' + data_type + 'Imgs_' + str(i) + '.png',
                          data_mat[img_set[0][i]][:])
    return


def h5py_readGT(root, data_mat, cell_part='Cytoplasm', data_type='train'):
    GT = data_mat[data_type + '_' + cell_part]
    for i in range(GT.size):
        os.makedirs(data_type + 'Masks_' + str(i), exist_ok=True)

    for i in range(GT.size):
        for j in range(data_mat[GT[0][i]].size):
            skimage.io.imsave(root + '\\' + data_type + cell_part + '_GT_' + str(i) + str(j) + '.png',
                              data_mat[data_mat[GT[0][i]][0][j]][:])
    return


def GTtoMasks(root, data_mat, cell_part='Cytoplasm', data_type='train'):
    masks = []
    GT = data_mat[data_type + cell_part]

    for i in range(GT.size):
        mask = np.zeros((512, 512), dtype=np.uint8)
        masks.append(mask)
        for j in range(data_mat[GT[0][i]].size):
            img_temp = Image.open(
                os.path.join(root, data_type + 'Masks_' + str(i), data_type + cell_part + '_GT_' + str(j) + '.png'))
            img_temp = np.asarray(img_temp)
            img_temp = img_temp * (j + 1)  # original data is represented by 0 and 1
            img_temp = img_temp.astype(np.uint8)
            masks[i] += img_temp
        skimage.io.imsave(root + '\\' + data_type + 'Mask_00' + str(i) + '.png', masks[i])
    return


# to check if the output pictures are right, we can convert the pictures to 'RGB'
# for the data in trainset_GT, the visualized pictures are somehow transformed
# so the function is used to anti-transform.
def transformImgs(img):
    return img.rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)


def visulizeImgs(root, path_list, data_type='train', cell_part='Cytoplasm'):
    os.makedirs(root + '\\' + data_type + cell_part + 'Visul_Imgs', exist_ok=True)

    if cell_part == 'Cytoplasm':
        for i in range(len(path_list)):
            os.makedirs(root + '\\' + data_type + cell_part + 'Visul_Imgs\\' + data_type + cell_part + '_' + str(i),
                        exist_ok=True)
        for i in range(len(path_list)):
            cytoplasm_path = list(sorted(os.listdir(os.path.join(root, data_type + 'Masks', path_list[i]))))
            for j in range(len(cytoplasm_path)):
                if data_type == 'train':
                    img_temp = transformImgs(
                        Image.open(os.path.join(root, data_type + 'Masks', path_list[i], cytoplasm_path[j])))
                else:
                    img_temp = Image.open(os.path.join(root, data_type + 'Masks', path_list[i], cytoplasm_path[j]))
                img_temp = np.asarray(img_temp)
                img_temp = img_temp * 255
                img_temp = img_temp.astype(np.uint8)
                skimage.io.imsave(
                    root + '\\' + data_type + cell_part + 'Visul_Imgs\\' + data_type + cell_part + '_' + str(i) + '\\' +
                    data_type + cell_part + str(j) + '.png', img_temp)
    elif cell_part == 'Nuclei':
        for i in range(len(path_list)):
            img_temp = Image.open(os.path.join(root, data_type + cell_part, path_list[i]))
            img_temp = np.asarray(img_temp)
            img_temp = img_temp * 255
            img_temp = img_temp.astype(np.uint8)
            skimage.io.imsave(
                root + '\\' + data_type + cell_part + 'Visul_Imgs\\' + data_type + cell_part + str(i) + '.png',
                img_temp)
    return


# DATA AUGMENTATION FUNCTIONS
def voc_rand_crop(feature, masks, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    i, j, h, w = torchvision.transforms.FiveCrop.get_params(
        feature, output_size=(height, width))
    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    crop_masks = []
    for mask in masks:
        crop_masks.append(torchvision.transforms.functional.crop(mask, i, j, h, w))
    return feature, crop_masks


def cer_crop(img, masks, height, width):
    """
        Five crop feature (PIL image) and label (PIL image).
    """
    aug = torchvision.transforms.Compose(
        # [torchvision.transforms.Pad((0,320,0,0), 232, padding_mode='constant'),
        #  torchvision.transforms.Resize(1024, interpolation=2),
        [torchvision.transforms.FiveCrop(512),
         ]
    )
    img = aug(img)
    crop_masks = []
    for mask in masks:
        crop_masks.append(aug(mask))
    return img, crop_masks


def blank_filter(img, thrashold):
    """
    this function is designed to filter some augment pictures without any masks
    this situation may cause trouble while training if not fixed.
    but this function cannot fix it perfectly because there might be some pictures
    with only nucleis, which are not filtered unless nuclei masks are used.
    """
    # for a picture with size 512*512, the selected area can avoid black edge to be counted in
    img_select = img[20:480, 20:480]
    # filter pictures without cells
    pixels_range = np.max(np.unique(img_select)) - np.min(np.unique(img_select))
    return True if pixels_range < thrashold else False


def cvtJson2Masks(json_path):
    data = json.load(open(json_path))
    img = utils.img_b64_to_arr(data['imageData'])
    label_name_to_value = {'_background_': 0}
    lbl, lbl_names = utils.shapes_to_label(img.shape, data['shapes'],
                                           label_name_to_value)  # 解析'shapes'中的字段信息，解析出每个对象的mask与对应的label   lbl存储 mask，lbl_names 存储对应的label
    # lal 像素取值 0、1、2 其中0对应背景，1对应第一个对象，2对应第二个对象
    # 使用该方法取出每个对象的mask mask=[] mask.append((lbl==1).astype(np.uint8)) # 解析出像素值为1的对象，对应第一个对象 mask 为0、1组成的（0为背景，1为对象）
    # lbl_names  ['background','cat_1','cat_2']
    captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
    lbl_viz = utils.draw_label(lbl, img, captions)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl_viz)
    plt.show()


def fromJson2Masks(JSON_PATH):
    jsons_path = os.listdir(os.path.join(JSON_PATH, 'JSONFiles'))
    for json_path in jsons_path:
        data = json.load(open(os.path.join(JSON_PATH, 'JSONFiles', json_path)))
        # create files for each images
        # os.makedirs(os.path.join(PATH, json_path[:-5] + '_masks'), exist_ok=True)
        img = utils.img_b64_to_arr(data['imageData'])
        img_shape = img.shape
        shapes = data['shapes']
        cls = np.zeros(img_shape[:2], dtype=np.int32)
        ins = np.zeros_like(cls)
        instances = []
        label_name_to_value = {'_background_': 0}
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            group_id = shape.get('group_id')
            if group_id is None:
                group_id = uuid.uuid1()
            shape_type = shape.get('shape_type', None)
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

            cls_name = label
            instance = (cls_name, group_id)

            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = utils.shape_to_mask(img_shape[:2], points, shape_type)
            # skimage.io.imsave(os.path.join(PATH, json_path[:-5] + '_masks', 'mask_'+str(ins_id)+'.png'), np.array(mask, dtype='uint8'))
            cls[mask] = cls_id
            ins[mask] = ins_id


def showDataloaderImg(root):
    color_palette = [
        0, 0, 0,  # black background
        255, 0, 0,  # index 1 is red
        255, 255, 0,  # index 2 is yellow
        255, 153, 0,  # index 3 is orange
        250, 240, 230,
        230, 230, 250,
        0, 0, 128,
        135, 206, 250,
        0, 255, 255,
        255, 255, 224,
        205, 173, 0,
        245, 245, 220,
        255, 127, 80,
        255, 0, 255
    ]
    imgPath_list = list(sorted(os.listdir(os.path.join(root, "testMasks_500"))))
    maskPath_list = list(sorted(os.listdir(os.path.join(root, "testMasks_500"))))
    img_list = []
    mask_list = []

    for i in range(len(maskPath_list)):
        img_path = os.path.join(root, "PNGImages", imgPath_list[i])
        mask_path = os.path.join(root, "testMasks_500", maskPath_list[i])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = np.asarray(img)
        img_list.append(img)
        mask = np.asarray(mask)
        print(np.unique(mask))
        mask_list.append(mask)
        showMultipleImgs(img_list, mask_list, 29, 6)
        plt.show()
        mask.putpalette(color_palette)


def readFudanPedImgs(root, is_train=True, max_num=None):
    txt_fname = '%s\%s' % (root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        features[i] = np.array(Image.open('%s\PNGImages\\%s.png' % (root, fname)).convert("RGB"))
        labels[i] = np.array(Image.open('%s\PedMasks\%s_mask.png' % (root, fname)).convert("RGB")) * 255
    return features, labels


def showNucleiMasks(img_path):
    data = np.asarray(Image.open(img_path))
    labels = measure.label(data, connectivity=2)  # 8连通区域标记
    obj = np.unique(labels)
    dst = color.label2rgb(labels)  # 根据不同的标记显示不同的颜色
    print('regions number:', labels.max() + 1)  # 显示连通区域块数(从0开始标记)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(data, plt.cm.gray, interpolation='nearest')
    ax1.axis('off')
    ax2.imshow(dst,interpolation='nearest')
    ax2.axis('off')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
    ax0.imshow(data, plt.cm.gray)
    ax1.imshow(labels)

    for region in measure.regionprops(labels):  # 循环得到每一个连通区域属性集
        # 绘制外包矩形
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax1.add_patch(rect)

    fig.tight_layout()
    plt.show()


def dataAugmentor(PATH):
    PATH = r'E:\大四下\毕设\数据集\CERVIX93_Seg.&Clss\cytology_dataset-master\avaliable_imgs'
    masks_paths = list(sorted(os.listdir(os.path.join(PATH, 'Masks'))))
    masks = []
    # load original masks
    for i in range(len(masks_paths)):
        mask_path = list(sorted(os.listdir(os.path.join(PATH, 'Masks', masks_paths[i]))))
        one_pic_masks = []
        for j in range(len(mask_path)):
            one_pic_masks.append(Image.open(os.path.join(PATH, 'Masks', masks_paths[i], mask_path[j])))
        masks.append(one_pic_masks)
    imgs_path = list(sorted(os.listdir(os.path.join(PATH, 'PNGImages'))))
    imgs = []
    # load original images
    for img_path in imgs_path:
        imgs.append(Image.open(os.path.join(PATH, 'PNGImages', img_path)))
    # create blank files for the augmented images
    os.makedirs(os.path.join(PATH, 'Aug_imgs'), exist_ok=True)

    for i in range(len(imgs)):
        # get crop images and masks
        crop_img, crop_masks = cer_crop(imgs[i], masks[i], 512, 512)
        for j in range(len(crop_img)):
            if not blank_filter(np.asarray(crop_img[j]), 150):
                skimage.io.imsave(os.path.join(PATH, 'Aug_imgs', imgs_path[i][:-4] + '_' + str(j) + '.png'),
                                  np.asarray(crop_img[j]).astype(np.uint8))
                os.makedirs(os.path.join(PATH, 'Aug_masks', imgs_path[i][:-4] + '_' + str(j)), exist_ok=True)
                # os.makedirs(os.path.join(PATH, 'Aug_masks', imgs_path[i][:-4], str(j)), exist_ok=True)
        for j in range(len(crop_masks)):
            for k in range(len(crop_masks[j])):
                if len(np.unique(np.asarray(crop_masks[j][k]).astype(
                        np.uint8)[10:500, 10:500])) > 1 and os.path.exists(os.path.join(PATH, 'Aug_masks', imgs_path[i][:-4] + '_' + str(k))):
                    skimage.io.imsave(os.path.join(PATH, 'Aug_masks', imgs_path[i][:-4] + '_' + str(k), str(j) + '.png'),
                                      np.asarray(crop_masks[j][k]).astype(np.uint8))


if __name__ == '__main__':
    # create masks examples
    root = r'E:\大四下\毕设\数据集\ISBI14_Seg\Dataset\Synthetic'
    data_mat_1 = h5py.File(os.path.join(root, 'testset_GT.mat'))
    data_mat_2 = h5py.File(os.path.join(root, 'testset.mat'))
    data_mat_3 = h5py.File(os.path.join(root, 'trainset.mat'))
    data_mat_4 = scipy.io.loadmat(os.path.join(root, 'trainset_GT.mat'))

    # these codes will create binary images for ground_truth and 8-bit images
    # because the pixel value is either 0 or 1, the masks are hard to view
    h5py_mask(root, data_mat_1, 'test', 'Cytoplasm')
    h5py_mask(root, data_mat_1, 'test', 'Nuclei')
    h5py_images(root, data_mat_2, 'test')
    h5py_images(root, data_mat_3, 'train')
    loadmat_mask(root, data_mat_4, 'train', 'Cytoplasm')
    loadmat_mask(root, data_mat_4, 'train', 'Nuclei')

    # the following codes are for creating masks from train45test90
    # root = r'E:\大四下\毕设\数据集\Train45Test90'
    # data_mat_1 = scipy.io.loadmat(os.path.join(root, 'isbi_test90.mat'))
    # data_mat_2 = scipy.io.loadmat(os.path.join(root, 'isbi_test90_GT.mat'))
    # data_mat_3 = scipy.io.loadmat(os.path.join(root, 'isbi_train.mat'))
    # data_mat_4 = scipy.io.loadmat(os.path.join(root, 'isbi_train_GT.mat'))
    #
    #
    # loadmat_mask(root, data_mat_2, 'test', 'Cytoplasm')
    # loadmat_mask(root, data_mat_4, 'train', 'Cytoplasm')
    # loadmat_images(root, data_mat_1, 'Test90')
    # loadmat_images(root, data_mat_3, 'Train')

    train_masks_path = list(sorted(os.listdir(os.path.join(root, 'trainMasks'))))
    test_masks_path = list(sorted(os.listdir(os.path.join(root, 'testMasks'))))
    train_Nuclei_path = list(sorted(os.listdir(os.path.join(root, 'trainNuclei'))))
    test_Nuclei_path = list(sorted(os.listdir(os.path.join(root, 'testNuclei'))))

    visulizeImgs(root, train_masks_path)
    visulizeImgs(root, test_masks_path, 'test')
    visulizeImgs(root, train_Nuclei_path, 'train', 'Nuclei')
    visulizeImgs(root, test_Nuclei_path, 'test', 'Nuclei')

    # test loader packing functions
    data_mat = h5py.File(os.path.join(root, 'testset_GT.mat'))
    root = r'E:\大四下\毕设\第四周_读取mat图像文件\Synthetic'
    data_mat = h5py.File(os.path.join(root, 'testset_GT.mat'))
    h5py_mask(root, data_mat)
    GTtoMasks(root, data_mat)
