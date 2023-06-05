import torch

from detectron2.data.detection_utils import read_image
from detectron2.data.transforms import ResizeTransform, TransformList


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]
        # 代码定义了一个名为normalize_bbox的函数，用于将bounding box信息进行归一化。具体来说，输入参数bbox是一个四元组，表示一个矩形的左上角和右下角的坐标；
        # size是一个二元组，表示图片的宽和高。函数的输出是一个四元组，表示归一化后的矩形的左上角和右下角的坐标。
        # 归一化的方法是将bounding box的坐标值乘以1000，然后除以图片的宽和高，得到的值即为归一化后的坐标值（单位为千分之一）。
        # 这样做的目的是将bounding box的坐标值缩放到[0, 1000]的范围内，方便后续处理。


def simplify_bbox(bbox):
    return [
        min(bbox[0::2]),
        min(bbox[1::2]),
        max(bbox[2::2]),
        max(bbox[3::2]),
    ]
        # 代码定义了一个名为simplify_bbox的函数，用于简化bounding box的表示形式。
        # 具体来说，输入参数bbox是一个四元组，表示一个矩形的左上角和右下角的坐标。
        # 函数的输出也是一个四元组，表示简化后的矩形的左上角和右下角的坐标。
        # 简化的方法是将bounding box的坐标值分别取出最小值和最大值，得到的四个值即为简化后的矩形的左上角和右下角的坐标。
        # 这样做的目的是将bounding box的表示形式从四个坐标值变成两个坐标值，方便后续处理。

def merge_bbox(bbox_list):
    x0, y0, x1, y1 = list(zip(*bbox_list))
    return [min(x0), min(y0), max(x1), max(y1)]
        # 代码定义了一个名为merge_bbox的函数，用于将多个bounding box合并成一个。
        # 具体来说，输入参数bbox_list是一个列表，列表中的每个元素都是一个四元组，表示一个矩形的左上角和右下角的坐标。
        # 函数的输出是一个四元组，表示合并后的矩形的左上角和右下角的坐标。
        # 合并的方法是将所有bounding box的x坐标和y坐标分别取出来，然后分别求出最小值和最大值，得到的四个值即为合并后的矩形的左上角和右下角的坐标。
        # 这样做的目的是将多个bounding box合并成一个bounding box


def load_image(image_path):
    image = read_image(image_path, format="BGR")
    h = image.shape[0]
    w = image.shape[1]
    img_trans = TransformList([ResizeTransform(h=h, w=w, new_h=224, new_w=224)])
    image = torch.tensor(img_trans.apply_image(image).copy()).permute(2, 0, 1)  # copy to make it writeable
    return image, (w, h)
    # 该函数采用一个图像路径作为输入参数，并使用read_image函数从该路径读取图像，并将其解码为BGR格式。
    # 然后，它获取图像的高度和宽度，并创建一个TransformList对象，该对象包含一个ResizeTransform对象，用于将图像的大小调整为224x224像素。
    # 接下来，该函数将图像传递给这个TransformList对象，并使用apply_image方法将其转换为指定大小的图像。
    # 最后，该函数将转换后的图像转换为PyTorch张量，并对其维度进行重排，以便它们符合PyTorch网络所需的格式。
    # 最终函数返回预处理后的图像张量以及原始图像的宽度和高度元组，以便后续处理或显示。