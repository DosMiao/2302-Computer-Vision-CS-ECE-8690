import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model


def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize((513, 513)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),])
    input_tensor = preprocess(input_image)
    return input_image, input_tensor.unsqueeze(0)


def segment_image(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output


def decode_segmap(image, output):
    _, preds = torch.max(output, 0)

    pascal_voc_colormap = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0]
    ]
    label_colors = np.array(pascal_voc_colormap, dtype=np.uint8)

    rgb = np.zeros((513, 513, 3), dtype=np.uint8)
    for label in range(0, len(label_colors)):
        idx = preds == label
        rgb[idx] = label_colors[label]

    return Image.fromarray(rgb), preds


def plot_results(input_image, segmentation_mask, preds):
    pascal_voc_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                          'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    pascal_voc_colormap = [
        [0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
        [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
        [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0]
    ]
    label_colors = np.array(pascal_voc_colormap, dtype=np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(input_image)
    axes[0].set_title('Original Image')
    axes[1].imshow(segmentation_mask)
    axes[1].set_title('Segmentation Mask')

    # Create a legend using class colors and names
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(
            color=(label_colors[i] / 255.), label=pascal_voc_classes[i])
        for i in range(len(pascal_voc_classes))
        if i in np.unique(preds)
    ]
    axes[1].legend(handles=legend_elements, loc='upper left',
                   bbox_to_anchor=(1, 1), title="Classes")

    # plt.show() with 2 seconds pause
    plt.show(block=False)   # show the image without blocking the code
    plt.pause(2)            # pause the code execution for 2 seconds

    # return the figure
    return fig


def main():
    test_images = ['./CV2023_HW4A/test_img/HW4a_Test1.jpg',
                   './CV2023_HW4A/test_img/HW4a_Test2.jpg',
                   './CV2023_HW4A/test_img/HW4a_Test3.jpg']
    model = load_model()

    for image_path in test_images:
        input_image, input_tensor = preprocess_image(image_path)
        output = segment_image(model, input_tensor)
        segmentation_mask, preds = decode_segmap(input_image, output)
        fig = plot_results(input_image, segmentation_mask, preds)
        # save the fig with associated image name
        fig.savefig(image_path[:-4] + '_segmentation.png')
        plt.close(fig)


if __name__ == "__main__":
    main()
