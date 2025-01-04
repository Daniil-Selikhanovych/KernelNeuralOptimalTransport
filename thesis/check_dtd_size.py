from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop

path_to_dtd = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/dtd/images"

img_size=64

transform = Compose(
            [Resize(300), RandomResizedCrop((img_size,img_size), scale=(128./300, 1.), ratio=(1., 1.)),
             RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
             ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
dataset = ImageFolder(path_to_dtd, transform=transform)

print(len(dataset))