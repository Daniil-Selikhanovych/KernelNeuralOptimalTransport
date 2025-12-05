from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop
from torchvision.datasets import ImageFolder

img_size = 64

path = "/trinity/home/daniil.selikhanovych/my_thesis/datasets/houzz/chairs"

transform_train = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), RandomHorizontalFlip()])
transform_test = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = ImageFolder(path, transform=transform_train)

print(f"len chairs = {len(dataset)}")