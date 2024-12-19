from PIL import Image

from torchvision import models
import torchvision.transforms.v2 as tfs_v2

vgg_weights = models.VGG16_Weights.DEFAULT

cats = vgg_weights.meta['categories']
transforms_1 = vgg_weights.transforms()
transforms_2 = tfs_v2.Compose([
    tfs_v2.ToImage(),
    tfs_v2.Resize(256),
    tfs_v2.CenterCrop(224),
    tfs_v2.ToDtype(dtype=torch.float32, scale=True)
    tfs_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img = Image.open('img_2.jpg').convert('RGB')
img_net = transforms_2(img).unsqueeze(0)

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

model.eval()
p = model(img_net).squeeze() # (1000)
res = p.softmax(dim=0).sort(descending=True)

mf = model.features

for s, i in zip(res[0][:5], res[1][:5]):
    print(f"{cats[i]}: {s:.4f}")