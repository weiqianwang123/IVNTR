import torch
from torchvision import transforms
from PIL import Image


REPO_DIR = "/home/qianwei/IVNTR/"  # path to the cloned DINO repo
# 1. Load model
dinov3_vits16 = torch.hub.load(
    REPO_DIR,               # your local DINO repo path
    'dinov3_vits16',        # model name
    source='local',
    weights="/home/qianwei/IVNTR/predicators/config/clean_table_real/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"  # your checkpoint
)
dinov3_vits16.eval()  # set to eval mode

# 2. Preprocess image
transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),  # DINO expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

img = Image.open("saved_raw_data_real/table-clean/2/whole_images/state_0.png").convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

# 3. Get features
with torch.no_grad():
    output = dinov3_vits16(img_tensor)  # forward pass

# 4. Inspect feature shape
print("Output shape:", output.shape)
