import json
import os
import tqdm
from PIL import Image
import io
import base64
from PIL import PngImagePlugin

# https://stackoverflow.com/questions/42671252/python-pillow-valueerror-decompressed-data-too-large
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def save_base64_image(base64_str, save_path):
    # 将Base64字符串解码为字节数据
    image_data = base64.b64decode(base64_str)

    # 创建BytesIO对象
    image_stream = io.BytesIO(image_data)

    # 打开图像并创建PIL图像对象
    image = Image.open(image_stream)

    # 获取图像的原始尺寸
    width, height = image.size

    # 计算缩放后的尺寸
    if width < height:
        new_width = 336
        new_height = int(height * (336 / width))
    else:
        new_width = int(width * (336 / height))
        new_height = 336

    # 缩放图像
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # 保存图像到指定路径
    image.save(save_path)

image_file = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/MIMIC-IT/TVC.json'
image_dir = '/cpfs/shared/research-llm/instruc_data_en/multimodal_instruct_tuning/MIMIC-IT/images/TVC'

print(image_file)
os.makedirs(image_dir, exist_ok=True)

images = json.load(open(image_file, 'r'))

for image_id, base64_str in tqdm.tqdm(images.items()):
    save_base64_image(images[image_id], os.path.join(image_dir, image_id+'.png'))