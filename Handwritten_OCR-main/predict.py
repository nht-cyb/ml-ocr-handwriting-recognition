import argparse
from PIL import Image
import os
from tool.predictor import Predictor
from tool.config import Cfg

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--img', required=True, help='foo help')
#     parser.add_argument('--config', required=True, help='foo help')

#     args = parser.parse_args()
#     config = Cfg.load_config_from_file(args.config)

#     detector = Predictor(config)

#     img = Image.open(args.img)
#     s = detector.predict(img)

#     print(s)
# https://drive.google.com/file/d/1gT0dHMt6Lw7umGm7QKXMhqF0kZ_FcTij/view?usp=sharing

if __name__ == '__main__':
 
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] ='/home/tienvh/hoang_uet/vietocr/checkpoint/transformerocr_checkpoint_pretrained_v7.pth'
    print(config)
    txt_file_path = '/home/tienvh/hoang_uet/vietocr/checkpoint/predict3.txt'
    detector = Predictor(config)
    test_path = '/home/tienvh/hoang_uet/vietocr/new_public_test'
    # Mở tệp txt để viết thông tin
    with open(txt_file_path, 'w') as txt_file:
        # Lặp qua các tệp hình ảnh trong thư mục
        for image_filename in os.listdir(test_path):
            # Kiểm tra nếu là tệp hình ảnh (có thể cần kiểm tra phần mở rộng tệp tùy thuộc vào loại hình ảnh bạn có)
            image_path = os.path.join(test_path, image_filename)
            img = Image.open(image_path)
            label = detector.predict(img)
            image_info = f"{image_filename} {label}\n"  # Thay label bằng nhãn thích hợp
            print(image_info)
            txt_file.write(image_info)

    print(f"Thông tin đã được lưu vào '{txt_file_path}'")



