import argparse
from model.trainer import Trainer
from       tool.config import Cfg
# https://drive.google.com/file/d/1xsUFEkrIK_R_cIDMvxiPHnELmXR0tLgj/view?usp=sharing
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='see example at ')
    parser.add_argument('--checkpoint', required=False, help='your checkpoint')

    args = parser.parse_args()
    config = Cfg.load_config_from_file(args.config)

    trainer = Trainer(config)

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        
    trainer.train()

if __name__ == '__main__':
    config = Cfg.load_config_from_name('vgg_transformer')
dataset_parameters ={
        'name': 'handwritten',
        'data_root': '/home/tienvh/hoang_uet/vietocr/new_train',
        'train_annotation': '/home/tienvh/hoang_uet/vietocr/train_file_1.txt',
        'valid_annotation': '/home/tienvh/hoang_uet/vietocr/vietocr/validation_file_2.txt'
    }
params = {
         'print_every':200,
         'valid_every':100*15,
          'iters':20000,
        #  'checkpoint': '/content/drive/MyDrive/ver1/checkpoint/transformerocr_checkpoint_pretrained_v4.pth',
          'export':'/home/tienvh/hoang_uet/vietocr/checkpoint/transformerocr_checkpoint_pretrained_v7.pth',
          'metrics': 10000
         }
config['trainer'].update(params)

config['dataset'].update(dataset_parameters)
config['device'] = 'cuda:0'
trainer = Trainer(config, pretrained= False)
trainer.train()