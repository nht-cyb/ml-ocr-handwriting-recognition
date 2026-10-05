import torch.nn as nn


class CRNN(nn.Module):
    def __init__(
        self,
        img_height,
        img_width,
        img_channel,
        num_class,
        leaky_relu=False,
        map_to_seq_hidden=256,
        seq_hidden=256,
    ):
        super().__init__()
        
        assert img_height % 16 == 0
        assert img_width % 4 == 0
        
        channels = [64, 128, 256, 256, 512, 512, 512]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        strides = [1, 1, 1, 1, 1, 1, 1]
        paddings = [1, 1, 1, 1, 1, 1, 0]

        cnn = nn.Sequential()

        def conv_relu(i, batch_norm=False):
            inp_channel = img_channel if i == 0 else channels[i - 1]
            out_channel = channels[i]

            cnn.add_module(
                f"conv{i}",
                nn.Conv2d(
                    inp_channel, out_channel, kernel_sizes[i], strides[i], paddings[i]
                ),
            )
            
            if batch_norm:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(out_channel))
            if leaky_relu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(inplace=True))

        conv_relu(0)
        cnn.add_module('pooling1', nn.MaxPool2d(kernel_size=2, stride=2))
        # (64, img_height // 2, img_width // 2)
        
        conv_relu(1)
        cnn.add_module('pooling2', nn.MaxPool2d(kernel_size=2, stride=2))
        # (128, img_height // 4, img_width // 4)
        
        conv_relu(2)
        conv_relu(3)
        cnn.add_module('pooling3', nn.MaxPool2d(kernel_size=(2, 1)))
        # (256, img_height // 8, img_width // 4)
        
        conv_relu(4, batch_norm=True)
        conv_relu(5, batch_norm=True)
        cnn.add_module('pooling4', nn.MaxPool2d(kernel_size=(2, 1)))
        # (512, img_height // 16, img_width // 8)
        
        conv_relu(6)
    
        self.cnn = cnn
        
        self.map_to_seq = nn.Linear(channels[-1], map_to_seq_hidden)
        self.rnn1 = nn.LSTM(map_to_seq_hidden, seq_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * seq_hidden, seq_hidden, bidirectional=True)
        
        self.fc = nn.Linear(2 * seq_hidden, num_class)
        
    def forward(self, inp):
        conv = self.cnn(inp)
        
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1) # (width, batch, channel * height)
        seq = self.map_to_seq(conv)
        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)
        
        scores = self.fc(recurrent)
        
        return scores # (seq_len, batch, num_class)
        
        