import math

import numpy
import torch
import torch.nn.functional as F
from torch import nn


# Bert + FNN
class Transformer(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.linear = nn.Linear(1536, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax()
        self.upsample = nn.Linear(768,1536)
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        # The pooler_output is made of CLS --> FNN --> Tanh
        # The last_hidden_state[:,0] is made of original CLS
        # Method one
        # cls_feats  = raw_outputs.pooler_output
        # Method two
        cls_feats = raw_outputs.last_hidden_state[:, 0, :]
        predicts = self.softmax(self.linear(self.upsample(self.dropout(cls_feats))))
        return predicts


class Gru_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Gru = nn.GRU(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        gru_output, _ = self.Gru(tokens)
        outputs = gru_output[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


# Try to use the softmax、relu、tanh and logistic
class Lstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=320,
                            num_layers=1,
                            batch_first=True)
        self.up = nn.Linear(320,1536)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(1536, 768),
                                nn.Linear(768, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))

        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        lstm_output, _ = self.Lstm(tokens)
        outputs = lstm_output[:, -1, :]
        outputs = self.up(outputs)
        outputs = self.fc(outputs)
        return outputs


class BiLstm_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        # Open the bidirectional
        self.BiLstm = nn.LSTM(input_size=self.input_size,
                              hidden_size=320,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320 * 2, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.BiLstm(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs


class Rnn_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.Rnn = nn.RNN(input_size=self.input_size,
                          hidden_size=320,
                          num_layers=1,
                          batch_first=True)
        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(320, 80),
                                nn.Linear(80, 20),
                                nn.Linear(20, self.num_classes),
                                nn.Softmax(dim=1))
        for param in base_model.parameters():
            param.requires_grad = (True)

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cls_feats = raw_outputs.last_hidden_state
        outputs, _ = self.Rnn(cls_feats)
        outputs = outputs[:, -1, :]
        outputs = self.fc(outputs)
        return outputs

class SKNet_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.convs = nn.ModuleList([])
        features = 1
        M = 3
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3 + i * 2,3 + i * 2),
                              stride=(1,1),
                              padding=1 + i
                              ), nn.BatchNorm2d(1)#,
                    #nn.ReLU(inplace=False))
                ))

        self.fc = nn.Linear(1, 32)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(32, 1))
        self.softmax = nn.Softmax(dim=1)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            fea = conv(tokens).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v.squeeze_(dim=1)
        out = fea_v[:, -1, :]
        predicts = self.block(out)
        return predicts

class SKNet_LSTM_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.convs = nn.ModuleList([])
        features = 1
        M = 2
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3 + i * 2,3 + i * 2),
                              stride=(1,1),
                              padding=1 + i
                              ), nn.BatchNorm2d(1)#,
                    #nn.ReLU(inplace=False))
                ))

        self.fc = nn.Linear(1, 32)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(32, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=1536,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False
                            )
        self.up = nn.Linear(1040,1536)
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            fea = conv(tokens).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v.squeeze_(dim=1)
        rnn_outputs, _ = self.lstm(fea_v)
        out = rnn_outputs[:, -1, :]
        predicts = self.block(out)
        return predicts

class SKNetandLSTM_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.convs = nn.ModuleList([])
        features = 1
        M = 2
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3 + i * 2,3 + i * 2),
                              stride=(1,1),
                              padding=1 + i
                              ), nn.BatchNorm2d(1)#,
                    #nn.ReLU(inplace=False))
                ))

        self.fc = nn.Linear(1, 32)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(32, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=768,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False
                            )
        self.up = nn.Linear(1040,1536)
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        for i, conv in enumerate(self.convs):
            fea = conv(tokens).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v.squeeze_(dim=1)
        fea_v = fea_v[:, -1, :]
        rnn_outputs, _ = self.lstm(raw_outputs.last_hidden_state)
        rnn_outputs = rnn_outputs[:, -1, :]
        out = torch.cat((fea_v, rnn_outputs),dim=1)
        predicts = self.block(out)
        return predicts

class SKNet_LSTM_Attention_Model(nn.Module):
    def __init__(self, base_model, num_classes, input_size):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.input_size = input_size
        self.convs = nn.ModuleList([])
        dim_att = 768
        self.key_layer = nn.Linear(dim_att, dim_att)
        self.query_layer = nn.Linear(dim_att, dim_att)
        self.value_layer = nn.Linear(dim_att, dim_att)
        self._norm_fact = 1 / math.sqrt(dim_att)

        features = 1
        M = 2
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=1,
                              kernel_size=(3 + i * 2,3 + i * 2),
                              stride=(1,1),
                              padding=1 + i
                              ), nn.BatchNorm2d(1)#,
                    #nn.ReLU(inplace=False))
                ))

        self.fc = nn.Linear(1, 32)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(32, 1))
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=768,
                            num_layers=1,
                            batch_first=True,
                            )
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768 * 2,128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

        dims = 1288 * 8
        self.heads = 8
        drop_out_num = 0.0
        self.multi_head_attention = torch.nn.MultiheadAttention(embed_dim=dims, num_heads=self.heads)

    def LAFF(self, sknet_out, lstm_out):
        Tanh = nn.Tanh()
        # sknet_out = Tanh(sknet_out)
        # lstm_out = Tanh(lstm_out)
        linear = nn.Linear(768,1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        linear = linear.to(device)
        fusion = torch.concat((sknet_out,lstm_out),dim=1)
        out = linear(fusion)
        softmax = torch.nn.Softmax(dim=1)
        out = softmax(out)
        final = torch.mul(fusion,out)
        final = torch.mean(final,dim=1)
        return final



    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state


        # attention_output = torch.mean(attention_output, dim=1)

        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)


        sknet_tokens = attention_output.unsqueeze(dim=1)

        for i, conv in enumerate(self.convs):
            fea = conv(sknet_tokens).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        fea_v.squeeze_(dim=1)

        # K = self.key_layer(fea_v)
        # Q = self.query_layer(fea_v)
        # V = self.value_layer(fea_v)
        # attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        # attention_output = torch.bmm(attention, V)
        # attention_output = torch.mean(attention_output, dim=1)

        rnn_tokens = attention_output
        rnn_outputs, _ = self.lstm(rnn_tokens)


        rnn_outputs = rnn_outputs[:, -1, :]
        fea_v = fea_v[:, -1, :]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # out = self.LAFF(fea_v.unsqueeze(1), rnn_outputs.unsqueeze(1))
        out = torch.cat((fea_v, rnn_outputs), 1)




        #self-attention
        # out = out.unsqueeze(dim=1)
        # K = self.key_layer(out)
        # Q = self.query_layer(out)
        # V = self.value_layer(out)
        # attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        # attention_output = torch.bmm(attention, V)
        # K = K.tile(1, 1, self.heads)
        # Q = Q.tile(1, 1, self.heads)
        # V = V.tile(1, 1, self.heads)
        #multihead-attention

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        #
        # layer = layer.to(device)
        # attention_output, _ = self.multi_head_attention(Q, K, V)

        # attention_output = out


        # attention_output = attention_output.squeeze(1)
        # attention_output = attention_output[:, -1, :]
        predicts = self.block(out)
        return predicts

class TextCNN_Model(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 2
        self.encode_layer = 12

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        tokens = conv(tokens)
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)
        tokens = F.max_pool1d(tokens, tokens.size(2))
        out = tokens.squeeze(2)
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state.unsqueeze(1)
        out = torch.cat([self.conv_pool(tokens, conv) for conv in self.convs],
                        1)
        predicts = self.block(out)
        return predicts


class Transformer_CNN_RNN(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        cnn_tokens = raw_outputs.last_hidden_state.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]
        rnn_tokens = raw_outputs.last_hidden_state
        rnn_outputs, _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        predicts = self.block(out)
        return predicts


class Transformer_Attention(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)
        self.upsample = nn.Linear(768,1536)
        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, inputs):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state

        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)
        attention_output = torch.mean(attention_output, dim=1)
        attention_output = self.upsample(attention_output)
        predicts = self.block(attention_output)
        return predicts


class Transformer_CNN_RNN_Attention(nn.Module):
    def __init__(self, base_model, num_classes, is_val=None):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        for param in base_model.parameters():
            param.requires_grad = (True)

        # Define the hyperparameters
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100
        self.is_val = is_val

        # TextCNN
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.num_filters,
                       kernel_size=(K, self.base_model.config.hidden_size)) for K in self.filter_sizes]
        )

        # LSTM
        self.lstm = nn.LSTM(input_size=self.base_model.config.hidden_size,
                            hidden_size=512,
                            num_layers=1,
                            batch_first=True,
                            )
        # Self-Attention
        self.key_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.query_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self.value_layer = nn.Linear(self.base_model.config.hidden_size, self.base_model.config.hidden_size)
        self._norm_fact = 1 / math.sqrt(self.base_model.config.hidden_size)

        self.block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(812, 128),
            nn.Linear(128, 16),
            nn.Linear(16, num_classes),
            nn.Softmax(dim=1)
        )

    def conv_pool(self, tokens, conv):
        # x -> [batch,1,text_length,768]
        tokens = conv(tokens)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1, 1]
        tokens = F.relu(tokens)
        tokens = tokens.squeeze(3)  # shape [batch_size, out_channels, x.shape[2] - conv.kernel_size[0] + 1]
        tokens = F.max_pool1d(tokens, tokens.size(2))  # shape[batch, out_channels, 1]
        out = tokens.squeeze(2)  # shape[batch, out_channels]
        return out

    def forward(self, inputs, is_val=None):
        raw_outputs = self.base_model(**inputs)
        tokens = raw_outputs.last_hidden_state
        # Self-Attention
        K = self.key_layer(tokens)
        Q = self.query_layer(tokens)
        V = self.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact)
        attention_output = torch.bmm(attention, V)

        # TextCNN
        cnn_tokens = attention_output.unsqueeze(1)  # shape [batch_size, 1, max_len, hidden_size]
        cnn_out = torch.cat([self.conv_pool(cnn_tokens, conv) for conv in self.convs],
                            1)  # shape  [batch_size, self.num_filters * len(self.filter_sizes]

        rnn_tokens = tokens
        rnn_outputs , _ = self.lstm(rnn_tokens)
        rnn_out = rnn_outputs[:, -1, :]
        # cnn_out --> [batch,300]
        # rnn_out --> [batch,512]
        out = torch.cat((cnn_out, rnn_out), 1)
        if is_val != None:
            print("out:",out)
            return out
        predicts = self.block(out)
        return predicts
