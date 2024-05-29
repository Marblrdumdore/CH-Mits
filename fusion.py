import pickle
import re
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.cuda

from torch.nn.init import xavier_normal
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.models import resnet18
from PIL import Image
import os
import jieba
from tqdm import tqdm
from transformers import ViTModel, AutoTokenizer
from functools import partial

import torch.nn.functional as F


fusion_method = 'LAFF'
txt_method = 'sknetandlstm'
img_method = 'cnn_vit'
datalength = 1

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.Linear(512,128),
            nn.Linear(128,32),
            nn.Linear(32,2)
        )
        self.fnnfc = nn.Sequential(
            nn.Linear(1536 * 2, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 2)
        )
        self.fusion_method = fusion_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tfn_block = nn.Sequential(
            nn.Linear(1537, 512),

            nn.Linear(512, 128),
            nn.Linear(128, 32),
            nn.Linear(32, 2)
        )
        self.drop = nn.Dropout(0.5)

        # LMF
        self.R = 4
        self.h = 1536
        self.Wa = Parameter(torch.Tensor(4, 1537, 1536))
        self.Wb = Parameter(torch.Tensor(4, 1537, 1536))
        self.Wf = Parameter(torch.Tensor(1, 4))
        self.bias = Parameter(torch.Tensor(1, 1536))

        xavier_normal(self.Wa)
        xavier_normal(self.Wb)
        xavier_normal(self.Wf)
        self.bias.data.fill_(0)


        # PTP
        self.PTP_R = 4
        self.PTP_h = 1536

        self.PTP_n = 5
        self.PTP_a = Parameter(torch.Tensor(4, 3073, 1536))
        self.PTP_b = Parameter(torch.Tensor(4, 3073, 1536))
        self.PTP_c = Parameter(torch.Tensor(4, 3073, 1536))
        self.PTP_d = Parameter(torch.Tensor(4, 3073, 1536))
        self.PTP_e = Parameter(torch.Tensor(4, 3073, 1536))

        self.PTP_Wf = Parameter(torch.Tensor(1, 4))
        self.PTP_bias = Parameter(torch.Tensor(1, 1536))

        xavier_normal(self.PTP_a)
        xavier_normal(self.PTP_b)
        xavier_normal(self.PTP_c)
        xavier_normal(self.PTP_d)
        xavier_normal(self.PTP_e)
        xavier_normal(self.PTP_Wf)
        self.PTP_bias.data.fill_(0)


    def LAFF(self, txt_f, img_f):
        Tanh = nn.Tanh()
        Tanh = Tanh.to(device)
        activated_txt = Tanh(txt_f)
        activated_img = Tanh(img_f)
        activated_txt = activated_txt.to(device)
        activated_img = activated_img.to(device)
        linear = nn.Linear(1536, 1)
        activated_txt = self.drop(activated_txt)
        activated_img = self.drop(activated_img)
        linear = linear.to(device)
        fusion = torch.concat((activated_txt.unsqueeze(1), activated_img.unsqueeze(1)), dim=1)
        out = linear(fusion)
        softmax = torch.nn.Softmax(dim=1)
        out = softmax(out)
        final = torch.mul(fusion, out)
        final = torch.mean(final, dim=1)
        return final

    def TFN(self, txt_f, img_f):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tail = torch.ones(txt_f.shape[0], 1)
        t = tail.to(device)
        txt_f_p = torch.cat((txt_f, t),dim=1)
        img_f_p = torch.cat((img_f, t),dim=1)
        txt_transpose = torch.unsqueeze(txt_f_p, 2)
        img_transform = torch.unsqueeze(img_f_p, 1)
        txt_transpose.to(device)
        img_transform.to(device)
        mat = torch.matmul(txt_transpose,img_transform)
        re = torch.mean(mat,dim=1)
        re = self.tfn_block(re)

        return re

    def LMF(self, txt_f, img_f):
        n = txt_f.shape[0]

        tail = torch.ones(n, 1)
        t = tail.to(device)

        txt_f = torch.cat((txt_f, t), dim=1)
        img_f = torch.cat((img_f, t), dim=1)
        Wa = self.Wa
        Wb = self.Wb
        Wf = self.Wf
        bias = self.bias

        Wt = Wa.to(device)
        Wi = Wb.to(device)
        Wff = Wf.to(device)
        b = bias.to(device)
        fusion_A = torch.matmul(txt_f, Wt)
        fusion_B = torch.matmul(img_f, Wi)

        funsion_ABC = fusion_A * fusion_B
        funsion_ABC = torch.matmul(Wff, funsion_ABC.permute(1, 0, 2)).squeeze() + b
        return funsion_ABC

    def PTP(self,txt_f, img_f):
        n = txt_f.shape[0]

        tail = torch.ones(n, 1)
        t = tail.to(device)

        i_t_f = torch.cat((t, img_f), dim=1)
        fea_v = torch.cat((i_t_f, txt_f), dim=1)

        fusion_a = torch.matmul(fea_v, self.PTP_a)
        fusion_b = torch.matmul(fea_v, self.PTP_b)
        fusion_c = torch.matmul(fea_v, self.PTP_c)
        fusion_d = torch.matmul(fea_v, self.PTP_d)
        fusion_e = torch.matmul(fea_v, self.PTP_e)

        result = fusion_a * fusion_b * fusion_c * fusion_d * fusion_e

        fusion = torch.matmul(self.PTP_Wf, result.permute(1, 0, 2)).squeeze() + self.PTP_bias
        return fusion



    def forward(self,t_input, i_input):
        if self.fusion_method == 'LAFF':
            fusion_output = self.LAFF(t_input,i_input)
        elif self.fusion_method == 'TFN':
            fusion_output = self.TFN(t_input, i_input)
            return fusion_output
        elif self.fusion_method == 'FNN':
            fusion_output = torch.concat((t_input, i_input), dim=1)
            x = self.fnnfc(fusion_output)
            return x
        elif self.fusion_method == 'LMF':
            fusion_output = self.LMF(t_input, i_input)
        elif self.fusion_method == 'PTP':
            fusion_output = self.PTP(t_input, i_input)
        x = self.fc(fusion_output)
        return x

from a import segmentor, clean_stopwords

text = '小档 溢 自由感 开心 快乐 永远'
img = 'D:\ei\Sentiment_Analysis_Imdb-master\p/p107/image_1.jpg'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

class MyDataset(Dataset):
    def __init__(self, sentences, labels, method_name, model_name):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        dataset = list()
        index = 0
        for data in sentences:
            tokens = data.split(' ')
            labels_id = labels[index]
            index += 1
            dataset.append((tokens, labels_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)

class MyDataset_single(Dataset):
    def __init__(self, sentences, labels, method_name, model_name):
        self.sentences = sentences
        self.labels = labels
        self.method_name = method_name
        self.model_name = model_name
        dataset = list()
        index = 0
        op = []
        op.append(sentences)
        for data in op:
            tokens = data.split(' ')
            labels_id = labels
            index += 1
            dataset.append((tokens, labels_id))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self.sentences)


# Make tokens for every batch
def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))

    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=320,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    # print(1,text_ids['position_ids'])
    # print(2,text_ids['attention_mask'])
    # print(3,text_ids['input_ids'])
    return text_ids, torch.tensor(label_ids)


# Load dataset
def load_txt_dataset(tokenizer, train_batch_size, test_batch_size, model_name, method_name, workers):
    data = pd.read_csv('label.csv', sep=None, header=0, encoding='utf-8', engine='python')
    len1 = int(len(list(data['labels'])) )
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8)
    # Dataset
    train_set = MyDataset(tr_sen, tr_lab, method_name, model_name)
    test_set = MyDataset(te_sen, te_lab, method_name, model_name)
    # DataLoader
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                             collate_fn=collate_fn, pin_memory=True)
    return train_loader, test_loader
class CustomDataset(Dataset):
    def __init__(self, address,label, transform=None):
        self.root_dir = ''
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像尺寸为 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        self.file_list = address
        self.labels = label




    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
def load_dataset(train_batch_size, test_batch_size,workers=0):
    data = pd.read_csv('label.csv',encoding='utf-8', engine='python')
    len1 = int(len(list(data['labels'])))
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8)
    # Dataset
    train_set = CustomDataset(tr_sen, tr_lab)
    test_set = CustomDataset(te_sen, te_lab)
    # DataLoader
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                               pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True)
    return train_loader, test_loader

class CombinedModel(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.fc1 = nn.Linear(vit_model.config.hidden_size + 50176, 768*2)
        self.fc2 = nn.Linear(768*2,2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        cnn_features = self.cnn_model(x)
        vit_features = self.vit_model(x).last_hidden_state[:,0,:]
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        x = self.fc1(combined_features)
        res = self.fc2(x)
        return res

# class VGG(nn.Module):
#     def __init__(self,  num_classes=2):
#         super(VGG, self).__init__()
#         self.features = make_features(cfg)
#         self.classifier = nn.Sequential(
#             nn.Linear(512*7*7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=0.5),
#             nn.Linear(4096, 768),
#             nn.Linear(768,2)
#         )
#         # if init_weights:
#         #     self._initialize_weights()
#
#     def forward(self, x):
#         # N x 3 x 224 x 224
#         x = self.features(x)
#         # N x 512 x 7 x 7
#         x = torch.flatten(x, start_dim=1)
#         # N x 512*7*7
#         x = self.classifier(x)
#         return x
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if txt_method == 'sknet_lstm_attention':
    txt_model = pickle.load(file=open('D:\ei\Sentiment_Analysis_Imdb-master/baseline_text_96.79.pkl', 'rb'))
elif txt_method == 'attention':
    txt_model = pickle.load(file=open('D:\ei\Sentiment_Analysis_Imdb-master/attention.pkl', 'rb'))
elif txt_method == 'bert-lstm' or txt_method == 'sknet_lstm' or txt_method == 'sknetandlstm':
    txt_model = pickle.load(file=open('D:\ei\Sentiment_Analysis_Imdb-master/bert-lstm.pkl', 'rb'))

if img_method == 'vgg':
    img_model = pickle.load(file=open('D:\ei\Sentiment_Analysis_Imdb-master/vgg1536.pkl', 'rb'))
elif img_method == 'cnn_vit':
    img_model = pickle.load(file=open('D:\ei\Sentiment_Analysis_Imdb-master/cnn+vit/cnn+vit/cv_combine.pkl', 'rb'))


def extract_txt_feature(txt_model,sentence,label,method=txt_method):
    bert_output = txt_model.base_model(**sentence)
    if method == 'sknet_lstm_attention':
        tokens = bert_output.last_hidden_state

        K = txt_model.key_layer(tokens)
        Q = txt_model.query_layer(tokens)
        V = txt_model.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * txt_model._norm_fact)
        attention_output = torch.bmm(attention, V)

        sknet_tokens = attention_output.unsqueeze(dim=1)

        for i, conv in enumerate(txt_model.convs):
            fea = conv(sknet_tokens).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = txt_model.fc(fea_s)
        for i, fc in enumerate(txt_model.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = txt_model.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)

        fea_v.squeeze_(dim=1)

        rnn_tokens = tokens
        rnn_outputs, _ = txt_model.lstm(rnn_tokens)

        rnn_outputs = rnn_outputs[:, -1, :]
        fea_v = fea_v[:, -1, :]
        return torch.concat((fea_v,rnn_outputs),dim=1)

    elif method == 'fnn':
        cls_feats = bert_output.last_hidden_state[:, 0, :]
        # fnn_linear = nn.Linear(768,1536)
        # fnn_linear.to(device)
        cls_feats.to(device)
        # cls_feats_2 = fnn_linear(cls_feats)
        cls_feats_2 = txt_model.upsample(cls_feats)
        cls_feats_2.to(device)
        return cls_feats_2

    elif method == 'attention':
        tokens = bert_output.last_hidden_state

        K = txt_model.key_layer(tokens)
        Q = txt_model.query_layer(tokens)
        V = txt_model.value_layer(tokens)
        attention = nn.Softmax(dim=-1)((torch.bmm(Q, K.permute(0, 2, 1))) * txt_model._norm_fact)
        attention_output = torch.bmm(attention, V)
        attention_output = torch.mean(attention_output, dim=1)
        attention_output = txt_model.upsample(attention_output)
        return attention_output

    elif method == 'bert-lstm':
        tokens = bert_output.last_hidden_state
        lstm_output, _ = txt_model.Lstm(tokens)
        lstm_output = txt_model.up(lstm_output)
        outputs = lstm_output[:, -1, :]
        return outputs

    elif method == 'sknet_lstm':
        tokens = bert_output.last_hidden_state.unsqueeze(1)
        for i, conv in enumerate(txt_model.convs):
            fea = conv(tokens).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = txt_model.fc(fea_s)
        for i, fc in enumerate(txt_model.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = txt_model.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v.squeeze_(dim=1)
        rnn_outputs, _ = txt_model.lstm(fea_v)
        out = rnn_outputs[:, -1, :]

        return out

    elif method == 'sknetandlstm':
        tokens = bert_output.last_hidden_state.unsqueeze(1)
        for i, conv in enumerate(txt_model.convs):
            fea = conv(tokens).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = txt_model.fc(fea_s)
        for i, fc in enumerate(txt_model.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)

            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = txt_model.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        fea_v.squeeze_(dim=1)
        fea_v = fea_v[:, -1, :]
        rnn_outputs, _ = txt_model.lstm(bert_output.last_hidden_state)
        rnn_outputs = rnn_outputs[:, -1, :]
        out = torch.cat((fea_v, rnn_outputs), dim=1)
        return out


def extract_img_feature(img_model, img, label):
    if img_method == 'vgg':
        img.to(device)
        x = img_model.features(img)
        x = torch.flatten(x, start_dim=1)
        for i, layer in enumerate(img_model.classifier):
            x = layer(x)
            if i == 6:
                break
        return x
    elif img_method == 'cnn_vit':
        img.to(device)
        cnn_features = img_model.cnn_model(img)
        vit_features = img_model.vit_model(img).last_hidden_state[:, 0, :]
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        x = img_model.fc(combined_features)
        return x

# trdata, tedata = load_txt_dataset(tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese'),
#                                   train_batch_size=1,
#                                   test_batch_size=1,
#                                   method_name='sknet_lstm_attention',
#                                   model_name='chi_bert',
#                                   workers=0)
#
# for k,v in trdata:
#     print(k)

# 获取单个标签
# text = '这个地方也太美了，风景真好'
# pattern = re.compile(r'[^\u4e00-\u9fa5]')
# chinese = re.sub(pattern,' ',text)
#
# l = list(segmentor.segment(chinese))
# print(l)
#
# li = clean_stopwords(l)
# li = " ".join(li)
# print(li)
#
# txt_model.to(device)
# tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')
#
# collate_fn = partial(my_collate, tokenizer=tokenizer)
# txt = MyDataset(li,1,'1','1')
#
# tr_lo = DataLoader(txt,collate_fn=collate_fn)
#
# for k,v in tr_lo:
#     k.to(device)
#     v.to(device)
#     predicts = txt_model.base_model(**k)
#     f = torch.argmax(predicts, dim=1)
#     print(f)
#     break


# 提取文本特征
def extract_txt_features():
    txt_model.to(device)
    tokenizer=AutoTokenizer.from_pretrained('bert-base-chinese')
    data = pd.read_csv('label.csv', sep=None, header=0, encoding='gbk', engine='python')
    len1 = int(len(list(data['labels'])) * datalength )
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    collate_fn = partial(my_collate, tokenizer=tokenizer)
    txt = MyDataset(sentences,labels,'1','1')
    txt_feature = DataLoader(txt,collate_fn=collate_fn,batch_size=batch_size,pin_memory=True,shuffle=False,num_workers=0)
    import numpy as np

    np.set_printoptions(threshold=2000)
    pd.set_option('display.max_rows', None)  # 显示所有行
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 显示所有列内容，不换行
    torch.set_printoptions(profile="full")
    #
    txt_features = []
    for sentence,label in tqdm(txt_feature, ascii='>='):
        sentence.to(device)
        # label.to(device)
        feature = extract_txt_feature(txt_model,sentence,label)
        with torch.no_grad():
            feature = feature.to('cpu')
        for index in range(feature.shape[0]):
            txt_features.append(feature[index])
    return txt_features


# 提取图像特征
def extract_img_features():
    img_model.to(device)
    data = pd.read_csv('cv_address.csv', sep=None, header=0, encoding='gbk', engine='python')
    len1 = int(len(list(data['labels'])) * datalength)
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    imgs = CustomDataset(sentences, labels)
    img_loader = DataLoader(imgs, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True)

    img_features = []
    for img_f, label in tqdm(img_loader, ascii='>='):
        img_ = img_f.to(device)
        i_feature = extract_img_feature(img_model,img_,label)
        with torch.no_grad():
            i_feature = i_feature.to('cpu')
        for index in range(i_feature.shape[0]):
            img_features.append(i_feature[index])
    return img_features

class FusionDataloader(Dataset):
    def __init__(self, tensors_tuple,  labels):
        self.features = tensors_tuple
        self.targets = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index],self.targets[index]

txt_features = extract_txt_features()
img_features = extract_img_features()
data = pd.read_csv('cv_address.csv', sep=None, header=0, encoding='gbk', engine='python')
len1 = int(len(list(data['labels'])) * datalength )
labels = list(data['labels'])[0:len1]

features = []
for index in range(len(txt_features)):
    element = (txt_features[index],img_features[index])
    features.append(element)

tr_data,  te_data, tr_lab, te_lab = train_test_split(features, labels, train_size=0.8)
train_set = FusionDataloader(tr_data,tr_lab)
test_set = FusionDataloader(te_data,te_lab)
train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True,num_workers=0, pin_memory=True )
test_dataloader = DataLoader(test_set, batch_size=4, shuffle=True,num_workers=0, pin_memory=True )

num_classes = 2
num_epochs = 100
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
fusion_model = FusionModel()
optimizer = optim.Adam(fusion_model.parameters(), lr=learning_rate)
fusion_model = fusion_model.to(device)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    n_classes = 2
    target_num = torch.zeros((1, n_classes))  # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))



    for features, labels in tqdm(dataloader, ascii='>='):
        txts = features[0]
        images = features[1]
        texts = txts.to(device)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(texts,images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()


        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
        tar_mask = torch.zeros(outputs.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)  # 得到数据中每类的数量
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量



        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    train_acc = correct / total
    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

    F1 = float(F1.sum() / 2)
    return train_loss, train_acc, F1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    n_classes = 2
    target_num = torch.zeros((1, n_classes))  # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, n_classes))
    acc_num = torch.zeros((1, n_classes))

    with torch.no_grad():
        for features, labels in tqdm(dataloader, ascii='>='):
            txts = features[0]
            images = features[1]
            texts = txts.to(device)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(texts,images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            tar_mask = torch.zeros(outputs.size()).scatter_(1, labels.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量

    val_loss = running_loss / len(dataloader)
    val_acc = correct / total

    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = 100. * acc_num.sum(1) / target_num.sum(1)
    print('F1', F1)
    F1 = float(F1.sum() / 2)
    precision = float(precision.sum() / 2)
    recall = float(recall.sum() / 2)

    print('test precision:',precision)
    print('test recall', recall)


    return val_loss, val_acc, F1


best_loss, best_acc = 0, 0
for epoch in range(num_epochs):
    train_loss, train_acc, train_f1 = train(fusion_model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc, test_f1 = evaluate(fusion_model, test_dataloader, criterion, device)
    if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
        best_acc, best_loss = val_acc, val_loss
        with open('cv_combine.pkl', "wb") as file:
            pickle.dump(fusion_model, file)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}  Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}  Test F1: {test_f1:.4f}")
    print("--------------------------")
print('best_acc',best_acc)