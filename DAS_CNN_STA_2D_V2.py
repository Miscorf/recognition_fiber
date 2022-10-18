import logging

logging.basicConfig(level=logging.INFO)
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from torch.utils.data import Dataset,DataLoader
from  torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import h5py
from scipy.fftpack import fft,ifft
import glob
import numpy as np
import scipy
import scipy.signal
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime,timedelta
def filter(signalarray):
    b, a = scipy.signal.butter(8, 0.2, 'lowpass')
    signalarray = scipy.signal.filtfilt(b, a, signalarray)
    return signalarray
def standardization(data):
    data_features_mean = np.mean(data, axis=1)
    data_features_sigma = np.std(data, axis=1)
    data_features = (data - data_features_mean.reshape((data.shape[0], 1))) / data_features_sigma.reshape(
        (data.shape[0], 1))
    return data_features
def TWO_d_READ_STA_LTA(PATH,CHANNEL_INDEX,lamda):#
    files = glob.glob(PATH)
    spatial_Interval = 2.0449898
    Frequency = 2000
    time_interval = 1 / Frequency
    reshapewindowslength = 2 / time_interval
    ns=200
    nl=1000
    signal = []
    signalarray = []
    for file in files:
        file_data=h5py.File(file,'r')
        data=file_data['MultiwavelengthData']
        data_array=np.array(data)
        data_array=data_array[:,:250]
        data_array=scipy.signal.decimate(data_array,2,axis=1)
        #data_array=filter(data_array)
        #print(data_array.shape)
        spatial_data = data_array[:, int(CHANNEL_INDEX / spatial_Interval) - 5:
                                     int(CHANNEL_INDEX / spatial_Interval) + 5]
        '''spatial_data = spatial_data.reshape(int(spatial_data.shape[0] / reshapewindowslength),
                                            int(reshapewindowslength), spatial_data.shape[1])'''
        spatial_data = np.array_split(spatial_data,
                                      int(spatial_data.shape[0] / reshapewindowslength), axis=0)
        spatial_data=np.array(spatial_data)
        spatial_data_cf = np.power(spatial_data, 2)#样本，时间，位置
        for i in range(spatial_data_cf.shape[0]):
            sample_data = spatial_data_cf[i]#提出一个样本
            OUT_A = []
            for m in range(sample_data.shape[1]):#提出一个样本的一列
                space_data = sample_data[:, m]
                OUT = []
                for n in range(space_data.shape[0]):
                    if space_data.shape[0] - n - 1 >= nl:
                        Sta = np.sum(space_data[n:n + ns ])
                        Lta = np.sum(space_data[n:n + nl ])
                        out = Sta / Lta
                        OUT.append(out)
                OUT_A.append(OUT)
            OUT_A = np.array(OUT_A)
            SUM = np.sum(OUT_A, axis=0)/OUT_A.shape[0]
            MAX = np.max(SUM)
            #print(MAX)
            if MAX>lamda:
              signal.append(spatial_data[i])
    signal=np.array(signal)
    for i in range(signal.shape[0]):
        frequency_data = fft(signal[i])
        frequency_data_half = np.abs(frequency_data[0:len(signal[i]) // 2])
        frequency_data_half=frequency_data_half.transpose(1,0)
        frequency_data_half=standardization(frequency_data_half)
        signalarray.append(frequency_data_half)
    signalarray=np.array(signalarray)
    return signalarray
def TWO_d_READ_STA_LTA_CAR(PATH,CHANNEL_INDEX,lamda):#
    files = glob.glob(PATH)
    spatial_Interval = 2.0449898
    Frequency = 2000
    time_interval = 1 / Frequency
    reshapewindowslength = 2 / time_interval
    ns=200
    nl=1000
    signal = []
    signalarray = []
    for file in files:
        file_data=h5py.File(file,'r')
        data=file_data['MultiwavelengthData']
        data_array=np.array(data)
        #data_array=scipy.signal.decimate(data_array,2,axis=1)
        #data_array=filter(data_array)
        #print(data_array.shape)
        spatial_data = data_array[:, int(CHANNEL_INDEX / spatial_Interval) - 5:
                                     int(CHANNEL_INDEX / spatial_Interval) + 5]
        '''spatial_data = spatial_data.reshape(int(spatial_data.shape[0] / reshapewindowslength),
                                            int(reshapewindowslength), spatial_data.shape[1])'''
        spatial_data = np.array_split(spatial_data,
                                      int(spatial_data.shape[0] / reshapewindowslength), axis=0)
        spatial_data=np.array(spatial_data)
        spatial_data_cf = np.power(spatial_data, 2)#样本，时间，位置
        for i in range(spatial_data_cf.shape[0]):
            sample_data = spatial_data_cf[i]#提出一个样本
            OUT_A = []
            for m in range(sample_data.shape[1]):#提出一个样本的一列
                space_data = sample_data[:, m]
                OUT = []
                for n in range(space_data.shape[0]):
                    if space_data.shape[0] - n - 1 >= nl:
                        Sta = np.sum(space_data[n:n + ns ])
                        Lta = np.sum(space_data[n:n + nl ])
                        out = Sta / Lta
                        OUT.append(out)
                OUT_A.append(OUT)
            OUT_A = np.array(OUT_A)
            SUM = np.sum(OUT_A, axis=0)/OUT_A.shape[0]
            MAX = np.max(SUM)
            #print(MAX)
            if MAX>lamda:
              signal.append(spatial_data[i])
    signal=np.array(signal)
    for i in range(signal.shape[0]):
        frequency_data = fft(signal[i])
        frequency_data_half = np.abs(frequency_data[0:len(signal[i]) // 2])
        frequency_data_half=frequency_data_half.transpose(1,0)
        frequency_data_half=standardization(frequency_data_half)
        signalarray.append(frequency_data_half)
    signalarray=np.array(signalarray)
    return signalarray
def creat_one_hot(num_classes,num_sample,index_classes):#制作标签
    label=np.zeros(num_sample)+index_classes
    label_array=label.reshape((num_sample,1))
    num_label=label_array.shape[0]
    index_offset=np.arange(num_label)*num_classes
    label_one_hot=np.zeros((num_label,num_classes))
    label_one_hot.flat[index_offset+label_array.astype(int).ravel()]=1
    label_one_hot=label_one_hot.astype(np.float32)
    return label_one_hot
car_data_read=TWO_d_READ_STA_LTA_CAR('E:\\DAS_AI_DIKONG\\CAR\\*.h5',350,0.23)
print(car_data_read.shape)
car_train_features=car_data_read[:115,:,:]
car_train_one_hot=creat_one_hot(4,car_train_features.shape[0],0)
car_test_features=car_data_read[115:,:,:]
car_test_one_hot=creat_one_hot(4,car_test_features.shape[0],0)
#print(car_train_one_hot,car_train_one_hot.shape)
#print(car_test_one_hot,car_test_one_hot.shape)
#print(car_train_features.shape,car_test_features.shape)
hammer_data_read=TWO_d_READ_STA_LTA('E:\\zipeng ai\\HIT\\*.h5',117,0.28)
hammer_train_features=hammer_data_read[:51,:,:]
hammer_train_one_hot=creat_one_hot(4,hammer_train_features.shape[0],1)
hammer_test_features=hammer_data_read[51:,:,:]
hammer_test_one_hot=creat_one_hot(4,hammer_test_features.shape[0],1)
#print(hammer_train_one_hot,hammer_train_one_hot.shape)
#print(hammer_test_one_hot,hammer_test_one_hot.shape)
#print(hammer_train_features.shape,hammer_test_features.shape)
print(hammer_data_read.shape)
hit_data_read=TWO_d_READ_STA_LTA('E:\\zipeng ai\\CHANDI\\*.h5',117,0.28)
hit_train_features=hit_data_read[:50,:,:]
hit_train_one_hot=creat_one_hot(4,hit_train_features.shape[0],2)
hit_test_features=hit_data_read[50:,:,:]
hit_test_one_hot=creat_one_hot(4,hit_test_features.shape[0],2)
#print(hit_train_one_hot,hit_train_one_hot.shape)
#print(hit_test_one_hot,hit_test_one_hot.shape)
#print(hit_train_features.shape,hit_test_features.shape)
print(hit_data_read.shape)
RUN_data_read=TWO_d_READ_STA_LTA('E:\\zi peng\\WALK\\*.h5',117,0.33)
RUN_train_features=RUN_data_read[:95,:]
RUN_train_one_hot=creat_one_hot(4,RUN_train_features.shape[0],3)
RUN_test_features=RUN_data_read[95:,:]
RUN_test_one_hot=creat_one_hot(4,RUN_test_features.shape[0],3)
#print(RUN_train_one_hot,RUN_train_one_hot.shape)
#print(RUN_test_one_hot,RUN_test_one_hot.shape)
#print(RUN_train_features.shape,RUN_test_features.shape)
print(RUN_data_read.shape)
train_features=np.vstack((car_train_features,hammer_train_features,
                          hit_train_features,RUN_train_features))
train_labels=np.vstack((car_train_one_hot,hammer_train_one_hot,
                        hit_train_one_hot,RUN_train_one_hot))
test_features=np.vstack((car_test_features,hammer_test_features,
                         hit_test_features,RUN_test_features))
test_labels=np.vstack((car_test_one_hot,hammer_test_one_hot,
                       hit_test_one_hot,RUN_test_one_hot))
print(train_features.shape,train_labels.shape)
print(test_features.shape,test_labels.shape)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#Hyper prameters
EPOCH=4000#训练的轮数
BATCH_SIZE=294
LR=0.000001
num_classes=4
#set datasetclasses
class BostonDataset(Dataset):

    # 这是构造方法，在构造方法中，传入data_features, data_target
    def __init__(self, data_features, data_target):
        self.data_features = data_features
        self.data_target = data_target
        # 下面一行是必须的，表示调用父类的构造方法
        super(BostonDataset, self).__init__()

    # 必须定义以下方法，传入index，传出对应index的features和target
    def __getitem__(self, index):
        return torch.tensor(self.data_features[index],
                            dtype=torch.float).to(device), torch.tensor(self.data_target[index],
                                                                dtype=torch.float).to(device)

    # 必须定义以下方法，传出数据的数目
    def __len__(self):
        return len(self.data_features)
train_dataset = BostonDataset(train_features, train_labels)
test_dataset = BostonDataset(test_features, test_labels)
train_data_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)
#creat model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        '''
        一般来说，卷积网络包括以下内容：
        1.卷积层
        2.神经网络
        3.池化层
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # --> (10,1,11,2000) batchsize=10
                in_channels=1,  # 单独一行是一个样本，所以设置输入通道数为1
                out_channels=16,  # 输出的图片是几层
                kernel_size=3,  # 代表扫描的区域点为1*3
                stride=1,  # 就是每隔多少步跳一下
                padding=1,  # 边框补全，其计算公式=（kernel_size-1）/2=(3-1)/2=1
            ),  # 2d代表二维卷积           --> (10,16,11,2000)
            nn.BatchNorm2d(16,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
            nn.ReLU(),  # 非线性激活层
            nn.MaxPool2d(kernel_size=3),  # 设定这里的扫描区域为2*2，且取出该2*2中的最大值          --> (10,16,3,666)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(  # --> (10,16,3,666)
                in_channels=16,  # 这里的输入是上层的输出为50层
                out_channels=32,  # 在这里我们需要将其输出为100层
                kernel_size=3,  # 代表扫描的区域点为5*5
                stride=1,  # 就是每隔多少步跳一下
                padding=1,  # 边框补全，其计算公式=（kernel_size-1）/2=(3-1)/2=1
            ),  # --> (10,32,3,666)
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),  # 设定这里的扫描区域为5*5，且取出该5*5中的最大值     --> (10,32,1,222)，这里是三维数据
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,3))#(10,64,1,74)
        )#10,64,40


        self.out1 = nn.Sequential(nn.Linear(64 *1*74, 74),
                                  nn.Dropout(0.2),
                                  nn.ReLU())# 注意一下这里的数据是二维的数据
        self.out2 = nn.Sequential(nn.Linear(74,37),
                                  nn.Dropout(0.2),
                                  nn.ReLU())
        self.out3 = nn.Sequential(nn.Linear(37,num_classes),
                                  nn.Dropout(0.2))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)# （batch,10）
        # 然后接下来进行一下扩展展平的操作，将三维数据转为二维的数据
        x = x.view(x.size(0), -1)  # (batch)
        output1 = self.out1(x)
        output2  = self.out2(output1)
        #output   = torch.sigmoid(self.out3(output2))
        #output   = F.softmax(self.out3(output2))
        output = self.out3(output2)
        return output
    #####def forward 训练过程的整个运行顺序与逻辑
cnn = CNN().cuda()
# 添加优化方法
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
# 指定损失函数使用交叉信息熵
loss_fn = nn.CrossEntropyLoss(reduction='mean')
step = 0
loss_list=[]
correct_list=[]
test_loss_list=[]
test_correct_list=[]
for epoch in range(EPOCH):
    al_loss = 0.0
    al_acc = 0
    al_num = 0
    correct_num = 0
    raw_num = 0
    # 加载训练数据
    cnn.train()
    for step, data in enumerate(train_data_loader):
        x, y = data
        # 分别得到训练数据的x和y的取值
        b_x = Variable(x)
        b_x=b_x.view(b_x.size(0),1,10,2000)
        b_y = Variable(y)
        b_y =b_y.view(b_y.size(0),num_classes)

        output = cnn(b_x)  # 调用模型预测
        loss = loss_fn(output, b_y)  # 计算损失值
        b_x_pred = torch.max(output, 1)[1].data.squeeze()
        b_y_view = torch.max(b_y, 1)[1].data.squeeze()
        correct_train = sum(b_x_pred == b_y_view).item() / b_y_view.size(0)
        al_num += len(data)
        al_loss += loss.item() * len(data)
        al_acc += correct_train * len(data)
        optimizer.zero_grad()  # 每一次循环之前，将梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        # 每执行5次，输出一下当前epoch、loss、accuracy
    loss_ep = al_loss / al_num
    acc_ep = al_acc / al_num
    loss_list.append(loss_ep)
    correct_list.append(acc_ep)
    if (epoch % 1 == 0):
        all_acc = 0
        all_loss = 0
        num = 0
        all_num_2pred=0
        all_num_2raw=0
        cnn.eval()
        with torch.no_grad():
         for step2,data in enumerate(test_data_loader):
            t_x,t_y=data
            test_x=Variable(t_x)
            test_x=test_x.view(test_x.size(0),1,10,2000)
            test_y=Variable(t_y)
            test_y=test_y.view(test_y.size(0),num_classes)

            logging.debug(test_y.size())
        # 计算一下模型预测正确率
            test_output = cnn(test_x)
            loss = loss_fn(test_output, test_y)
            y_pred = torch.max(test_output, 1)[1].data.squeeze()
            test_y=torch.max(test_y,1)[1].data.squeeze()
            accuracy = sum(y_pred == test_y).item() / test_y.size(0)
            num_2_pred=sum(y_pred==0).item()
            num_2_raw=sum(test_y==0).item()

            all_acc += accuracy * len(data)
            all_loss += loss.item() * len(data)
            num += len(data)
            all_num_2pred +=num_2_pred
            all_num_2raw+=num_2_raw


        logging.info(f'now epoch :  , {epoch},    |  loss :  {all_loss/num},      |   accuracy :  , {all_acc/num},      | num_y_pre:, {all_num_2pred},  |   num_y_raw:,   {all_num_2raw}')
        test_loss_list.append(all_loss / num)
        test_correct_list.append(all_acc / num)
plt.figure()
plt.plot(loss_list,'blue')
plt.plot(test_loss_list,'green')
#plt.yticks(my_y_ticks)
plt.show()
plt.figure()
plt.plot(correct_list,'blue')
plt.plot(test_correct_list,'green')
plt.show()