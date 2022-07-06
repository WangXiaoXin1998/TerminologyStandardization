from common.Data2ID import getData2ID, getCodeID
from model.config import hyper
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from model.Lstm import Lstm
import time

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(hyper['gpu_id'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if hyper['bert'] == False:
    X_train, y_train = getData2ID(hyper['train_path'])
    X_valid, y_valid = getData2ID(hyper['valid_path'])
    code = getCodeID()
    X_train = (Variable(torch.tensor(X_train)).long()).to(device)
    y_train = (Variable(torch.tensor(y_train)).long()).to(device)
    X_valid = (Variable(torch.tensor(X_valid)).long()).to(device)
    y_valid = (Variable(torch.tensor(y_valid)).long()).to(device)
    code = (Variable(torch.tensor(code)).long()).to(device)
else:
    from common.Data2ID import getData2ID_Bert,getCodeID_Bert
    X_train, y_train = getData2ID_Bert(hyper['train_path'])
    X_valid, y_valid = getData2ID_Bert(hyper['valid_path'])
    code = getCodeID_Bert()
    X_train = (Variable(torch.tensor(X_train)).long())
    y_train = (Variable(torch.tensor(y_train)).long())
    X_valid = (Variable(torch.tensor(X_valid)).long())
    y_valid = (Variable(torch.tensor(y_valid)).long())
    code =  (Variable(torch.tensor(code)).long()).to(device)

if hyper['bert'] == False:
    net = Lstm()
else:
    from model.Bert import Bert
    net = Bert()
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=hyper['learning_rate'])  # 创建优化器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True,
                                                       threshold=0.0001,
                                                       threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

print('-------------------------   hyper   ------------------------------')
print(hyper)
epoch = hyper['epoch']

for i in range(epoch):
    print('-------------------------   training   ------------------------------')
    time0 = time.time()
    batch = 0
    ave_loss, num = 0, 0

    for j in range(len(X_train)):
        batch_x, batch_y = X_train[j], y_train[j]
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_y = torch.unsqueeze(batch_y, dim=0)
        net.train()

        optimizer.zero_grad()  # 清空梯度缓存

        output = net(batch_x, code)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()  # 更新权重

        ave_loss += loss
        batch += 1
        if batch % 100 == 0:
            print('batch: {}/{}, train_loss: {:.5}, time:{:.5}'.format(batch, len(X_train), ave_loss / batch,
                                                                     time.time() - time0))

    scheduler.step(ave_loss)
    print('------------------ epoch:{} ----------------'.format(i + 1))
    print('train_loss: {:.5}, time: {:5}, learning_rate: {:.7}'.format(ave_loss/len(X_train), time.time() - time0,
                                                                                optimizer.param_groups[0]['lr']))
    print('============================================')

    time0 = time.time()
    if (i + 1) % 2 == 0:
        print('-------------------------    valid     ------------------------------')
        torch.save(net.state_dict(), 'save_model/params' + str(i + 1) + '.pkl')
        P = 0
        for j in range(len(X_valid)):
            batch_x, batch_y = X_valid[j], y_valid[j]
            if hyper['bert'] == True:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            net.eval()
            with torch.no_grad():
                output = net(batch_x, code)
            output = torch.argmax(output, dim=1)
            if output[0] == batch_y:
                P += 1
        print('valid_p:{:.5}, time: {:.5}'.format(P / len(X_valid), time.time()-time0))
        print('============================================'.format(i + 1))
