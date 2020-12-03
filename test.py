from dataType import *


def reload():
    CNNnet = torch.load('CNN.pkl')
    return CNNnet

def imshow(img):
    nimg = img.numpy()
    plt.imshow(np.transpose(nimg, (1,2,0)))
    plt.show()

def display(num=10):
    val_data = MyDataSet('test', transform=transforms.ToTensor())
    val_data_loader = data.DataLoader(val_data, batch_size=num, num_workers=4, shuffle=True)
    CNNnet = reload()
    if(use_gpu):
        CNNnet = CNNnet.cpu()
    test_dataiter = iter(val_data_loader)
    image, label = test_dataiter.next()
    output = CNNnet(image)[0]
    _, predict = torch.max(output.data, 1)
    for i in range(len(image)):
        fig = plt.figure()
        mngr = plt.get_current_fig_manager()
        mngr.window.wm_geometry("+380+310")
        plt.imshow(np.transpose(image[i],(1,2,0)))
        plt.title('Groundtruth: %d, predict: %d' % (predict[i]+1,label[i]+1))
        plt.show()

import time
def acc_cal(test_num=100):
    startTime = time.time()
    val_data = MyDataSet('test', transform=transforms.ToTensor())
    val_data_loader = data.DataLoader(val_data, batch_size=test_num, num_workers=4, shuffle=True)
    CNNnet = reload()
    if(use_gpu):
        CNNnet = CNNnet.cpu()
    acc_dataiter = iter(val_data_loader)
    image, label = acc_dataiter.next()
    output = CNNnet(image)[0]
    _, pred = torch.max(output.data, 1)
    print("accuracy rate: %.2f%%" % ((pred == label).sum() / float(test_num) * 100))
    endTime = time.time()
    print("test time： %d s" % (endTime - startTime))