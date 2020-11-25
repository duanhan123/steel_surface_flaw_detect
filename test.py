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
    test_dataiter = iter(val_data_loader)
    image, label = test_dataiter.next()
    output = CNNnet(image)[0]
    _, predict = torch.max(output.data, 1)
    # imshow(torchvision.utils.make_grid(image, nrow=1))
    # print(len(image))
    for i in range(len(image)):
        plt.imshow(np.transpose(image[i],(1,2,0)))
        plt.title('Groundtruth: %d, predict: %d' % (predict[i]+1,label[i]+1))
        plt.show()
    # print('GroundTruth: ', " ".join('%5s' % label[j] for j in range(10)))

    # print(predict == label)
    # print('Predicted: ', " ".join('%5s' % predict[j] for j in range(10)))

import time
def acc_cal(test_num=100):
    startTime = time.time()
    val_data = MyDataSet('test', transform=transforms.ToTensor())
    val_data_loader = data.DataLoader(val_data, batch_size=test_num, num_workers=4, shuffle=True)
    CNNnet = reload()
    acc_dataiter = iter(val_data_loader)
    image, label = acc_dataiter.next()
    output = CNNnet(image)[0]
    _, pred = torch.max(output.data, 1)
    print("accuracy rate: %.2f%%" % ((pred == label).sum() / float(test_num) * 100))
    endTime = time.time()
    print("测试用时： %d s" % (endTime - startTime))