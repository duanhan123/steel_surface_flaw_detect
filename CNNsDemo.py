from dataType import *

train_data = MyDataSet('train',transform=transforms.ToTensor())
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
# print(len(train_data_loader))

CNN = cnn()
if(use_gpu):
    CNN = cnn().cuda()


def train():
    optimizer = optim.SGD(CNN.parameters(),lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    L = []
    if(use_gpu):
        criterion = criterion.cuda()
    for epoch in range(num_epochs):
        running_loss = 0.0
        input_num = 0
        for i, data in enumerate(train_data_loader,0):
            input, label = data
            input_num += len(input)
            if(use_gpu):
                input, label = input.cuda(), label.cuda()
                # print(type(input))
            optimizer.zero_grad()
            output = CNN(input)[0]
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            if(use_gpu):
                loss = loss.cpu()
                input  =input.cpu()
                label = label.cpu()
            running_loss += loss.item()
            # if i % 10 == 0:
            #     print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss/10))
            #     running_loss = 0.0
        running_loss = running_loss / input_num
        print("the %d epoch Loss: %.3f" % (epoch,running_loss))
        L.append(running_loss)
    # plt.plot(range(num_epochs), L,'r')
    # plt.show()

    print('Finish training')
    torch.save(CNN, 'CNN.pkl')
    torch.save(CNN.state_dict(), 'CNN_params.pkl')
