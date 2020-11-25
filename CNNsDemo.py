from dataType import *

train_data = MyDataSet('train',transform=transforms.ToTensor())
train_data_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)

CNN = cnn()


def train():
    optimizer = optim.SGD(CNN.parameters(),lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader,0):
            input, label = data
            optimizer.zero_grad()
            output= CNN(input)[0]
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss/10))
                running_loss = 0.0
    print('Finish training')
    torch.save(CNN, 'CNN.pkl')
    torch.save(CNN.state_dict(), 'CNN_params.pkl')
