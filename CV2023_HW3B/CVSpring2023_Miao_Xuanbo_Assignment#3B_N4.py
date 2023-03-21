# pylint: disable=all
if __name__ == '__main__':
    import torch
    import torchvision
    from torchvision import transforms
    import torch.nn.functional as F
    from torch import optim, nn
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 40
    epoch_num = 5
    disp_interval = 4000/batch_size
    batch_size_show = 8

    folder_path = './CV2023_HW3B'
    model_path = folder_path+'/cifar_net_N4.pth'

    trainset = torchvision.datasets.CIFAR10(root=folder_path+'/CIFAR10_data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=folder_path+'/CIFAR10_data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader_show = torch.utils.data.DataLoader(testset, batch_size=batch_size_show, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img.cpu()
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
            self.bn3 = nn.BatchNorm2d(128)
            self.conv4 = nn.Conv2d(128, 256, 5, padding=2)
            self.bn4 = nn.BatchNorm2d(256)
            self.pool = nn.MaxPool2d(2, 2)
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(256 * 2 * 2, 512)
            self.fc2 = nn.Linear(512, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(self.dropout(x)))))
            x = self.pool(F.relu(self.bn2(self.conv2(self.dropout(x)))))
            x = self.pool(F.relu(self.bn3(self.conv3(self.dropout(x)))))
            x = self.pool(F.relu(self.bn4(self.conv4(self.dropout(x)))))
            
            x = x.view(-1, 256 * 2 * 2) # Update the size for reshaping
            x = F.relu(self.fc1(self.dropout(x)))
            x = self.fc2(x)
            return x

    if 1:
        net = Net().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

        for epoch in range(epoch_num):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % disp_interval == disp_interval - 1:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / disp_interval:.3f}')
                    running_loss = 0.0
            scheduler.step(loss)

        print('Finished Training')
        torch.save(net.state_dict(), model_path)

    dataiter = iter(testloader_show)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_show)))

    net = Net().to(device)
    net.load_state_dict(torch.load(model_path))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size_show)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader_show:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader_show:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    conf_matrix = np.zeros((10, 10))
    with torch.no_grad():
        for data in testloader_show:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                conf_matrix[labels[i]][predicted[i]] += 1
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=int(conf_matrix[i, j]), va='center', ha='center', size='xx-large')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show(block=False)   # show the image without blocking the code
    plt.pause(2)            # pause the code execution for 1 second
    plt.close()             # close the image