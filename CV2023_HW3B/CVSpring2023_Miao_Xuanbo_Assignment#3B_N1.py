if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    import torch
    import torchvision
    from torchvision import transforms
    import torch.nn.functional as F
    from torch import optim, nn
    import matplotlib.pyplot as plt
    import numpy as np

    # Check if CUDA is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Define image transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Set batch size
    batch_size = 4

    # Load training set
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load test set
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Function to display images
    def imshow(img):
        img = img.cpu()
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show(block=False)   # show the image without blocking the code
        plt.pause(2)            # pause the code execution for 1 second
        plt.close()             # close the image

    # Get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Display images and labels
    imshow(torchvision.utils.make_grid(images))
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Instantiate the neural network and move it to GPU
    net = Net().to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the neural network
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Display test images and labels
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    # Load the trained model and predict on test data
    net = Net().to(device)
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Display predicted labels
    print('Predicted: ', ' '.join(
        f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

    # Calculate the accuracy of the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Calculate the accuracy of the model for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    conf_matrix = np.zeros((10, 10))
    with torch.no_grad():
        for data in testloader:
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
            ax.text(x=j, y=i, s=int(
                conf_matrix[i, j]), va='center', ha='center', size='xx-large')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.show(block=False)   # show the image without blocking the code
    plt.pause(2)            # pause the code execution for 1 second
    plt.close()             # close the image
