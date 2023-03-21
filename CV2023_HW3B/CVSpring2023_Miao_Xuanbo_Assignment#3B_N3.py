#pylint disable=all
if __name__ == '__main__':
    import torch
    import torchvision
    from torchvision import transforms
    import torch.nn.functional as F
    from torch import optim, nn
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    #test git  
    # Check if CUDA is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define image transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Set batch size
    batch_size = 40
    epoch_num = 4
    disp_interval = 4000/batch_size
    batch_size_show = 8

    folder_path = './CV2023_HW3B/'
    model_path = folder_path+'N3_cifar_net.pth'
    img1_path  = folder_path+'N3_img1.png'
    img2_path  = folder_path+'N3_img2.png'


    # Load training set
    trainset = torchvision.datasets.CIFAR10(
        root=folder_path+'/CIFAR10_data', train=True, download=True, transform=transform)
    # filter out the other classes
    trainset.targets = np.array(trainset.targets)
    trainset.data = trainset.data[trainset.targets < 4]
    trainset.targets = trainset.targets[trainset.targets < 4]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load test set
    testset = torchvision.datasets.CIFAR10(
        root=folder_path+'/CIFAR10_data', train=False, download=True, transform=transform)
    # filter out the other classes
    testset.targets = np.array(testset.targets)
    testset.data = testset.data[testset.targets < 4]
    testset.targets = testset.targets[testset.targets < 4]
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader_show = torch.utils.data.DataLoader(testset, batch_size=batch_size_show, shuffle=False, num_workers=2)

    # Define classes
    classes = ('cat', 'car', 'frog', 'other')

    def im_show(img, save_path=None):
        img = img.cpu()
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show(block=False)   # show the image without blocking the code
        plt.pause(2)            # pause the code execution for 1 second
        plt.close()             # close the image


    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
            self.fc1 = nn.Linear(32* 2 * 2 , 120) 
            self.fc2 = nn.Linear(120, 4) # change the number of output classes to 4

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(-1, 32 * 2 * 2)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    if 1: 
    #if not os.path.exists(model_path):
        # Instantiate the neural network and move it to GPU
        net = Net().to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        # Train the neural network
        for epoch in range(epoch_num):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                labels = labels.long()
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
                if i % disp_interval == disp_interval-1:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        print('Finished Training')
        # Save the trained model
        torch.save(net.state_dict(), model_path)

    # Display test images and labels
    dataiter = iter(testloader_show)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    im_show(torchvision.utils.make_grid(images),img1_path)
    print('GroundTruth: ', ' '.join(
        f'{classes[labels[j]]:5s}' for j in range(batch_size_show)))

    # Load the trained model and predict on test data
    net = Net().to(device)
    net.load_state_dict(torch.load(model_path))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Display predicted labels
    print('Predicted: ', ' '.join(
        f'{classes[predicted[j]]:5s}' for j in range(batch_size_show)))

    # Calculate the accuracy of the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader_show:
            images, labels = data; 
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f} %')

    # Calculate the accuracy of the model for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader_show:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs =net(images)
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

    # create confusion matrix
    classes = ['cat', 'car', 'frog', 'other'] # update classes list
    conf_matrix = np.zeros((4, 4))
    with torch.no_grad():
        for data in testloader_show:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()
                if true_label < 3 and predicted_label < 3: # if true label and predicted label are in the 0-2 range
                    conf_matrix[true_label][predicted_label] += 1
                elif true_label >= 3 and predicted_label >= 3: # if true label and predicted label are in the 3-9 range
                    conf_matrix[3][3] += 1
                else: # if true label is in the 0-2 range and predicted label is in the 3-9 range, or vice versa
                    conf_matrix[true_label%3][3] += 1

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
    plt.savefig(img2_path)
    plt.show(block=False)   # show the image without blocking the code
    plt.pause(2)            # pause the code execution for 1 second
    plt.close()      
