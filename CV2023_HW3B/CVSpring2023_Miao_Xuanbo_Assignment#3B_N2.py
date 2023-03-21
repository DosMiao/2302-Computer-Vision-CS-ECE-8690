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
    import time


    # Check if CUDA is available, else use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define image transformations
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Set batch size
    batch_size = 4
    epoch_num = 4
    disp_interval = batch_size*50
    batch_size_show = 8

    if batch_size<10:
        disp_interval*=2

        if device!='cpu':
            device = torch.device('cpu')
            print(f"Change to use device: {device}"+" because batch size is too small")


    folder_path = './CV2023_HW3B/'
    model_path = folder_path+'N2_cifar_net.pth'
    img1_path  = folder_path+'N2_img1.png'
    img2_path  = folder_path+'N2_img2.png'
    img3_path  = folder_path+'N2_img3.png'

    trainset = torchvision.datasets.CIFAR10(root=folder_path+'/CIFAR10_data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=folder_path+'/CIFAR10_data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader_show = torch.utils.data.DataLoader(testset, batch_size=batch_size_show, shuffle=False, num_workers=2)

    # Define classes
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    

    def im_show(img, save_path=None):
        img = img.cpu()
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        
        if npimg.shape[0]==1:
            npimg=np.reshape(npimg,(npimg.shape[1],npimg.shape[2],npimg.shape[3]))
        
        plt.imshow(np.transpose(npimg,(1, 2, 0)))

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show(block=False)   # show the image without blocking the code
        plt.pause(2)            # pause the code execution for 1 second
        plt.close()             # close the image

    def plot_loss(loss_history, epoch, i):
        plt.plot(loss_history)
        plt.xlabel('Display Interval')
        plt.ylabel('Loss')
        plt.title(f'Loss History - Epoch: {epoch + 1}, Batch: {i + 1}')
        plt.pause(0.001)
        plt.clf()

    def plot_loss_save(loss_history, epoch, i, img_path):
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'Epoch {epoch + 1}, Iteration {i + 1}')
        plt.savefig(img_path)
        plt.close()

    # Define the neural network
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
            self.fc1 = nn.Linear(32* 2 * 2, 120)
            self.fc2 = nn.Linear(120, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(-1, 32 * 2 * 2)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    #if 1: 
    if not os.path.exists(model_path):
                # Instantiate the neural network and move it to GPU
        net = Net().to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        #optimizer = optim.Adam(net.parameters(), lr=0.001)

        loss_history = []
        # Train the neural network
        start_time = time.time()  # Record the start time

        for epoch in range(epoch_num):  # loop over the dataset multiple times
            
            epoch_start_time = time.time() 
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
                if i % disp_interval == disp_interval-1:    # print every disp_interval mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / disp_interval:.3f}')
                    loss_history.append(running_loss / disp_interval)
                    plot_loss(loss_history, epoch, i)
                    running_loss = 0.0
                    
            epoch_elapsed_time = time.time() - epoch_start_time  # Calculate the elapsed time for the epoch
            print(f'Time taken for epoch {epoch + 1}: {epoch_elapsed_time:.2f} seconds')

        print('Finished Training')

        end_time = time.time()  # Record the end time
        total_training_time = end_time - start_time  # Calculate the elapsed time
        print(f'Total training time: {total_training_time:.2f} seconds')

        plot_loss_save(loss_history, epoch, i, img2_path)

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
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # Calculate the accuracy of the model for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader_show:
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
            ax.text(x=j, y=i, s=int(
                conf_matrix[i, j]), va='center', ha='center', size='xx-large')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(img3_path)
    plt.show(block=False)   # show the image without blocking the code
    plt.pause(2)            # pause the code execution for 1 second
    plt.close()             # close the image

    # Calculate worst accuracy class and the class it is most confused with
    worst_accuracy = 100
    worst_class = None
    most_confused_class = None

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_class = classname

    worst_class_index = classes.index(worst_class)
    conf_row = conf_matrix[worst_class_index, :]
    most_confused_class_index = np.argmax(conf_row[np.arange(len(conf_row)) != worst_class_index])
    most_confused_class = classes[most_confused_class_index]

    print(f"The class with the worst accuracy is: {worst_class} with accuracy: {worst_accuracy:.1f}%")
    print(f"The class it is most confused with is: {most_confused_class}")

    # Collect 5 confused images and their true labels
    confused_images = []
    confused_labels = []

    with torch.no_grad():
        for data in testloader:
            if len(confused_images) >= 5:
                break
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if labels[i] == classes.index(worst_class) and predicted[i] == classes.index(most_confused_class):
                    confused_images.append(images[i].cpu())
                    confused_labels.append(labels[i].cpu())
                    if len(confused_images) >= 5:
                        break

    # Display the confused images and their true labels
    print("Confused images:")
    for i, img in enumerate(confused_images):
        image_name = folder_path+ f"N2_confused_image_{i+1}.png"
        im_show(img.unsqueeze(0),image_name)
        print(f"True label: {classes[confused_labels[i]]}")