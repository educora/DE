from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from PIL import Image


class ImageDetection(object):
    torch_models =['AlexNet',
                   'DenseNet', 
                   'GoogLeNet', 
                   'GoogLeNetOutputs', 
                   'Inception3', 
                   'InceptionOutputs', 
                   'MNASNet', 
                   'MobileNetV2', 
                   'MobileNetV3', 
                   'ResNet', 
                   'ShuffleNetV2',
                   'SqueezeNet', 
                   'VGG', 
                   
                   'alexnet',                    
                   'densenet', 
                   'densenet121', 
                   'densenet161', 
                   'densenet169', 
                   'densenet201', 
                   'detection', 
                   'googlenet', 
                   'inception', 
                   'inception_v3', 
                   'mnasnet',
                   'mnasnet0_5', 
                   'mnasnet0_75', 
                   'mnasnet1_0',
                   'mnasnet1_3',
                   'mobilenet',
                   'mobilenet_v2',
                   'mobilenet_v3_large',
                   'mobilenet_v3_small',
                   'mobilenetv2',
                   'mobilenetv3',
                   'quantization',
                   'resnet',
                   'resnet101',
                   'resnet152',
                   'resnet18',
                   'resnet34',
                   'resnet50',
                   'resnext101_32x8d',
                   'resnext50_32x4d',
                   'segmentation',
                   'shufflenet_v2_x0_5',
                   'shufflenet_v2_x1_0',
                   'shufflenet_v2_x1_5',
                   'shufflenet_v2_x2_0',
                   'shufflenetv2',
                   'squeezenet',
                   'squeezenet1_0',
                   'squeezenet1_1',
                   'utils',
                   'vgg',
                   'vgg11',
                   'vgg11_bn',
                   'vgg13',
                   'vgg13_bn',
                   'vgg16',
                   'vgg16_bn',
                   'vgg19',
                   'vgg19_bn',
                   'video',
                   'wide_resnet101_2', 
                   'wide_resnet50_2']

    def __init__(self,dataset:str,class_labels:str):
        cudnn.benchmark = True     
        self.dataset = dataset
        self.datadir = os.path.join("datasets",self.dataset,"data")
        self.class_labels = os.path.join("datasets",class_labels)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return None

    def load_model(self,model:str,pretrained:bool=True):
        model = torch.hub.load("pytorch/vision:v0.10.0", model, pretrained)
        model.eval()
        return model

    def load_model_from_disk(self,model_path:str):
        model = torch.load(model_path)
        model.eval()
        return model

    def transform(self):        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess
    
    def predict(self,model,image:str):
        image = Image.open(image)
        preprocess = self.transform()
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)            
        out = model(batch_t)
    
        with open(self.class_labels) as f:
            labels = [line.strip() for line in f.readlines()]

        _, index = torch.max(out, 1)
        percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

        return labels[index[0]].strip(), percentage[index[0]].item()

    def train_model(self, model, workers=4, num_epochs=25):
        since = time.time()
        
        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.datadir, x),data_transforms[x]) for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=workers,shuffle=True, num_workers=workers) for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        inputs, classes = next(iter(dataloaders['train']))
    
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        # load a pretrained model
        model_ft = self.load_model(model,pretrained=True)
        num_ftrs = model_ft.fc.in_features

        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)

        model_ft = model_ft.to(self.device)
        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
                
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_acc = 0.0 

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model        


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.waitforbuttonpress(0)

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0    

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    try:
                        loss = criterion(outputs, labels)
                    except:
                        pass

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        try:
                            loss.backward()
                            optimizer.step()
                        except:
                            pass

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def predict_image(model, image):        
    image = Image.open(image)    
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(254),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0)
    
    model.eval()
    out = model(batch_t)
    
    with open('hymenoptera\\data\\class_labels.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    return labels[index[0]].split(",")[1].strip(), percentage[index[0]].item()
    

if __name__ == '__main__':    
    cudnn.benchmark = True    
    project_name = "hymenoptera"    
    plt.ion()   # interactive mode
    # Data augmentation and normalization for training
    # Just normalization for validation
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = os.path.join("datasets","animals",project_name,"data")
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    #imshow(out, title=[class_names[x] for x in classes])
    
    # load a pretrained model
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=25)
    #visualize_model(model_ft, num_images=6)

    torch.save(model_ft,os.path.join(project_name,"model",project_name + "." + "pth"))
    
    # save the class labels in a file
    with open(os.path.join(project_name,"data", "class_labels.txt"),"w+") as f:
        for i in range(0,len(class_names)):
            f.write(str(i) + "," + class_names[i] + "\n")
    """

    # load pretrained model from disk
    #model_ft = torch.load(os.path.join(project_name,"model",project_name + "." + "pth"))
    
    # provided an input image, returns the predicted class
    #p = predict_image(model_ft,"hymenoptera\\data\\val\\ants\\8124241_36b290d372.jpg")
    #p = predict_image(model_ft,"hymenoptera\\data\\val\\bees\\10870992_eebeeb3a12.jpg")
    #print(p)
    #print(dir(models))

    p = ImageDetection("animals\\hymenoptera","animals\\class_labels.txt")
    model = models.resnet18(pretrained=True)
    x = p.predict(model,"datasets\\animals\\hymenoptera\\data\\val\\ants\\8398478_50ef10c47a.jpg")
    
    print(x)
    """
    