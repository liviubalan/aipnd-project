import argparse
from torchvision import datasets, transforms, models
import torch
import json
from torch import nn
from torch import optim

# This class is based on "Image Classifier Project"
class train_model():
    def __init__(self):
        self.data_dir = args.data_dir.rstrip("/")
        self.train_dir = self.data_dir + '/train'
        self.valid_dir = self.data_dir + '/valid'
        self.test_dir = self.data_dir + '/test'
        
        self.save_dir = args.save_dir.rstrip("/")
        self.arch = args.arch
        self.learning_rate = args.learning_rate
        self.hidden_units = args.hidden_units
        self.epochs = args.epochs
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Overwrite device if CPU is specified
        # No checking for CUDA is made here because availability is checked above
        if not args.gpu:
            device = "cpu"
        self.device = device
        
        self.cat_to_name = self.label_mapping()
        
    def transform(self):
        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        valid_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        # DONE: Load the datasets with ImageFolder
        #image_datasets = 
        self.train_data = datasets.ImageFolder(self.train_dir, transform=train_transforms)
        self.valid_data = datasets.ImageFolder(self.valid_dir, transform=valid_transforms)
        self.test_data = datasets.ImageFolder(self.test_dir, transform=test_transforms)

        # DONE: Using the image datasets and the trainforms, define the dataloaders
        #dataloaders = 
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.validloader = torch.utils.data.DataLoader(self.valid_data, batch_size=64)
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size=64)
    
    def label_mapping(self):
        with open('cat_to_name.json', 'r') as f:
            return json.load(f)
    
    def build(self):
        # See: https://gist.github.com/indraniel/da11c4f79c79b5e6bfb8
        func = getattr(models, self.arch)
        self.model = func(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.classifier = nn.Sequential(nn.Linear(25088, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(512, 102),
                                         nn.LogSoftmax(dim=1))

        self.criterion = nn.NLLLoss()

        # Only train the classifier parameters, feature parameters are frozen
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)

        self.model.to(self.device)
        
    def test(self):
        steps = 0
        running_loss = 0
        print_every = 5
        for epoch in range(self.epochs):
            for inputs, labels in self.trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.testloader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{self.epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(self.testloader):.3f}.. "
                          f"Test accuracy: {accuracy/len(self.testloader):.3f}")
                    running_loss = 0
                    self.model.train()
    
    def save(self):
        checkpoint_path = self.save_dir + '/checkpoint.pth'
        self.model.class_to_idx = self.train_data.class_to_idx
        checkpoint = {'classifier': self.model.classifier,
                      'state_dict': self.model.state_dict(),
                      'class_to_idx': self.model.class_to_idx,
                      'modules': self.model.modules}
        torch.save(checkpoint, checkpoint_path)
    
    def train(self):
        self.transform()
        self.build()
        self.test()
        self.save()
        #self.debug()
    
    def debug(self):
        print(self.data_dir, self.save_dir, self.train_dir, self.valid_dir, self.test_dir, self.arch, self.learning_rate, self.hidden_units, self.epochs, self.device, self.model)

# See: https://docs.python.org/3/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="data directory", type=str)
parser.add_argument("--save_dir", help="save directory", type=str, default=".")
parser.add_argument("--arch", help="arch", type=str, default="vgg16")
parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.003)
parser.add_argument("--hidden_units", help="hidden units", type=int, default=512)
parser.add_argument("--epochs", help="epochs", type=int, default=1)
parser.add_argument("--gpu", help="gpu", type=bool, default=True)
args = parser.parse_args()

model = train_model()
model.train()

