import argparse
import torch
import json
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

# This class is based on "Image Classifier Project"
class predict_model():
    def __init__(self):        
        self.image_path = args.image_path
        self.checkpoint_path = args.checkpoint_path
        self.top_k = args.top_k
        self.category_names = args.category_names
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Overwrite device if CPU is specified
        # No checking for CUDA is made here because availability is checked above
        if not args.gpu:
            device = "cpu"
        self.device = device
        
        self.arch = args.arch
        self.cat_to_name = self.label_mapping()
    
    def label_mapping(self):
        with open(self.category_names, 'r') as f:
            return json.load(f)
    
    def load(self):
        checkpoint = torch.load(self.checkpoint_path)
        
        # See: https://gist.github.com/indraniel/da11c4f79c79b5e6bfb8
        func = getattr(models, self.arch)
        model = func(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
        model.state_dict = checkpoint['state_dict']
        model.class_to_idx = checkpoint['class_to_idx']
        model.modules = checkpoint['modules']
        self.model = model
    
    def process_image(self, image_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # DONE: Process a PIL image for use in a PyTorch model
        pil_image = Image.open(image_path)
        # https://stackoverflow.com/questions/26649716/how-to-show-pil-image-in-ipython-notebook
        # Debugging purpose
        #display(pil_image)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(pil_image)
    
    def imshow(self, image, ax=None, title=None):
        """Imshow for Tensor."""
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        return image
    
    def predict(self, image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''

        # DONE: Implement the code to predict the class from an image file
        # Tested only on GPU because of CPU loading time
        model.to("cuda")
        image = self.process_image(image_path)
        image_tensor = torch.tensor(image, dtype=torch.float32)
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to("cuda")
        with torch.no_grad():
            output = torch.exp(model.forward(image_tensor))
        # See: https://www.geeksforgeeks.org/how-to-find-the-k-th-and-the-top-k-elements-of-a-tensor-in-pytorch/
        topk_p, topk_c = torch.topk(output, topk) 
        return topk_p, topk_c
    
    def get_result(self):
        # DONE: Display an image along with the top 5 classes
        # See: "Part 4 - Fashion-MNIST (Solution)"
        p, c = self.predict(self.image_path, self.model)
        p_cpu = p.data.cpu().numpy()[0]
        c_cpu = c.data.cpu().numpy()[0]
        labels = [self.cat_to_name[str(c)] for c in c_cpu]
        self.imshow(self.process_image(self.image_path), title = labels[0])
        # See: https://stackoverflow.com/questions/26984414/efficiently-sorting-a-numpy-array-in-descending-order
        order = np.arange(len(labels))[::-1]
        # Here is a little unkown on how to show the result. Maybe you can give me more details on the output format
        for i in order:
            print(labels[i])
    
    def psychic(self):
        self.load()
        self.get_result()

# See: https://docs.python.org/3/howto/argparse.html
parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to image", type=str)
parser.add_argument("checkpoint_path", help="path to checkpoint", type=str)
parser.add_argument("--top_k", help="top_k", type=int, default=5)
parser.add_argument("--category_names", help="category names path", type=str, default="cat_to_name.json")
parser.add_argument("--gpu", help="gpu", type=bool, default=True)
# Added in order to comply to "train.py" CLI arguments
parser.add_argument("--arch", help="arch", type=str, default="vgg16")
args = parser.parse_args()

model = predict_model()
model.psychic()

