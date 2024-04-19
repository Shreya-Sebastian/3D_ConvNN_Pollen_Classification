import random, os
import numpy as np
import pickle
import tifffile
from imutils import paths
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

from models import ConvNet3D, ResNet3D  

class Param:
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = '/content/drive/MyDrive/pollen_classification/pollen_raw_data'
    save_dir = '/content/drive/MyDrive/pollen_classification/models'

    model = ResNet3D
    model_name = 'ResNet3D_20_frames'

    # Create path to store model.pth and results.pkl 
    results_path = os.path.join(save_dir, model_name)
    os.makedirs(results_path, exist_ok=True)

    height = 224
    width = 224
    aug_threshold = 0.50        # threshold for whether data in train_data is augmented
    max_rotation_angle = 100   # Maximum rotation angle for data augmentation
    train_batch_size = 16
    valid_test_batch_size = 16
    learning_rate = 0.0001
    num_epochs = 30
    num_frames = 20            # frames in the image
    
def make_datasets(param):
    # Retrieve image paths
    img_paths = list(paths.list_images(param.data_dir))
    random.seed(30)
    random.shuffle(img_paths)
    img_data = []
    labels = []
    # Loop over all input images
    for img_path in img_paths:
        # Load image, pre-process it, and store it in the list
        img = tifffile.imread(img_path) 
        label = img_path.split(os.path.sep)[-2]
        # Calculate the required padding for each dimension
        pad_height = max(param.height - img.shape[1], 0)
        pad_width = max(param.width - img.shape[2], 0)
        # Pad the image with zeros on all sides
        pad_values = ((0, 0), (0, pad_height), (0, pad_width))
        padded_img = np.pad(img, pad_values, mode='constant')
        # Crop the padded image to the desired size (if image is bigger than desired HxW)
        cropped_img = padded_img[:, :param.height, :param.width]
        # Determine the start and end indices for the middle frames
        middle_start = (img.shape[0] - param.num_frames) // 2
        middle_end = middle_start + param.num_frames
        # Select the middle frames
        middle_frames = cropped_img[middle_start:middle_end]

        img_array = np.array(middle_frames)
        img_data.append(img_array)
        labels.append(label)

    # Convert data and labels to numpy arrays
    img_data = np.array(img_data)
    labels = np.array(labels)
    
    # Map class labels to an integer
    class_labels = ['P_judaica_P_officinalis', 'U_dioica_U_urens', 'U_membranacea']
    label_map = {class_label: idx for idx, class_label in enumerate(class_labels)}
    # Convert all labels to integer labels 
    integer_labels = [label_map[label] for label in labels]

    dataset = []
    for i in range(len(img_data)):
        dataset.append((img_data[i], integer_labels[i], False))   # Third dimension = augmentation boolean (False by default)

    # split the dataset into 90% training and validation data and 10% validation data
    train_val_set, test_set = train_test_split(dataset, test_size=0.1, random_state=42)
    # split the train_val_set into 90% training data and 10% validation data
    train_set, valid_set = train_test_split(train_val_set, test_size=0.1, random_state=42)

    return train_set, valid_set, test_set

# Create datasets and apply data augmentation
class Datasets(Dataset):
    def __init__(self, param, data):
        self.data = data
        self.height = param.height
        self.width = param.width
        self.num_frames = param.num_frames
        self.max_rotation_angle = param.max_rotation_angle
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        image = self.data[idx][0]
        label = self.data[idx][1]
        img_tensor = self.augmentation(img = image, augment = self.data[idx][2])
        # Rescale to (-1, 1)
        img_tensor = (img_tensor - 0.5) * 2     # [0, 1] --> -0.5 --> [-0.5, 0.5] --> apply * 2 --> [-1, 1]
        label_tensor = torch.tensor(label, dtype= torch.long)

        return img_tensor, label_tensor
    
    def augmentation(self, img, augment):
        transform = transforms.Compose([
            transforms.ToTensor(),                              # Convert frames to tensors
            # Augmentation
                transforms.RandomHorizontalFlip(p = 0.8 if augment else 0),             # Randomly apply horizontal flip with probability 0.8
                transforms.RandomVerticalFlip(p = 0.8 if augment else 0),
                transforms.RandomRotation(self.max_rotation_angle if augment else 0), 
        ])

        # Initialize aug_image tensor
        aug_image = torch.empty(self.num_frames, 3, self.height, self.width)
        # Apply transformation to each frame of stack and save in aug_image
        for i in range(self.num_frames):
            aug_image[i, :, :, :] = transform(img[i, :, :])   

        return aug_image

# Dataloader
def dataloaders(param):

    train_dataset, valid_dataset, test_dataset = make_datasets(param)
    
    # Create list of data to be augmentated and add to training set
    aug_dataset = []
    for data in train_dataset:
        if random.random() >= param.aug_threshold:        # aug_threshold of 1.1 means no augmentation
            aug_dataset.append((data[0], data[1], True))
    aug_train_dataset = train_dataset + aug_dataset

    aug_train = Datasets(param, aug_train_dataset)
    valid = Datasets(param, valid_dataset)
    test = Datasets(param, test_dataset)
    
    aug_train_loader = DataLoader(aug_train, batch_size=param.train_batch_size, shuffle=True)       # shuffeled to help training process
    valid_loader = DataLoader(valid, batch_size=param.valid_test_batch_size, shuffle=False)     
    test_loader = DataLoader(test, batch_size=param.valid_test_batch_size, shuffle=False)     

    return aug_train_loader, valid_loader, test_loader

# Function to evaluate model and calculate metrics
def evaluate(model, dataloader, device):
    
    y_targets = []
    y_preds = []
    batch_loss_list = []
    model.eval()                            # model in evaluation mode
    with torch.no_grad():                   # no need for gradient calculation during evaluation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)       # Move inputs to the GPU
            targets = targets.to(device)     # Move targets to the GPU
            logits = model(inputs)       
            probs = F.softmax(logits, dim=1)
            _, labels = torch.max(probs, dim=1)
            loss = nn.CrossEntropyLoss()(logits, targets)
            batch_loss_list.append(loss.item())
            # Save targets and predictions
            y_targets.extend(targets.to('cpu').numpy().tolist())
            y_preds.extend(labels.to('cpu').numpy().tolist())
    
    y_targets = np.array(y_targets)
    y_preds = np.array(y_preds)

    # Calculate accuracy, loss and f1 scores 
    acc = accuracy_score(y_targets, y_preds)
    loss = np.mean(batch_loss_list)
    f1 = f1_score(y_targets, y_preds, average = 'weighted')
    
    return f1, acc, loss

# Function to train model 
def trainer(param, model, aug_train_loader, valid_loader, optimizer): 

    # Produce summary of the network
    summary(model, (param.num_frames, param.train_batch_size, param.height, param.width))

    # Function to save current model 
    def save_checkpoint(model, epoch):
        save_path = os.path.join(param.results_path, 'model.pth')
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }
        torch.save(checkpoint, save_path)
        print("Model and Results saved")
        
    def eval_results(best_acc, results, epoch):             
        
        # Evaluation
        train_f1, train_acc, train_loss = evaluate(model, aug_train_loader, device) 
        valid_f1, valid_acc, valid_loss = evaluate(model, valid_loader, device)
        
        # Append results
        results.append((epoch, train_loss, valid_loss, train_f1, valid_f1, train_acc, valid_acc))
        
        # Save best model so far 
        if valid_acc > best_acc:
            save_checkpoint(model, epoch)
            best_acc = valid_acc  # Update best_acc
        
        # Save results
        with open(os.path.join(param.results_path, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)

        # Show results
        print(f''' {epoch + 1}/{param.num_epochs}  \n
                loss      --->     train: {train_loss}     valid: {valid_loss} \n
                f1 score  --->     train: {train_f1}       valid: {valid_f1}\n
                accuracy  --->     train: {train_acc}      valid: {valid_acc} \n''')    

        return best_acc   

    # Initialize best accuracy and results list
    best_acc = 0
    results = []
    device = param.device

    train_loss, valid_loss, train_f1, valid_f1 = 0, 0, 0, 0
    for epoch in range(param.num_epochs):
        model.train()                       # put model in train mode
        # Loading bar
        que = tqdm(enumerate(aug_train_loader), total=len(aug_train_loader), ncols=160)
        for i, (images, targets) in que:
            
            logits = model(images.to(device))            # Forward pass    
            targets = targets.to(device)
            loss = nn.CrossEntropyLoss()(logits, targets)
            loss = loss
            loss.backward()                                    # backward pass
            optimizer.step()                                   # step
            optimizer.zero_grad()                              # zero grad

        # Evaluate train and valid sets and append to results
        best_acc = eval_results(best_acc, results, epoch)      # Save best accuracy
            
    return results

def train(param):
    
    # define model
    model = param.model(param)
    model.to(param.device)
    
    # Optmizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = param.learning_rate, weight_decay=0.01)
    # Dataloaders
    aug_train_loader, valid_loader, test_loader = dataloaders(param)     

    results = trainer(param, model, aug_train_loader, valid_loader, optimizer)     
    
    ### SAVE RESULTS
    with open(os.path.join(param.results_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

def test(param):

    _, _, test_loader = dataloaders(param)
    model_path = os.path.join(param.results_path, 'model.pth')
    loaded_data = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Extract the model's state dictionary from the loaded data
    model_state_dict = loaded_data['model_state_dict']
    model = param.model(param)
    model.load_state_dict(model_state_dict)
    model.to(param.device)
    model.eval()
      
    f1, acc, loss = evaluate(model, test_loader, param.device)

    print(f'''\n  Test metrics for {param.model_name}:\n    
                loss      --->     {loss} \n
                f1 score  --->     {f1}   \n
                accuracy  --->     {acc}  \n''')    

# test(Param)