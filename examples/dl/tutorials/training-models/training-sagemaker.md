---
description: How to Train models on AWS SageMaker using Deep Lake datasets
---

# Training on AWS SageMaker

## How to Train an PyTorch Image Classification Model on AWS SageMaker Using Deep Lake Datasets

[AWS SageMaker](https://aws.amazon.com/sagemaker/) provides scalable infrastructure for developing, training, and deploying deep learning models. In this tutorial, we demonstrate how to run SageMaker training jobs for training a PyTorch image classification model using a Deep Lake dataset. This tutorial will focus on the SageMaker integration, and less so on the details of the training (see other [training tutorials](./) for details)

### Dataset

In this tutorial we will use the [Stanford Cars Dataset](https://app.activeloop.ai/activeloop/stanford-cars-train), which classifies the make+model+year of various vehicles. Though the dataset contains bounding boxes, we ignore those and only use the data for classification purposes.

### Running the Sagemaker Job

We run the SageMaker job using the docker container below that can be found among [these deep learning containers provided by AWS](https://github.com/aws/deep-learning-containers/blob/master/available\_images.md).&#x20;

`"763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker"`

The SageMaker job is initiated using the script below. By also running this script in a SageMaker notebook, the permissions and role access are automatically taken care of within the AWS environment.

```python
import sagemaker

sess = sagemaker.Session()
role = sagemaker.get_execution_role()
```

The training script (`entry_point`) and the directory (`source_dir`) containing the training script and `requirements.txt` file is passed to the [`Estimator`](https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator). The `argparse` parameters for the training script are passed via the `hyperparameters` dictinary in the `Estimator.` Note that we also pass the Deep Lake paths to the training and validation datasets via this input.

```python
 estimator = sagemaker.estimator.Estimator(
                source_dir = "./train_code",  # Directory of the training script
                entry_point = "train_cars.py", # File for the training script    
                image_uri = image_name,
                role = role,
                instance_count = 1,
                instance_type = instance_type,
                output_path = output_path,
                sagemaker_session = sess,
                max_run = 2*60*60,
                hyperparameters = {"train-dataset": "hub://activeloop/stanford-cars-train",
                                   "val-dataset": "hub://activeloop/stanford-cars-test",
                                   "batch-size": 64, "num-epochs": 40,
                                })
```

The training job is triggered using the command below. Typically, the `.fit()` function accepts as inputs the S3 bucket containing the training data, which is then downloaded onto the local storage of the SageMaker job. Since we've passed the Deep Lake dataset paths via the `hyperparameters`, and since Deep Lake does not require data to be downloaded prior to training, we skip these inputs. &#x20;

```
estimator.fit()
```

SageMaker offers a variety of method for advanced data logging. In this example, we can monitor the training performance in real-time in the training notebook where the jobs are triggered, or in the [CloudWatch](https://aws.amazon.com/cloudwatch/) logs for each job. We observe that the validation accuracy after 40 epochs is 75%.

### Training Script

The contents of the `train_code` folder, as well as the `train_cars.py` file, are shown below. The training script follow the same workflow as other PyTorch training workflows using Deep Lake. As mentioned above, the inputs to the `argparse` function are those from the `hyperparameters` inputs in the `estimator`.&#x20;

```python
import deeplake
import argparse
import logging
import os
import sys
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms, models

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


#----------- Define transformations and their parameters -----------#

WIDTH = 320
HEIGHT = 320

tform_train = transforms.Compose([
#     transforms.ToPILImage(), # Not needed because decode_method is set to PIL in the dataloader
    transforms.RandomResizedCrop((WIDTH, HEIGHT), scale=(0.75, 1.0), ratio=(0.75, 1.25)),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=(-0.1,0.1)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)), # Adjust tensor if the image is grayscale
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
    
tform_val = transforms.Compose([
#     transforms.ToPILImage(), # Not needed because decode_method is set to PIL in the dataloader
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1)), # Adjust tensor if the image is grayscale
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])    




#----------- Define helper functions -----------#

# Helper function for loading the model
def get_model_classification(num_classes):
    # Load a pre-trained classification model
    model = models.resnet34(pretrained=True)
    
    # Adjust the fully connected layer based on the number of classes in the dataset
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model


# Helper function for training for 1 epoch
def train_one_epoch(model, optimizer, criterion, data_loader, device, log_interval):
        
    # Set the model to train mode    
    model.train()
    
    # Zero the performance stats for each epoch
    running_loss = 0.0
    start_time = time.time()
    total = 0
    correct = 0
    
    for i, data in enumerate(data_loader):
        
        # Parse the inputs
        inputs = data['images']
        labels = data['car_models'][:, 0] # Get rid of the extra axis

        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        # Update the accuracy for the epoch
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        # Print statistics
        running_loss += loss.item()
        batch_time = time.time()
        
        if i % log_interval == 0:    # print every 100 mini-batches
            speed_cumulative = (i+1)/(batch_time-start_time)
            
            logger.debug('[%5d] running loss: %.3f, epoch accuracy: %.3f, cumulative speed: %.2f ' %
                    (i, running_loss, accuracy, speed_cumulative))

            running_loss = 0.0
      
    
# Helper function for testing the model      
def test_model(model, data_loader, device):
    
    # Set the model to eval mode
    model.eval()
    
    total = 0
    correct = 0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            
            # Parse the inputs
            inputs = data['images']
            labels = data['car_models'][:, 0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs.float())

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
    
    return accuracy
        
    
# Helper function for saving the model    
def save_model(model, model_dir):
    logger.info("Saving the model")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.state_dict(), path)

    
def train(args):
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Load dataset and create dataloaders.
    ds_train = deeplake.load(args.train_dataset, read_only = True, token = args.token, creds = args.creds)
    ds_val = deeplake.load(args.val_dataset, read_only = True, token = args.token, creds = args.creds)
    
    train_loader = ds_train.dataloader()\
                            .batch(args.batch_size)\
                            .shuffle(args.shuffle)\
                            .transform(transform = {'images': tform_train, 'car_models': None})\
                            .pytorch(num_workers = args.num_workers, decode_method = {'images': 'pil'})

    val_loader = ds_val.dataloader()\
                        .batch(args.batch_size)\
                        .transform(transform = {'images': tform_val, 'car_models': None})\
                        .pytorch(num_workers = args.num_workers, decode_method = {'images': 'pil'})
    
    # Load the model
    model = get_model_classification(len(ds_train.car_models.info.class_names))
    model = model.to(device)

    # Define the optimizer, loss, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
        
    # Run the training
    for epoch in range(args.num_epochs):
        logger.debug("Training Epoch: {}".format(epoch))
        
        train_one_epoch(model, optimizer, criterion, train_loader, device, args.log_interval)
        lr_scheduler.step()

        accuracy = test_model(model, val_loader, device)
        logger.debug("Validation Accuracy: {}".format(accuracy))
                
    logger.debug('Finished Training')
    
    save_model(model, args.model_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train-dataset",
        type=str,
        required=True,
        help="path to deeplake training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        type=str,
        required=True,
        help="path to deeplake validation dataset",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for the dataloaders (default: 8)",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="shuffling for the training dataloader (default: True)",
    )    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status (default: 10)",
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None, 
        help="token for accessing the Deep Lake dataset (default: None)"
    )
    parser.add_argument(
        "--creds", 
        type=dict, 
        default=None, 
        help="creds dictionary for accessing the Deep Lake dataset (default: None)"
    )
    parser.add_argument(
        '--model_dir', 
        type=str, 
        default=os.environ['SM_MODEL_DIR'])
        
    train(parser.parse_args())
```

{% file src="../../../../.gitbook/assets/train_code.zip" %}

Congrats! You're now able to train models using AWS SageMaker Jobs while streaming Deep Lake Datasets! 🎉
