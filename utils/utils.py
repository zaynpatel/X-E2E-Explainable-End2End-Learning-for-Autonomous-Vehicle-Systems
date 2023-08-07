import os
import torch
import torchvision
import torch.nn.functional as F
from time import time
import multiprocessing as mp
import numpy as np
import preprocess

def pad_images(image, height=448, width=608):

    """
    When using the dataset dimensions from https://github.com/YassineYousfi/comma10k-baseline/tree/main, I encounter the following error:
    
        RuntimeError: Wrong input shape height=437, width=582. Expected image height and width divisible by 32. Consider pad your images to shape (448, 608).

    This function just serves as a way for me to pad my images to the desire dimensions needed to run the autoencoder.

    """

    h, w  = image.size(1), image.size(2)

    pad_height = max(height - h, 0)
    pad_width = max(width - w, 0)

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Perform the padding
    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom))

    return image


def optimalWorkers(width=582, height=437):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    dataset = preprocess.CommaPreprocess(width=width, height=height, transform=transform, target_transform=None)

    for num_workers in range(4, mp.cpu_count(), 2):  
        passTime = 100000000
        train_loader = torch.utils.data.DataLoader(dataset,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()

        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
            print("Epoch #", epoch)

        end = time()
        if (end-start) < passTime: 
            passTime = end - start
            cores = num_workers
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    return cores

def train_model(model, data_loader, val_loader, epochs, steps_per_epoch, device, optim, iou, dice, precision, recall):
    outputs = []
    highest_dice = 0.0
    highest_iou = 0.0
    highest_prec = 0.0
    highest_rec = 0.0
    for epoch in range(epochs):
        print('-'*20)
        for i, (img, annotation) in enumerate(data_loader):
            img = img.to(device)
            annotation = annotation.to(device)
            
            output = model(img)
            iou_loss = iou(output, annotation)
            dice_loss = dice(output, annotation)
            precision_met = precision(output, annotation)
            recall_met = recall(output, annotation)
            
            optim.zero_grad()
            iou_loss.backward()
            optim.step()

            if highest_iou < 1-iou_loss.item():
                highest_iou = 1-iou_loss.item()

            if highest_dice < dice_loss:
                highest_dice = dice_loss

            if highest_prec < precision_met:
                highest_prec = precision_met

            if highest_rec < recall_met:
                highest_rec = recall_met    
            
            if (int(i+1))%(steps_per_epoch//5) == 0:
                print(f"epoch {epoch+1}/{epochs}, step {i+1}/{steps_per_epoch}, IoU score = {1-iou_loss.item():.4f}, Precision = {precision_met:.4f}, Recall = {recall_met:.4f}, F1/Dice score: {dice_loss:.4f}")
                
        model.eval()
        for img, annotation in val_loader:
            img = img.to(device)
            annotation = annotation.to(device)

            output = model(img)
            iou_loss = iou(output, annotation)
            dice_loss = dice(output, annotation)
            precision_met = precision(output, annotation)
            recall_met = recall(output, annotation)
        print(f"validation loss: IoU score = {1-iou_loss.item():.4f}, Precision = {precision_met:.4f}, Recall = {recall_met:.4f}, F1/Dice score: {dice_loss:.4f}")
        outputs.append((img, annotation, output))
    print("-"*20)
    print(f"highest values, IoU score = {highest_iou:.4f}, Precision = {highest_prec:.4f}, Recall = {highest_rec:.4f}, F1/Dice score: {highest_dice:.4f}")

    return model, outputs

#VARIABLES FOR ENCODING AND DECODING - MODIFY CLASSES HERE
valid = [1, 2, 3, 4, 5, 6]
classes = ['road', 'lane', 'driveable', 'movable', 'my car', 'movable in my car']
colors = [[64, 32, 32],
          [255, 0, 0],
          [128, 128, 96],
          [0, 255, 102],
          [204, 0, 255],
          [190, 204, 255]]

def numClasses():
    return len(valid)

colorMap = dict(zip(valid, colors))
reverseMap = dict(zip(range(1,numClasses()+1), colors))

def encodeMask(seg):
    # seg = seg.reshape(seg.shape[2], seg.shape[0], seg.shape[1])
    encseg = torch.zeros((seg[0].shape[0],seg[0].shape[1]))  # Create an empty encoded mask
    # s = seg.clone().reshape(seg.shape[1], seg.shape[2], seg.shape[0])

    for label, color in colorMap.items():
        encseg[seg[0] == color[0]] = label

        # color_array = np.array(color)  # Convert color to a NumPy array
        # color_match = np.all(seg == color_array, axis=0)  # Find pixels matching the color
        # print(color_match)
        # encseg[color_match] = label  # Assign the label to matching pixels in the encoded mask

        # for x in range(seg.shape[1]):
        #     for y in range(seg.shape[2]):
        #         print(seg[:, x, y])
        #         if seg[:, x, y] == color:
        #             encseg[x, y] = label

    return encseg




def decodeMask(seg):
    seg = seg.clone().cpu().numpy().astype("uint8")
    r = seg.copy()
    g = seg.copy()
    b = seg.copy()
    
    for c in range(1, numClasses()+1):
        r[seg == c] = reverseMap[c][0]
        g[seg == c] = reverseMap[c][1]
        b[seg == c] = reverseMap[c][2]

    rgb = np.zeros((3, seg.shape[-2], seg.shape[-1]),dtype=np.float32) #stitch everything back together
    rgb[0, :, :] = r / 255.0
    rgb[1, :, :] = g / 255.0
    rgb[2, :, :] = b / 255.0
    return rgb #returning normalized values

def save_preds(output, path):
    num_images = 16 #so that i can get a nice square (batch size is 32)
    temp = torch.zeros((num_images,3,output.shape[-2],output.shape[-1])) #allocate memory for RGB images
    for i in range(num_images):
        temp[i] = torch.from_numpy(decodeMask(output[i]))
    grid = torchvision.utils.make_grid(temp, nrow=4)
    torchvision.utils.save_image(grid, path)


if __name__ == "__main__": #note: run in main directory and NOT in this folder (python utils/utils.py)
    print("Running optimal num workers test now: ")
    optimalWorkers()
    print("-"*30)
    print("Testing padded tensors: ")

    original_tensor = torch.randn(1, 437, 582)

    # Pad the tensor
    padded_tensor = pad_images(original_tensor)
    print(padded_tensor.shape)