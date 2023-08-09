import torch
import torchvision
import torchmetrics
from torch import nn
from PIL import Image, ImageFile
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import *
from model import CommaNetwork
import segmentation_models_pytorch as smp

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

if __name__=="__main__":
    #PARAMS
    LR = 1e-4
    WEIGHT_DECAY = 1e-7
    BATCH_SIZE = 12
    EPOCHS = 100
    CHECKPOINT_EPOCH = 0
    NUM_WORKERS = 22
    LOAD_MODEL = False
    SAVE_MODEL = True
    WIDTH = 582 #using same width and height from https://github.com/YassineYousfi/comma10k-baseline/tree/main
    HEIGHT = 437
    BEST_LOSS = 1e5 #defining some arbitary value
    VAL_SPLIT = 0.2

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    target_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    inv_norm = torchvision.transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.255]
    )

    dataset = CommaPreprocess(WIDTH, HEIGHT, transform=transform, target_transform=None)
    val_length = int(VAL_SPLIT * len(dataset))
    train_length = len(dataset) - val_length
    trainset, valset = torch.utils.data.random_split(dataset, [train_length, val_length])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(len(dataset), len(trainset), len(valset))
    img, seg = trainset[0]
    print("image dimensions: ", img.shape, "segmentation size: ", seg.shape)
    print(torch.unique(seg))

    deseg = decodeMask(torch.clone((seg)))
    print(deseg.shape)

    fig,ax=plt.subplots(ncols=3,figsize=(12,8))
    ax[0].imshow(img.permute(1,2,0))
    ax[1].imshow(seg)
    ax[2].imshow(torch.from_numpy(deseg).permute(1,2,0))
    plt.show()

    model = smp.Unet(encoder_name="efficientnet-b4", 
                     #encoder_weights="imagenet",
                     in_channels=3, 
                     classes=numClasses(),).to(device)    

    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=numClasses()).to(device)

    criterion = nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss(mode="multiclass")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if LOAD_MODEL:
        print("Loading models...")
        checkpoint = torch.load("saved_models/model.pth")
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        CHECKPOINT_EPOCH = checkpoint['epoch']+1
        BEST_LOSS = checkpoint['loss']
        print(BEST_LOSS)
        print("Done!")

    for epoch in range(CHECKPOINT_EPOCH, EPOCHS+CHECKPOINT_EPOCH):
        print('-'*20)

        model.train()
        for i, (img, annotation) in enumerate(trainloader):
            img = img.to(device, dtype=torch.float32)
            annotation = annotation.to(device).long()

            output = model(img)
            _output = torch.max(output, dim=1)[1]
            
            ce_loss = criterion(output, annotation)
            iou_loss = iou(_output, annotation)
            dice_loss = dice(output, annotation)

            loss = 0.25*ce_loss + 0.75*dice_loss # weigh losses


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (int(i+1))%(len(trainset)//BATCH_SIZE//5) == 0:
                print(f"Epoch: {epoch+1}/{EPOCHS+CHECKPOINT_EPOCH}, Step: {i+1}/{len(trainset)//BATCH_SIZE}, CE loss: {loss.item():4f}, Dice score: {1-dice_loss.item():4f}, IOU: {iou_loss:4f}")

        running_val_loss = 0
        running_iou_loss = 0
        running_dice_loss = 0
        path = f"predictions/epoch_{epoch+1}.png"
        apath = f"ann/epoch_{epoch+1}.png"

        model.eval()
        with torch.no_grad(): 
            # trace = torch.jit.trace(model, x)
            for idx, (img, annotation) in enumerate(valloader, 0):
                img = img.to(device, dtype=torch.float32)
                annotation = annotation.to(device).long()

                output = model(img)
                _output = torch.max(output,dim=1)[1]

                if idx==0:
                    save_preds(_output, path)

                val_loss = criterion(output, annotation)
                val_iou = iou(_output, annotation)
                val_dice = dice(output, annotation)
                running_val_loss += val_loss.item()
                running_iou_loss += val_iou.item()
                running_dice_loss += val_dice.item()

        avg_val_loss = running_val_loss/len(valloader)
        avg_iou_loss = running_iou_loss/len(valloader)
        avg_dice_loss = running_dice_loss/len(valloader)

        print(f"Validation Loss: {avg_val_loss:4f}, Dice score: {1-avg_dice_loss:4f}, IOU: {avg_iou_loss:4f}")
        

        if (avg_val_loss) < BEST_LOSS: 
            print("Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, "saved_models/model.pth")

            # torch.jit.save(trace, "saved_models/jit_model.pth")
            print("Done!")
            BEST_LOSS = avg_val_loss
            print('-'*20)