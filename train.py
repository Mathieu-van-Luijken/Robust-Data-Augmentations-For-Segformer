"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import arguments
import utils
import wandb 

from pathlib import Path
from datetime import datetime 

from torchvision.datasets import Cityscapes
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from torchmetrics import Dice
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torchmetrics.functional as metrics


from dataloader import *
from model import Model
from visualization import visualize_tensor


def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    # Setting a seed to ensure reproducability 
    if  not args.random_seed:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Setting some visualisation tools:
    utils.initialize_wandb(args=args)

    # data loading
    dataloader = CityscapesDataLoader()
    train_loader, val_loader = dataloader.load_train_data(args=args)

    # visualize example images
    
    # define model
    model = Model()
    model.to(device=device)
    

    # define optimizer and loss function (don't forget to ignore class index 255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(device) 
    dice = MulticlassF1Score(num_classes=19, ignore_index=255).to(device)
    miou = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='macro').to(device)

    wandb.log({'Criterion settings':criterion, "Dice settings": dice, "Miou settings": miou })

    # training/validation loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        for i, (image, target) in enumerate(train_loader):
            image = image.to(device)

            target = target.long().view(args.batch_size, -1)
            target = utils.map_id_to_train_id(target) 
            target = target.to(device)

            logits = model(image)
            upsampled_logits =  F.interpolate(logits, size=(1024, 2048), mode='bilinear', align_corners=False)
            flat_upsampled_logits = upsampled_logits.view(args.batch_size, upsampled_logits.size(1), -1)

            loss = criterion(flat_upsampled_logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step

            total_loss += loss.item()
            
            del image, target, logits, upsampled_logits, loss
        mean_loss = total_loss/len(train_loader)
        # Logging information 

        # Clearing cache
        torch.cuda.empty_cache()

        # Training validation
        with torch.no_grad():
            model.eval()
            val_dice = 0.0
            val_miou = 0.0
    
            for j, (image, label) in enumerate(val_loader):
                image = image.to(device)

                label = label.long()
                label = utils.map_id_to_train_id(label) 
                label = label.to(device)

                logits = model(image)
                upsampled_logits = F.interpolate(logits, size=(1024,2048), mode='bilinear', align_corners=False)

                prediction  = torch.argmax(input=upsampled_logits, dim=1).to(device)

                # visualize_tensor(label[0].squeeze(0))
                # visualize_tensor(prediction[0].squeeze(0))

                val_dice += dice(prediction, label.squeeze(1)).item()
                val_miou += miou(prediction, label.squeeze(1)).item()

                del image, label, logits, upsampled_logits
    
            
            val_dice = val_dice/len(val_loader)
            val_miou = val_miou/len(val_loader)
            wandb.log({"Dice": val_dice, "Mean IoU": val_miou, "Loss": mean_loss})
        print(f"Epoch {epoch}: \n"
              f"Current Validation Dice: {val_dice} \n"
              f"Current Validation mIoU: {val_miou}")
        
        torch.cuda.empty_cache()        
        
    # save model
    save_model(args, model)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Training done at: {current_time} with augmentation {args.augmentation}")

    # visualize some results
    

def save_model(args, model):
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()

    date = datetime.now().strftime("%d-%H%M%S")
    torch.save(model.state_dict(), Path(f"results\{args.augmentation}-{date}.pth"))
    


if __name__ == "__main__":
    # Get the arguments
    parser = arguments.get_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Training starting at: {current_time} with augmentation {args.augmentation}")
    main(args)
