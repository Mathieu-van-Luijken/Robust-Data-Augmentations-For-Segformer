"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import arguments
import utils 

from pathlib import Path
from datetime import datetime 

from torchvision.datasets import Cityscapes
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import Dice
from transformers import SegformerForSemanticSegmentation, SegformerConfig


from dataloader import *
from model import Model


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
    train_loader, val_loader = dataloader.load_data(args=args)

    # visualize example images
    
    # define model
    model = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path="nvidia/segformer-b2-finetuned-cityscapes-1024-1024")
    model.to(device=device)
    

    # define optimizer and loss function (don't forget to ignore class index 255)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean').to(device)
    dice = Dice(num_classes=20, ignore_index=255, average=None)
    miou = MulticlassJaccardIndex(num_classes=20, ignore_index=255, average=None)

    loss_dict = {"Loss": [], "Dice": [], "Mean IoU": []}

    # training/validation loop
    for epoch in range(args.num_epochs):
        model.train()
        for image, target in train_loader:
            image = image.to(device)

            target = target.long().squeeze(0).view(-1)
            target = utils.map_id_to_train_id(target) 
            target = target.to(device)

            logits = model(image).logits
            upsampled_logits = F.interpolate(logits, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
            upsampled_logits = upsampled_logits.view(-1, upsampled_logits.size(1))

            loss = loss(upsampled_logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step

            del image, target, logits, upsampled_logits, loss
        # Logging information 

        # Clearing cache
        torch.cuda.empty_cache()

        # Training validation
        with torch.no_grad:
            model.eval()
            for image, label in val_loader:
                image = image.to(device)

                label = target.long().squeeze(0).view(-1)
                label = utils.map_id_to_train_id(target) 
                label = target.to(device)

                logits = model(image).logits
                upsampled_logits = F.interpolate(logits, size=(image.size(2), image.size(3)), mode='bilinear', align_corners=False)
                upsampled_logits = upsampled_logits.view(-1, upsampled_logits.size(1))

                prediction  = torch.argmax(input=upsampled_logits, dim=1).to(device)
                
                dice_loss = dice(prediction, label).item().cpu()
                miou_loss = miou(prediction, label).item().cpu()

                







    torch.cuda.empty_cache()        

    # save model
    save_model(args, model)

    # visualize some results
    pass

def save_model(args, model):
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()

    date = datetime.now().strftime("%d-%H%M")

    torch.save(model.state_dict(), Path(f"results\{args.augmenter} {date}.pt"))
    


if __name__ == "__main__":
    # Get the arguments
    parser = arguments.get_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    main(args)
