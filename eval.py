"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import arguments
import utils
import wandb 
import json

from pathlib import Path
from datetime import datetime 

from torchvision.datasets import Cityscapes
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score
from torchmetrics import Dice
from transformers import SegformerForSemanticSegmentation, SegformerConfig
import torchmetrics.functional as metrics


from dataloader import *
from model import Model
from visualization import *
from process_data import postprocess

def eval(args, model_name, dataset, method):
    model = Model()
    state_dict=torch.load(Path(f'results/{model_name}'))
    model.load_state_dict(state_dict=state_dict)
    model = model.to(device)

    dataloader = CityscapesDataLoader()
    test_loader = dataloader.load_eval_data(args=args, data_path=Path(f"eval_data/Cityscapes_{dataset}"))

    dice = MulticlassF1Score(num_classes=19, ignore_index=255).to(device)
    miou = MulticlassJaccardIndex(num_classes=19, ignore_index=255, average='macro').to(device)
    softmax = nn.Softmax(dim=1)

    test_dice = 0
    test_miou = 0

    model.eval()
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.to(device)

            label = label.long()
            label = utils.map_id_to_train_id(label)
            label = label.to(device)

            logits = model(image)
            upsampled_logits = utils.upsample(logits=logits, method=method)
            prediction  = torch.argmax(input=upsampled_logits, dim=1).to(device)

            test_dice += dice(prediction, label.squeeze(0)).item()
            test_miou += miou(prediction, label.squeeze(0)).item()

            del image, label, logits, upsampled_logits
            

        test_dice = test_dice/len(test_loader)
        test_miou = test_miou/len(test_loader)
        print(f"Final test values for {dataset} are \n"
                f"Dice: {test_dice} \n"
                f"Mean IoU: {test_miou}")
        return test_dice, test_miou

if __name__ == "__main__":
# Get the arguments
    parser = arguments.get_arg_parser()
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # The model we want to test for
    model_name = 'basic-19-115010.pth'
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Evaluation starting at: {current_time}")
    
    
    # Create a dictionary for all of the test sets
    results = {}
    for method in ['bilinear', 'bicubic', 'lanczos']:
        sub_results = {}
        for dataset in ['night', 'night_drops', 'overcast', 'overcast_drops', 'snow', 'snow_drops', 'wet', 'wet_drops']:
            dice, miou = eval(args=args,
                model_name=model_name,
                dataset=dataset,
                method=method)
            sub_results[dataset] = [dice, miou]
        results[method] = sub_results
    with open(f'results/{model_name[:-3]}-{method}.json', "w") as f:
        json.dump(results, f)

    