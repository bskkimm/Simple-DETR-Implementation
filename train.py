import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from config import train_config, model_config


def train(model: torch.nn.Module,
          dataloader: DataLoader,
          scheduler: MultiStepLR,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          loss_fn: torch.nn.Module,
          device: torch.device):

    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    steps = 0
    loss_history = {
        'classification': [],
        'bbox_regression': []
    }
    for i in range(epochs):
        detr_classification_losses = []
        detr_localization_losses = []
        for idx, (x, y, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training Epoch {i+1}", leave=True):
            for target in y:
                target['boxes'] = target['boxes'].float().to(device)
                target['labels'] = target['labels'].long().to(device)
            x = torch.stack([im.float().to(device) for im in x], dim=0)
            preds = model(x)
            pred_logits = preds['pred_logits']
            pred_bboxes = preds['pred_boxes']

            batch_losses = loss_fn(pred_logits, pred_bboxes, y)['loss']

            loss = (sum(batch_losses['classification']) +
                    sum(batch_losses['bbox_regression']))

            detr_classification_losses.append(sum(batch_losses['classification']).item())
            detr_localization_losses.append(sum(batch_losses['bbox_regression']).item())
 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if steps % train_config['log_steps'] == 0 and steps != 0:
                loss_output = ''
                loss_output +=  'Class. Loss : {:.4f}'.format(
                    np.mean(detr_classification_losses))
                loss_output += ' | Loc. Loss : {:.4f}'.format(
                    np.mean(detr_localization_losses))
                print(loss_output, scheduler.get_last_lr())
            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)

            steps += 1

        scheduler.step()
        print('Finished epoch {}'.format(i+1))
        loss_output = ''
        loss_output += 'Class. Loss : {:.4f}'.format(
            np.mean(detr_classification_losses))
        loss_output += ' | Loc. Loss : {:.4f}'.format(
            np.mean(detr_localization_losses))
        print(loss_output)
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                         f"{model_config['backbone_model']}_{train_config['ckpt_name']}"))
        loss_history['classification'].append(np.mean(detr_classification_losses))
        loss_history['bbox_regression'].append(np.mean(detr_localization_losses))
    print('Done Training...')

    return loss_history

