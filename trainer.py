import copy
import wandb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from networks import _create_vision_transformer

def training(args, prototype_mode, taskid, train_loader, test_loader, known_classes=0, vitprompt=None, numclass=10):
    trainer = pl.Trainer(default_root_dir='./checkpoints/'+wandb.run.name, accelerator="gpu", devices=1,
                         max_epochs=args.max_epochs, progress_bar_refresh_rate=1,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="train_acc"),
                                    LearningRateMonitor("epoch"),])

    if taskid == 0:
        model = prototype_mode(num_cls=numclass, lr=args.lr, max_epoch=args.max_epochs, weight_decay=0.0005,
                        known_classes=known_classes, freezep=False, using_prompt=args.using_prompt,
                        anchor_energy=args.anchor_energy, lamda=args.lamda, energy_beta=args.energy_beta)
    else:
        model = prototype_mode(num_cls=numclass, lr=args.lr, max_epoch=args.max_epochs, weight_decay=0.0005,
                        known_classes=known_classes, freezep=True, using_prompt=args.using_prompt,
                               anchor_energy=args.anchor_energy, lamda=args.lamda, energy_beta=args.energy_beta)
        model.vitprompt = vitprompt


    trainer.fit(model, train_loader)
    if args.dataset == "core50":
        val_result = [{'test_acc':0}]
    else:
        val_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(val_result)
    return model, val_result


def eval(args, load_path, datamanage):
    assembles = torch.load(load_path, map_location=torch.device('cpu'))
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    ptvit = _create_vision_transformer('vit_base_patch16_224', pretrained=True, **model_kwargs)

    device = 'cuda:0'
    ptvit = ptvit.to(device)
    ptvit.eval()
    all_tabs = assembles['all_tabs']
    all_classifiers = assembles['all_classifiers']
    all_tokens = assembles['all_tokens']
    vitpromptlist = assembles['vitpromptlist']

    all_tabs = [i.to(device) for i in all_tabs]
    all_classifiers = [i.to(device) for i in all_classifiers]
    all_tokens = [i.to(device) for i in all_tokens]
    vitpromptlist = [i.to(device) for i in vitpromptlist]

    _known_classes=0
    # fast mode
    candidata_temperatures = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # slow mode
    # candidata_temperatures = [i/1000 for i in range(1,1000,1)]
    select_temperature = [0.001] # for first stage with no former data
    accuracy_table, accs = [], []
    print("Testing...")
    for taskid in range(datamanage.nb_tasks):
        _total_classes = _known_classes + datamanage.get_task_size(taskid)
        test_till_now_dataset = datamanage.get_dataset(np.arange(0, _total_classes), source='test', mode='test')
        test_till_now_loader = DataLoader(test_till_now_dataset, batch_size=128, shuffle=False, num_workers=8)

        if taskid > 0:
            classifiers = all_classifiers[:taskid+1]
            task_tokens = all_tokens[:taskid+1]
            tabs = all_tabs[:taskid+1]
            current_dataset = datamanage.get_dataset(np.arange(_known_classes, _total_classes), source='train', mode='test')
            current_dataloader = DataLoader(current_dataset, batch_size=128, shuffle=False, num_workers=8)
            all_energies = {i: [] for i in candidata_temperatures}
            with torch.no_grad():
                for _, (_, inputs, targets) in enumerate(current_dataloader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    energys = {i: [] for i in candidata_temperatures}
                    image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)

                    B = image_features.shape[0]
                    for idx, fc in enumerate(classifiers):
                        task_token = task_tokens[idx].expand(B, -1, -1)
                        task_token, attn, v = tabs[idx](torch.cat((task_token, image_features), dim=1), mask_heads=None)
                        task_token = task_token[:, 0]
                        logit = fc(task_token)

                        for tem in candidata_temperatures:
                            energys[tem].append(torch.logsumexp(logit / tem, axis=-1))
                    energys = {i: torch.stack(energys[i]).T for i in candidata_temperatures}
                    for i in candidata_temperatures:
                        all_energies[i].append(energys[i])

            all_energies = {i: torch.cat(all_energies[i]) for i in candidata_temperatures}
            seperation_accuracy = []
            for i in candidata_temperatures:
                seperation_accuracy.append((sum(all_energies[i].max(1)[1]==(taskid))/len(all_energies[i])).item())
            select_temperature.append(candidata_temperatures[np.array(seperation_accuracy).argmax()])

        set_select_temperature = list(set(select_temperature))
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(test_till_now_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            candiatetask = {i: [] for i in set_select_temperature}
            seperatePreds = []

            with torch.no_grad():
                image_features = ptvit(inputs, instance_tokens=vitpromptlist[taskid].weight, returnbeforepool=True)
                B = image_features.shape[0]
                for idx, fc in enumerate(all_classifiers[:taskid+1]):
                    task_token = all_tokens[:taskid+1][idx].expand(B, -1, -1)
                    task_token, attn, v = all_tabs[:taskid+1][idx](torch.cat((task_token, image_features), dim=1), mask_heads=None)
                    task_token = task_token[:, 0]
                    logit = fc(task_token)
                    for tem in set_select_temperature:
                        candiatetask[tem].append(torch.logsumexp(logit / tem, axis=-1))
                    seperatePreds.append(logit.max(1)[1]+idx*logit.shape[1])

            candiatetask = {i: torch.stack(candiatetask[i]).T  for i in set_select_temperature}
            seperatePreds = torch.stack(seperatePreds).T

            pred = []
            for tem in set_select_temperature:
                val, ind = candiatetask[tem].max(1)
                pred.append(ind)
            indexselection = torch.stack(pred, 1)
            selectid = torch.mode(indexselection, dim=1, keepdim=False)[0]
            outputs = []
            for row, idx in enumerate(selectid):
                outputs.append(seperatePreds[row][idx])
            outputs = torch.stack(outputs)
            y_pred.append(outputs.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        if args.dil:
            accs.append(np.around((y_pred.T%args.max_cls == y_true%args.max_cls).sum() * 100 / len(y_true), decimals=2))
        else:
            accs.append(np.around((y_pred.T == y_true).sum() * 100 / len(y_true), decimals=2))

        tempacc = []
        for class_id in range(0, np.max(y_true), _total_classes-_known_classes):
            idxes = np.where(np.logical_and(y_true >= class_id, y_true < class_id + _total_classes-_known_classes))[0]
            tempacc.append(np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=3))
        accuracy_table.append(tempacc)

        _known_classes = _total_classes


    np_acctable = np.zeros([taskid + 1, taskid + 1])
    for idxx, line in enumerate(accuracy_table):
        idxy = len(line)
        np_acctable[idxx, :idxy] = np.array(line)
    # import pdb;pdb.set_trace()
    np_acctable = np_acctable.T
    print("Accuracy table:")
    print(np_acctable)
    print("Accuracy curve:")
    print(accs)
    print("FAA: {}".format(accs[-1]))

    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, taskid])[:taskid])
    print("FF: {}".format(forgetting))
