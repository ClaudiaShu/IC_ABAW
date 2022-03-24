import argparse
import warnings
import logging
import torch.optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, MultiStepLR, CosineAnnealingWarmRestarts, StepLR
from pytorch_metric_learning import distances, losses, miners, reducers, testers
import pytorch_warmup as warmup
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import *
from models import *
from config import *
from loss import *
from models_final import *
from utils import *
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

def args_config():
    parser = argparse.ArgumentParser(description='EXP Training')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')  # 1e-3#
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')  # 256#
    parser.add_argument('--num_epochs', default=5, type=int, help='number epochs')  # 12#
    parser.add_argument('--num_classes', default=8, type=int, help='number classes')
    parser.add_argument('--weight_decay', default=5e-4, type=float)  # 5e-4#
    parser.add_argument('--seq_len', default=5, type=int)

    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--alpha', type=float, default=0.7,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--remix_tau', type=float, default=0.9,
                        help='remixup interpolation coefficient tau(default: 1)')
    parser.add_argument('--remix_kappa', type=float, default=15.,
                        help='remixup interpolation coefficient kappa(default: 1)')

    parser.add_argument('--dataset', type=str, default="v2")  # ["v1","v2","v3"]
    parser.add_argument('--net', type=str, default="RES_M")  # ["RES","INC","HRN",'C','M','T']
    parser.add_argument('--mode', type=str, default="train")  # ["train","trainMixup","trainRemix"]
    parser.add_argument('--file', type=str, default=f"{parser.parse_args().net}_ex_best")

    parser.add_argument('--sample', type=str, default="ori")
    parser.add_argument('--resume', type=bool, default=False)

    parser.add_argument('--loss', type=str, default='CrossEntropyLabelAwareSmooth')  #['CrossEntropyLabelAwareSmooth','SuperContrastive']
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='SGD')

    args = parser.parse_args()

    return args

def train(train_loader, model, itr):
    cost_list = 0

    cat_preds = []
    cat_labels = []

    model.train()

    for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{args.net} Train_mode with warmup {args.warmup}'):
        images = samples['images'].to(device).float()
        labels_cat = samples['labels'].to(device).long()
        # import pdb; pdb.set_trace()

        pred_cat = model(images)
        loss = criterion(pred_cat, labels_cat)
        cost_list += loss.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step(itr)
        if args.warmup:
            warmup_scheduler.dampen()
        itr += 1

        pred_cat = F.softmax(pred_cat)
        pred_cat = torch.argmax(pred_cat, dim=1)

        cat_preds.append(pred_cat.detach().cpu().numpy())
        cat_labels.append(labels_cat.detach().cpu().numpy())
        t.set_postfix(Lr=optimizer.param_groups[0]['lr'],
                      Loss=f'{cost_list / (batch_idx + 1):04f}',
                      itr=itr)

    cat_preds = np.concatenate(cat_preds, axis=0)
    cat_labels = np.concatenate(cat_labels, axis=0)
    cm = confusion_matrix(cat_labels, cat_preds)
    cr = classification_report(cat_labels, cat_preds)
    f1, acc, total = EXPR_metric(cat_preds, cat_labels)
    print(f'f1 = {f1} \n'
          f'acc = {acc} \n'
          f'total = {total} \n',
          'confusion metrix: \n', cm, '\n',
          'classification report: \n', cr, '\n')

def train_tri(train_loader, loss_func, mining_func, model, itr):
    '''
    This training function is for triplet training
    essembling the classification output from two different modalities and the fused feature
    joint training from triplet loss from two embeddings
    :param train_loader: training dataset
    :param criterion: loss function
    :param loss_func: triplet loss
    :param mining_func: getting triplet pairs
    :param model: training model (with essemble)
    :param itr: iteration number for learning rate degrade
    '''
    cost_list1 = 0
    cost_list2 = 0
    cost_list3 = 0
    cost_list4 = 0
    cost_list5 = 0

    cat_preds = []
    cat_labels = []

    model.train()

    for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{args.net} trainTriplet_mode with warmup {args.warmup}'):
        images = samples['images'].to(device).float()
        labels_cat = samples['labels'].to(device).long()
        # import pdb; pdb.set_trace()

        class1, class2, class3, emb1, emb2 = model(images)
        loss1 = criterion(class1, labels_cat)  # 0.5
        loss2 = criterion(class2, labels_cat)  # 0.5
        loss3 = criterion(class3, labels_cat)  # 1

        tuple1 = mining_func(emb1, labels_cat)
        tuple2 = mining_func(emb2, labels_cat)
        tri_loss1 = loss_func(emb1, labels_cat, tuple1)
        tri_loss2 = loss_func(emb2, labels_cat, tuple2)

        loss = 0.5*loss1+0.5*loss2+loss3+tri_loss1+tri_loss2

        cost_list1 += loss1.item()
        cost_list2 += loss2.item()
        cost_list3 += loss3.item()
        cost_list4 += tri_loss1.item()
        cost_list5 += tri_loss2.item()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step(itr)
        if args.warmup:
            warmup_scheduler.dampen()
        itr += 1

        pred_cat1 = F.softmax(class1)
        pred_cat2 = F.softmax(class2)
        pred_cat3 = F.softmax(class3)
        pred_cat = torch.tensor(0.25*pred_cat1+0.25*pred_cat2+0.5*pred_cat3)
        pred_cat = torch.argmax(pred_cat, dim=1)

        cat_preds.append(pred_cat.detach().cpu().numpy())
        cat_labels.append(labels_cat.detach().cpu().numpy())
        t.set_postfix(Lr=optimizer.param_groups[0]['lr'],
                      Loss1=f'{cost_list1 / (batch_idx + 1):04f}',
                      Loss2=f'{cost_list2 / (batch_idx + 1):04f}',
                      Loss3=f'{cost_list3 / (batch_idx + 1):04f}',
                      Loss4=f'{cost_list4 / (batch_idx + 1):04f}',
                      Loss5=f'{cost_list5 / (batch_idx + 1):04f}')

    cat_preds = np.concatenate(cat_preds, axis=0)
    cat_labels = np.concatenate(cat_labels, axis=0)
    cm = confusion_matrix(cat_labels, cat_preds)
    cr = classification_report(cat_labels, cat_preds)
    f1, acc, total = EXPR_metric(cat_preds, cat_labels)
    print(f'f1 = {f1} \n'
          f'acc = {acc} \n'
          f'total = {total} \n',
          'confusion metrix: \n', cm, '\n',
          'classification report: \n', cr, '\n')

def train_mixup(train_loader, model, itr, use_cuda=True):
    cost_list = 0
    correct = 0
    num = 0

    cat_preds = []
    cat_labels = []

    model.train()

    for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{args.net} TrainMixup_mode with warmup {args.warmup}'):
        images = samples['images'].to(device).float()
        labels_cat = samples['labels'].to(device).long()
        # import pdb; pdb.set_trace()

        inputs, targets_a, targets_b, lam = mixup_data(images, labels_cat, args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        pred_cat = model(inputs)

        loss = mixup_criterion(criterion, pred_cat, targets_a, targets_b, lam)
        cost_list += loss.item()

        pred_cat = F.softmax(pred_cat)
        pred_cat = torch.argmax(pred_cat, dim=1)
        correct += (lam * pred_cat.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * pred_cat.eq(targets_b.data).cpu().sum().float())
        num += labels_cat.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(itr)
        if args.warmup:
            warmup_scheduler.dampen()
        itr += 1

        cat_preds.append(pred_cat.detach().cpu().numpy())
        cat_labels.append(labels_cat.detach().cpu().numpy())

        t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                      Acc=f'{correct / num:04f}',
                      Lr=optimizer.param_groups[0]['lr'])

def train_remix(train_loader, model, itr, use_cuda=True):
    cost_list = 0
    correct = 0
    num = 0

    cat_preds = []
    cat_labels = []

    model.train()

    for batch_idx, samples in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'{args.net} TrainRemix_mode with warmup {args.warmup}'):
        images = samples['images'].to(device).float()
        labels_cat = samples['labels'].to(device).long()
        # import pdb; pdb.set_trace()

        inputs, targets_a, targets_b, lamX = mixup_data(images, labels_cat, args.alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

        # κ-Majority. A sample (xi, yi), is considered to be κ-majority than sample (xj, yj ),
        # what remix does
        lamY = torch.empty(images.shape[0]).fill_(lamX).float().to(device)
        n_i, n_j = targets_a, targets_b
        if lamX < args.remix_tau:
            lamY[n_i / n_j >= args.remix_kappa] = 0
        if 1 - lamX < args.remix_tau:
            lamY[(n_i * args.remix_kappa) / n_j <= 1] = 1

        pred_cat = model(inputs)

        loss = mixup_criterion(criterion, pred_cat, targets_a, targets_b, lamY)
        loss = loss.mean()
        cost_list += loss.item()

        pred_cat = F.softmax(pred_cat)
        pred_cat = torch.argmax(pred_cat, dim=1)

        dim1 = lamY * accuracy(pred_cat.cpu().numpy(), targets_a.cpu().numpy())
        dim2 = (1 - lamY) * accuracy(pred_cat.cpu().numpy(), targets_b.cpu().numpy())
        correct += (dim1 + dim2).mean()

        num += labels_cat.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(itr)
        if args.warmup:
            warmup_scheduler.dampen()
        itr += 1

        cat_preds.append(pred_cat.detach().cpu().numpy())
        cat_labels.append(labels_cat.detach().cpu().numpy())

        t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                      Acc=f'{correct / num:04f}',
                      Lr=optimizer.param_groups[0]['lr'])

def valid(valid_loader, model):
    cost_test = 0
    model.eval()
    with torch.no_grad():
        cat_preds = []
        cat_labels = []
        for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
            images = samples['images'].to(device).float()
            labels_cat = samples['labels'].to(device).long()

            pred_cat = model(images)
            test_loss = criterion(pred_cat, labels_cat)
            cost_test += test_loss.item()

            pred_cat = F.softmax(pred_cat)
            pred_cat = torch.argmax(pred_cat, dim=1)

            cat_preds.append(pred_cat.detach().cpu().numpy())
            cat_labels.append(labels_cat.detach().cpu().numpy())
            t.set_postfix(Loss=f'{cost_test / (batch_idx + 1):04f}')

        cat_preds = np.concatenate(cat_preds, axis=0)
        cat_labels = np.concatenate(cat_labels, axis=0)
        cm = confusion_matrix(cat_labels, cat_preds)
        cr = classification_report(cat_labels, cat_preds)
        f1, acc, total = EXPR_metric(cat_preds, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')

    return f1, acc, total, cm, cr

def valid_tri(valid_loader, model):
    cost_test = 0
    model.eval()
    with torch.no_grad():
        cat_preds = []
        cat_labels = []
        for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
            images = samples['images'].to(device).float()
            labels_cat = samples['labels'].to(device).long()

            class1, class2, class3, emb1, emb2 = model(images)

            loss1 = criterion(class1, labels_cat)
            loss2 = criterion(class2, labels_cat)
            loss3 = criterion(class3, labels_cat)
            test_loss = (loss1+loss2+loss3)/3
            cost_test += test_loss.item()

            pred_cat1 = F.softmax(class1)
            pred_cat2 = F.softmax(class2)
            pred_cat3 = F.softmax(class3)
            pred_cat = torch.tensor(0.25 * pred_cat1 + 0.25 * pred_cat2 + 0.5 * pred_cat3)
            # pred_cat = torch.tensor(pred_cat3)
            pred_cat = torch.argmax(pred_cat, dim=1)

            cat_preds.append(pred_cat.detach().cpu().numpy())
            cat_labels.append(labels_cat.detach().cpu().numpy())
            t.set_postfix(Loss=f'{cost_test / (batch_idx + 1):04f}')

        cat_preds = np.concatenate(cat_preds, axis=0)
        cat_labels = np.concatenate(cat_labels, axis=0)
        cm = confusion_matrix(cat_labels, cat_preds)
        cr = classification_report(cat_labels, cat_preds)
        f1, acc, total = EXPR_metric(cat_preds, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')

    return f1, acc, total, cm, cr

def setup(df_train, df_valid, args):
    if args.net == "RNN":
        train_dataset = Aff2_Dataset_series_shuffle(df=df_train, root=False,
                                                    transform=train_transform, type_partition='ex',
                                                    length_seq=args.seq_len)

        valid_dataset = Aff2_Dataset_series_shuffle(df=df_valid, root=False,
                                                    transform=test_transform, type_partition='ex',
                                                    length_seq=args.seq_len)
    else:
        train_dataset = Aff2_Dataset_static_shuffle(df=df_train, root=False,
                                                    transform=train_transform, type_partition='ex')

        valid_dataset = Aff2_Dataset_static_shuffle(df=df_valid, root=False,
                                                    transform=test_transform, type_partition='ex')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              num_workers=12,
                              shuffle=True,
                              drop_last=False)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=12,
                              shuffle=False,
                              drop_last=False)

    best_acc = 0
    if args.net == "RES":
        model = BaselineRES(num_classes=args.num_classes)
    elif args.net == "INC":
        model = BaselineINC(num_classes=args.num_classes)
    elif args.net == "HRN":
        model = BaselineHRN(num_classes=args.num_classes)
    elif args.net == "RES_C":
        model = BaselineRES_C(num_classes=args.num_classes)
    elif args.net == "INC_C":
        model = BaselineINC_C(num_classes=args.num_classes)
    elif args.net == "RES_M":
        model = BaselineRES_M(num_classes=args.num_classes)
    elif args.net == "INC_M":
        model = BaselineINC_M(num_classes=args.num_classes)
    elif args.net == "RES_T":  # tri
        model = BaselineRES_T(num_classes=args.num_classes)
    elif args.net == "INC_T":  # tri
        model = BaselineINC_T(num_classes=args.num_classes)
    else:
        model = Baseline(num_classes=args.num_classes)

    return train_loader, valid_loader, model, best_acc


if __name__ == '__main__':
    args = args_config()
    seed_everything()
    # train set
    df_train = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Train_Set_{args.dataset}/*')
    # valid set
    df_valid = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_{args.dataset}/*')

    list_labels_ex = np.array(df_train['labels_ex'].values)
    weight, num_class_list = ex_count_weight(list_labels_ex)
    weight = weight.to(device)
    print("Exp weight: ", weight)

    train_loader, valid_loader, model, best_acc = setup(df_train, df_valid, args)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    para_dict = {
        "num_classes": args.num_classes,
        "num_class_list": num_class_list,
        "device": device,
        "cfg": cfg_loss,
    }

    # loss
    if args.loss == 'CrossEntropyLabelAwareSmooth':
        criterion = CrossEntropyLabelAwareSmooth(para_dict=para_dict)
    elif args.loss == 'BalancedSoftmaxCE':
        criterion = BalancedSoftmaxCE(para_dict=para_dict)
    else:
        criterion = nn.CrossEntropyLoss()

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    criterion_tri = losses.TripletMarginLoss(margin=0.1, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.1, distance=distance, type_of_triplets="semihard")

    # optim & scheduler
    num_steps = len(train_loader) * args.num_epochs
    if args.warmup:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, eta_min=1e-6, last_epoch=-1)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=len(train_loader))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=num_steps, eta_min=1e-6, last_epoch=-1)
        # scheduler = StepLR(optimizer=optimizer, step_size=len(train_loader), gamma=0.5, last_epoch=-1)
        # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=len(train_loader), T_mult=2, eta_min=0, last_epoch=-1)

    optimizer.zero_grad()
    optimizer.step()

    itr = 0

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            t.set_description('Epoch %i' % epoch)
            # train here
            if args.mode == "train":
                train(train_loader, model, itr)
            elif args.mode == "trainMixup":
                train_mixup(train_loader, model, itr)
            elif args.mode == "trainRemix":
                train_remix(train_loader, model, itr)
            elif args.mode == "trainTriplet":
                train_tri(train_loader, criterion_tri, mining_func, model, itr)
            else:
                assert AttributeError
            itr += len(train_loader)
            if args.mode == "trainTriplet":
                f1, acc, total, cm, cr = valid_tri(valid_loader, model)
            else:
                f1, acc, total, cm, cr = valid(valid_loader, model)

            state = {
                'net': model,
                'acc': total,
                'epoch': epoch,
                'optim': optimizer.state_dict(),
                'rng_state': torch.get_rng_state()
            }
            os.makedirs(f'./checkpoint/expression/{args.net}_{args.dataset}_{args.mode}', exist_ok=True)
            torch.save(state, f'./checkpoint/expression/{args.net}_{args.dataset}_{args.mode}/{args.file}_{epoch}.pth')

            os.makedirs(f'./log/expression/{args.net}_{args.dataset}_{args.mode}', exist_ok=True)
            logging.basicConfig(level=logging.INFO,
                                filename=f'./log/expression/{args.net}_{args.dataset}_{args.mode}/{args.file}.log',
                                filemode='a+')

            logging.info(f'start epoch {epoch} ----------------------------')
            logging.info(f'Currently {args.mode} using {args.net} model')
            logging.info(f'Sampling the data in {args.sample} way')
            logging.info(f'Using {args.dataset} dataset')
            logging.info(f'Training loss: {args.loss}')
            logging.info(f'optimizer: {args.optim}')
            logging.info(f'hyperparameter setting: ')
            logging.info(f'learning rate: {str(args.lr)} ')
            logging.info(f'batch size: {str(args.batch_size)} ')
            logging.info(f'trianing epochs: {str(args.num_epochs)}')
            logging.info(f'epoch {epoch}: f1 = {f1}, acc = {acc}, total = {total}')
            logging.info('end   -------------------------------------------')
