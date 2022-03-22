import argparse
import warnings
import logging
import torch.optim
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import *
from models import *
from config import *
from loss import *
from models_abaw import *
from utils import *
import os


warnings.filterwarnings("ignore")

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def args_config():
    parser = argparse.ArgumentParser(description='EXP Training')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # 1e-3#
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')  # 256#
    parser.add_argument('--num_epochs', default=1, type=int, help='number epochs')  # 12#
    parser.add_argument('--num_classes', default=8, type=int, help='number classes')  # 12#
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
    parser.add_argument('--net', type=str, default="ESS")  # ["CNN","RNN"]
    parser.add_argument('--mode', type=str, default="train")  # ["trainMixup","trainRemix"]
    parser.add_argument('--file', type=str, default="Ess_ex_best")

    parser.add_argument('--sample', type=str, default="ori")
    parser.add_argument('--resume', type=bool, default=True)

    parser.add_argument('--loss', type=str, default='CrossEntropyLabelAwareSmooth')  #['CrossEntropyLabelAwareSmooth','SuperContrastive']
    parser.add_argument('--optim', type=str, default='SGD')

    args = parser.parse_args()

    return args

def valid(valid_loader, criterion, model):
    cost_test = 0
    model.eval()
    with torch.no_grad():
        cat_preds = []
        cat_labels = []

        cat_preds1 = []
        cat_preds2 = []
        cat_preds3 = []
        for batch_idx, samples in tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid_mode'):
            images = samples['images'].to(device).float()
            labels_cat = samples['labels'].to(device).long()
            # import pdb; pdb.set_trace()

            class1, class2, class3, emb1, emb2 = model(images)

            loss1 = criterion(class1, labels_cat)
            loss2 = criterion(class2, labels_cat)
            loss3 = criterion(class3, labels_cat)
            test_loss = (loss1+loss2+loss3)/3
            cost_test += test_loss.item()

            pred_cat1 = F.softmax(class1)
            pred_cat2 = F.softmax(class2)
            pred_cat3 = F.softmax(class3)
            pred_cat = torch.tensor(pred_cat1 + pred_cat2 + pred_cat3)

            # ess
            pred_cat = torch.argmax(pred_cat, dim=1)
            cat_preds.append(pred_cat.detach().cpu().numpy())


            # class 1
            pred_cat1 = torch.argmax(pred_cat1, dim=1)
            cat_preds1.append(pred_cat1.detach().cpu().numpy())

            # class 2
            pred_cat2 = torch.argmax(pred_cat2, dim=1)
            cat_preds2.append(pred_cat2.detach().cpu().numpy())

            # class 3
            pred_cat3 = torch.argmax(pred_cat3, dim=1)
            cat_preds3.append(pred_cat3.detach().cpu().numpy())

            cat_labels.append(labels_cat.detach().cpu().numpy())
            t.set_postfix(Loss=f'{cost_test / (batch_idx + 1):04f}')

        cat_labels = np.concatenate(cat_labels, axis=0)
        cat_preds = np.concatenate(cat_preds, axis=0)
        cat_preds1 = np.concatenate(cat_preds1, axis=0)
        cat_preds2 = np.concatenate(cat_preds2, axis=0)
        cat_preds3 = np.concatenate(cat_preds3, axis=0)

        # ess
        cm = confusion_matrix(cat_labels, cat_preds)
        cr = classification_report(cat_labels, cat_preds)
        f1, acc, total = EXPR_metric(cat_preds, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')
        # 1
        cm = confusion_matrix(cat_labels, cat_preds1)
        cr = classification_report(cat_labels, cat_preds1)
        f1, acc, total = EXPR_metric(cat_preds1, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')
        # 2
        cm = confusion_matrix(cat_labels, cat_preds2)
        cr = classification_report(cat_labels, cat_preds2)
        f1, acc, total = EXPR_metric(cat_preds2, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')
        # 3
        cm = confusion_matrix(cat_labels, cat_preds3)
        cr = classification_report(cat_labels, cat_preds3)
        f1, acc, total = EXPR_metric(cat_preds3, cat_labels)
        print(f'f1 = {f1} \n'
              f'acc = {acc} \n'
              f'total = {total} \n',
              'confusion metrix: \n', cm, '\n',
              'classification report: \n', cr, '\n')

    return 0

def setup(df_valid, args):
    if args.net == "RNN":
        valid_dataset = Aff2_Dataset_series_shuffle(df=df_valid, root=False,
                                                    transform=test_transform, type_partition='ex',
                                                    length_seq=args.seq_len)
    else:
        valid_dataset = Aff2_Dataset_static_shuffle(df=df_valid, root=False,
                                                    transform=test_transform, type_partition='ex')

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=args.batch_size,
                              num_workers=12,
                              shuffle=False,
                              drop_last=False)

    return valid_loader

if __name__ == '__main__':
    args = args_config()
    seed_everything()

    # valid set
    df_valid = create_original_data(
        f'/data/users/ys221/data/ABAW/labels_save/expression/Validation_Set_{args.dataset}/*')

    list_labels_ex = np.array(df_valid['labels_ex'].values)
    weight, num_class_list = ex_count_weight(list_labels_ex)
    weight = weight.to(device)
    print("Exp weight: ", weight)

    valid_loader = setup(df_valid, args)


    # Load model checkpoint.
    print('==> Valid from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(
        f'./checkpoint/expression/{args.net}_{args.dataset}_{args.mode}/{args.file}_1.pth')
    net = checkpoint['net']
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)

    model = BaselineESS(num_classes=args.num_classes)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.load_state_dict(net.state_dict())
    model.to(device)

    # loss
    para_dict = {
        "num_classes": args.num_classes,
        "num_class_list": num_class_list,
        "device": device,
        "cfg": cfg_loss,
    }
    if args.loss == 'CrossEntropyLabelAwareSmooth':
        criterion = CrossEntropyLabelAwareSmooth(para_dict=para_dict)
    elif args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = BalancedSoftmaxCE(para_dict=para_dict)
        # assert AttributeError

    itr = 0

    labels_all = []
    preds_all = []

    with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
        for epoch in t:
            t.set_description('Epoch %i' % epoch)
            valid(valid_loader, criterion, model)




