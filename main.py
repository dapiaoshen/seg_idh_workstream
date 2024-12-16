import argparse
import logging
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
from models.Fusion import *
from data.BraTS_IDH import BraTS
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score,accuracy_score
from models.TransBTS.TransBTS_downsample8x_skipconnection_2 import TransBTS_1,Decoder_modual,IDH_network,Grade_netwoek
import csv
import nibabel as nib

local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()

# Basic Information
parser.add_argument('--user', default='UCSF', type=str)

parser.add_argument('--experiment', default='TransBTS_Boundary', type=str)

parser.add_argument('--date', default=local_time.split(' ')[0], type=str)

parser.add_argument('--description',
                    default='TransBTS_Boundary,'
                            'training on train.txt!',
                    type=str)

# DataSet Information
parser.add_argument('--root', default='./test_data', type=str)

parser.add_argument('--test_dir', default='test', type=str)

parser.add_argument('--mode', default='test', type=str)

parser.add_argument('--test_file', default='test.txt', type=str)

parser.add_argument('--dataset', default='UCSF', type=str)

parser.add_argument('--model_name', default='TransBTS', type=str)

parser.add_argument('--amsgrad', default=True, type=bool)

parser.add_argument('--criterion', default='softmax_dice', type=str)#softmax_dice

parser.add_argument('--no_cuda', default=False, type=bool)

parser.add_argument('--gpu', default='2', type=str)

parser.add_argument('--num_workers', default=2, type=int)

parser.add_argument('--start_epoch', default=0, type=int)

parser.add_argument('--load', default='./checkpoint/model.pth', type=str)

parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')

args = parser.parse_args()

def main_worker():
    if args.local_rank == 0:
        log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
        log_file = log_dir + '.txt'
        logging.info('--------------------------------------This is all argsurations----------------------------------')
        for arg in vars(args):
            logging.info('{}={}'.format(arg, getattr(args, arg)))
        logging.info('----------------------------------------This is a halving line----------------------------------')
        logging.info('{}'.format(args.description))


    en = TransBTS_1(dataset='brats', _conv_repr=True, _pe_type="learned")
    de = Decoder_modual()
    idh = IDH_MTTU_Predict()
    tf = BilinearFusion()
    model_dict = torch.load(args.load)
    en_dict = model_dict['en']
    en.load_state_dict(en_dict, strict=False)

    de_dict = model_dict['de']
    de.load_state_dict(de_dict, strict=False)

    idh_dict = model_dict['idh']
    idh.load_state_dict(idh_dict, strict=False)

    fusion_dict=model_dict['fusion']
    tf.load_state_dict(fusion_dict, strict=False)

    nets = {
        'en': en.cuda(),
        'de': de.cuda(),
        'idh': idh.cuda(),
        'fusion': tf.cuda(),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_list = os.path.join(args.root, args.test_dir, args.test_file)
    test_root = os.path.join(args.root, args.test_dir)
    test_set = BraTS(test_list, test_root, 'valid')

    logging.info('Samples for test = {}'.format(len(test_set)))
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    roott = "./MTTU_idh_model"
    segmentation_path = os.path.join(roott, 'segmentation.nii.gz')

    # 加载 segmentation.nii.gz 以获取 affine
    if not os.path.exists(segmentation_path):
        raise FileNotFoundError(f"The file {segmentation_path} does not exist.")

    reference_nifti = nib.load(segmentation_path)
    reference_affine = reference_nifti.affine
    checkpoint_dir = os.path.join(roott, 'checkpoint', args.experiment + args.date)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    seg_dir = os.path.join(roott, 'seg')
    if not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    csv_path = os.path.join(roott, 'test_results.csv')

    labels = [0, 1]
    idh_names = ['IDH野生型', 'IDH突变型']
    idh_pred_scores_test, idh_preds_test, idh_trues_test = [], [], []
    wt_dices_test, tc_dices_test, et_dices_test = [], [], []

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Sample Name', 'True IDH', 'Predicted IDH', 'WT Dice', 'TC Dice', 'ET Dice'])

        with torch.no_grad():
            nets['en'].eval()
            nets['de'].eval()
            nets['idh'].eval()
            nets['fusion'].eval()

            for i, data in enumerate(test_loader):
                x, target, idh, grade, name = data
                x = x.cuda(args.local_rank, non_blocking=True).float()
                target = target.cuda(args.local_rank, non_blocking=True)
                idh = idh.cuda(args.local_rank, non_blocking=True)

                x1_1, x2_1, x3_1, x4_1, encoder_output = nets['en'](x)
                output = nets['de'](x1_1, x2_1, x3_1, encoder_output)
                feature = nets['fusion'](encoder_output, x4_1)
                idh_p = nets['idh'](feature)

                pred_labels = torch.argmax(output, dim=1)
                mapping = {3: 4}
                for k, v in mapping.items():
                    pred_labels[pred_labels == k] = v

                for idx in range(pred_labels.shape[0]):
                    wt_dice = dice_coeff(pred_labels[idx], target[idx], [1, 2, 4])
                    tc_dice = dice_coeff(pred_labels[idx], target[idx], [1, 4])
                    et_dice = dice_coeff(pred_labels[idx], target[idx], [4])
                    wt_dices_test.append(wt_dice.item())
                    tc_dices_test.append(tc_dice.item())
                    et_dices_test.append(et_dice.item())

                    # 保存分割结果为nii.gz文件
                    pred_nifti = nib.Nifti1Image(pred_labels[idx].cpu().numpy().astype(np.uint8), reference_affine)
                    nib.save(pred_nifti, os.path.join(seg_dir, f'{name[idx]}_seg.nii.gz'))

                    csv_writer.writerow([
                        name[idx],
                        idh[idx].item(),
                        torch.argmax(idh_p[idx]).item(),
                        wt_dice.item(),
                        tc_dice.item(),
                        et_dice.item()
                    ])

                idh_pred_scores_test.extend(F.softmax(idh_p, dim=1)[:, 1].cpu().numpy())
                idh_preds_test.extend(torch.argmax(idh_p, dim=1).cpu().numpy())
                idh_trues_test.extend(idh.cpu().numpy())

    idh_trues_test = np.array(idh_trues_test).tolist()
    idh_preds_test = np.array(idh_preds_test).tolist()
    idh_pred_scores_test = np.array(idh_pred_scores_test).tolist()

    results_idh_test = evalution_metirc_boostrap(
        y_true=idh_trues_test,
        y_pred_score=idh_pred_scores_test,
        y_pred=idh_preds_test,
        labels=labels,
        target_names=idh_names
    )

    wt_test_mean = np.mean(wt_dices_test)
    wt_test_std = np.std(wt_dices_test)

    tc_test_mean = np.mean(tc_dices_test)
    tc_test_std = np.std(tc_dices_test)

    et_test_mean = np.mean(et_dices_test)
    et_test_std = np.std(et_dices_test)

    print('测试集 IDH 真实结果:', idh_trues_test)
    print('测试集 IDH 预测结果:', idh_preds_test)

    print(f'WT Dice 平均值: {wt_test_mean}, 标准差: {wt_test_std}')
    print(f'TC Dice 平均值: {tc_test_mean}, 标准差: {tc_test_std}')
    print(f'ET Dice 平均值: {et_test_mean}, 标准差: {et_test_std}')

def dice_coeff(pred, target, class_labels):
        class_labels = torch.tensor(class_labels, device=pred.device)  # 将列表转换为张量
        pred = torch.isin(pred, class_labels).float()
        target = torch.isin(target, class_labels).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return dice

def evalution_metirc_boostrap(y_true, y_pred_score, y_pred, labels, target_names):
    y_true = np.array(y_true)
    y_pred_score = np.array(y_pred_score)
    y_pred = np.array(y_pred)
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names))

    auc_ = roc_auc_score(y_true, y_pred_score)
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)

    accuracy_ = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    specificity_ = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    sensitivity_ = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    F1_score_ = f1_score(y_true, y_pred, labels=labels, pos_label=1)

    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_AUC = []
    bootstrapped_ACC = []
    bootstrapped_SEN = []
    bootstrapped_SPE = []
    bootstrapped_F1 = []
    rng = np.random.RandomState(rng_seed)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred_score), len(y_pred_score))
        if len(np.unique(y_true[indices.astype(int)])) < 2:
            # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
            continue
        auc = roc_auc_score(y_true[indices], y_pred_score[indices])
        bootstrapped_AUC.append(auc)

        confusion = confusion_matrix(y_true[indices], y_pred[indices])
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
        F1_score = f1_score(y_true[indices], y_pred[indices], labels=labels, pos_label=1)

        bootstrapped_ACC.append(accuracy)
        bootstrapped_SPE.append(specificity)
        bootstrapped_SEN.append(sensitivity)
        bootstrapped_F1.append(F1_score)

    sorted_AUC = np.array(bootstrapped_AUC)
    sorted_AUC.sort()
    sorted_ACC = np.array(bootstrapped_ACC)
    sorted_ACC.sort()
    sorted_SPE = np.array(bootstrapped_SPE)
    sorted_SPE.sort()
    sorted_SEN = np.array(bootstrapped_SEN)
    sorted_SEN.sort()
    sorted_F1 = np.array(bootstrapped_F1)
    sorted_F1.sort()

    results = {
        'AUC': (auc_, sorted_AUC[int(0.05 * len(sorted_AUC))], sorted_AUC[int(0.95 * len(sorted_AUC))]),
        'Accuracy': (accuracy_, sorted_ACC[int(0.05 * len(sorted_ACC))], sorted_ACC[int(0.95 * len(sorted_ACC))]),
        'Specificity': (specificity_, sorted_SPE[int(0.05 * len(sorted_SPE))], sorted_SPE[int(0.95 * len(sorted_SPE))]),
        'Sensitivity': (sensitivity_, sorted_SEN[int(0.05 * len(sorted_SEN))], sorted_SEN[int(0.95 * len(sorted_SEN))]),
        'F1_score': (F1_score_, sorted_F1[int(0.05 * len(sorted_F1))], sorted_F1[int(0.95 * len(sorted_F1))])
    }

    print("Confidence interval for the AUC: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['AUC']))
    print("Confidence interval for the Accuracy: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Accuracy']))
    print("Confidence interval for the Specificity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Specificity']))
    print("Confidence interval for the Sensitivity: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['Sensitivity']))
    print("Confidence interval for the F1_score: {:0.4f} [{:0.4f} - {:0.4f}]".format(*results['F1_score']))

    return results

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    assert torch.cuda.is_available(), "Currently, we only support CUDA version"
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    main_worker()

