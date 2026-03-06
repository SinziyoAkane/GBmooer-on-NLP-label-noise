# -*- coding: utf-8 -*-
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os, logging, warnings
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import HyperBallCluster

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')  # 每一类预测对的占比取平均
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


def adjust_probabilities(arr, reduce_total=0.2, threshold=0.6):
    arr_new = arr.copy()
    n_rows, n_cols = arr.shape

    for i in range(n_rows):
        row = arr_new[i]
        target_indices = np.where(row >= threshold)[0]
        if len(target_indices) == 0:
            continue

        target = target_indices[0]
        others = [j for j in range(n_cols) if j != target]
        if not others:
            continue

        share = reduce_total / len(others)
        deducted_total = 0.0

        for j in others:
            deduction = min(row[j], share)
            row[j] -= deduction
            deducted_total += deduction

        row[target] += deducted_total
        row /= row.sum()

    return arr_new


class SelfMixTrainer:
    def __init__(self, model, train_data=None, eval_data=None, train_sample=None, model_args=None, training_args=None,
                 data_args=None, noise=False, rank=None):
        self.rank = rank
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.train_sample = train_sample
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        self.noise = noise
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.KLDivLoss(reduction="batchmean")

        if self.training_args is not None:
            self.optimizer = Adam(self.model.parameters(), lr=training_args.lr)

    def dynamic_train(self):
        test_last_l = []

        test_best = 0.0
        train_loader = self.train_data.run(mode="all", sampler=False)
        eval_loader = self.eval_data.run(mode="all", sampler=False)  # 之前没有

        if self.rank == 0:
            logging.info("Training begin...")
            print("Training begin...")

        for epoch in range(1, self.training_args.train_epochs + 1):

            self.model.train()
            epoch_loss = 0
            for idx, (input_ids, att_mask, labels_id, true_labels, index) \
                    in enumerate(tqdm(train_loader, desc=f"Rank {self.rank} Epoch {epoch}")):

                input_ids, att_mask, labels_id, true_labels, index = (
                    input_ids.cuda(self.rank), att_mask.cuda(self.rank), labels_id.cuda(self.rank),
                    true_labels.cuda(self.rank), index.cuda(self.rank))

                sents_x = self.model.get_sentence_embedding(input_ids, att_mask)
                sents_x2 = self.model.get_sentence_embedding(input_ids, att_mask)
                logits_x = self.model.classify(sents_x)
                logits_x2 = self.model.classify(sents_x2)

                if epoch <= 1:
                    file_path = "./no_data_40.csv"
                    sents = sents_x.detach().cpu().numpy()
                    l = true_labels.detach().cpu().numpy().reshape(-1, 1)
                    data = np.hstack([l, sents])  # shape (4, 769)
                    df = pd.DataFrame(data)
                    df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
                    continue

                # if self.noise:
                if epoch < 2:
                    loss_cl = self.criterion1(logits_x, true_labels)
                else:
                    result, centers = HyperBallCluster.main(sents_x, true_labels, self.training_args.min_synonyms)
                    soft_labels = HyperBallCluster.compute_soft_label(result, sents_x,
                                                                      label_num=self.model_args.num_classes)

                    # adjust_soft_labels = adjust_probabilities(soft_labels)
                    targets_x = torch.tensor(soft_labels).cuda(self.rank)

                    log_probs = torch.log(F.softmax(logits_x) + 1e-8)
                    loss_cl = self.criterion2(log_probs, targets_x)
                    # loss_cl = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=-1) * targets_x, dim=-1))

                kl_loss = compute_kl_loss(logits_x, logits_x2)

                loss = loss_cl + kl_loss * self.training_args.lambda_r

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            if self.rank == 0:
                print(f"Epoch {epoch}/{self.training_args.train_epochs}, Loss: {avg_epoch_loss:.4f}")
                logging.info(f"Epoch {epoch}/{self.training_args.train_epochs}, Loss: {avg_epoch_loss:.4f}")
                # torch.distributed.barrier()

            if epoch % 1 == 0 and self.rank == 0:
                logging.info("Testing Stage in %d epoch", int(epoch))
                print(f"Testing Stage in {epoch} epoch")
                current_test, _, _ = self.evaluate(eval_loader)
                if current_test > test_best:
                    test_best = current_test
                    self.save_model()
                    logging.info(
                        f"Test|{epoch}|Current Accuracy: {current_test * 100:.2f}% | Best Accuracy: {test_best * 100:.2f}%")
                print(
                    f"Test|{epoch}|Current Accuracy: {current_test * 100:.2f}% | Best Accuracy: {test_best * 100:.2f}%")
                test_last_l.append(current_test)
                # torch.distributed.barrier()
            # torch.distributed.barrier()
            del sents_x, sents_x2, logits_x, logits_x2
            # torch.cuda.empty_cache()

        return test_best, test_last_l

    def evaluate(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.eval_data.run("all")
        self.model.eval()
        y_true, y_pred = np.zeros(len(eval_loader.dataset), dtype=int), np.zeros(len(eval_loader.dataset), dtype=int)
        for j, data in enumerate(eval_loader):
            val_input_ids, val_att, val_labels, _, index = [Variable(elem.cuda()) for elem in data]
            with torch.no_grad():
                index = index.long().cpu().detach().numpy()
                pred = self.model(val_input_ids, val_att).argmax(dim=-1).cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
            y_true[index] = val_labels
            y_pred[index] = pred

        eval_res = metric(y_true, y_pred)
        logging.info("Eval Results: Accuracy: {:.4%}, Precision: {:.4%}, Recall: {:.4%}, F1: {:.4%}"
                     .format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1']))
        self.model.train()
        return eval_res['accuracy'], eval_res['f1'], eval_res['recall']

    def save_model(self, comm=None):
        suffix = '.pth'
        if comm:
            suffix = (self.data_args.dataset_name + '_' + str(self.training_args.noise_ratio) + '_' +
                      self.training_args.noise_type + '_' + str(self.training_args.train_epochs) + '_' + str(comm) + suffix)
        else:
            suffix = (self.data_args.dataset_name + '_' + str(self.training_args.noise_ratio) +
                      '_' + self.training_args.noise_type + '_' + str(self.training_args.train_epochs)
                      + '_' + str(self.training_args.min_synonyms) + suffix)

        path = os.path.join(self.training_args.model_save_path, suffix)

        torch.save(self.model.state_dict(), path)

    def load_model(self, comm=None):
        suffix = '.pth'
        if comm:
            suffix = '_' + str(comm) + suffix
        path = self.training_args.model_save_path + suffix
        model_state_dict = torch.load(path)
        self.model.load_state_dict(model_state_dict)
