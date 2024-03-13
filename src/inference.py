import torch
from transformers import get_linear_schedule_with_warmup
from torch import optim
import torch.nn.functional as F
from statistics import mean
import time
from cross_encoder import BertClause
import data_loader
from utils.utils import *
# from allennlp.training.learning_rate_schedulers.noam import NoamLR

class Framework(object):
    def __init__(self, con):
        self.config = con
        print("../../models/" + self.config.model_name)
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(self.config.model_dir, self.config.model_name))
        self.id2subreddits = {0: 'android', 1: 'apple', 2: 'technology', 3: 'dota2', 4: 'playstation', 5: 'movies', 6: 'nba', 7: 'steam'}
        self.subreddit2ids = {'android': 0, 'apple': 1, 'technology': 2, 'dota2': 3, 'playstation': 4, 'movies': 5, 'nba': 6, 'steam': 7}

    def logging(self, s, print_=True, log_=True):
        log_path = os.path.join(self.config.log_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        else:
            if print_:
                print(s)
            if log_:
                with open(os.path.join(self.config.log_dir, self.config.log_save_name), 'a+') as f_log:
                    f_log.write(s + '\n')

    def sep_params(self, model):
        """Separate the parameters into loaded and not loaded."""
        small_lr_params = dict()
        large_lr_params = dict()
        for n, p in model.named_parameters():
            if re.search(r'(.*)(bert\.)(.*)', n):
                # print(n)
                small_lr_params[n] = p
            else:
                large_lr_params[n] = p

        return small_lr_params, large_lr_params

    def train(self):
        # check the checkpoint dir
        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        run_doc_accs = []
        run_precisions = []
        run_recalls = []
        run_f1s = []
        run_subreddit_precisions = {'android': [], 'apple': [], 'technology': [], 'dota2': [], 'playstation': [], 'movies': [], 'nba': [], 'steam': []}
        run_subreddit_recalls = {'android': [], 'apple': [], 'technology': [], 'dota2': [], 'playstation': [], 'movies': [], 'nba': [], 'steam': []}
        run_subreddit_f1s = {'android': [], 'apple': [], 'technology': [], 'dota2': [], 'playstation': [], 'movies': [], 'nba': [], 'steam': []}
        for run in range(1, 6):
            # initialize the model
            print(os.path.join(self.config.model_dir, self.config.model_name))

            # get the data loader
            train_data_loader = data_loader.get_loader(self.config, prefix=self.config.train_prefix)
            print("Train", len(train_data_loader))
            dev_data_loader = data_loader.get_loader(self.config, prefix=self.config.dev_prefix)
            print("Dev", len(dev_data_loader))
            test_data_loader = data_loader.get_loader(self.config, prefix=self.config.test_prefix)
            print("Test", len(test_data_loader))

            num_steps_all = len(train_data_loader) * self.config.max_epoch
            self.logging("Num_steps_all {}.".format(num_steps_all))
            warmup_steps = int(num_steps_all * self.config.warmup_proportion)
            self.logging("Nums_steps_warmup {}.".format(warmup_steps))

            self.logging("Run {} starts.".format(run))

            # other
            best_epoch = 0

            torch.cuda.empty_cache()
            best_model = BertClause.from_pretrained(os.path.join(self.config.checkpoint_dir, str(run)),
                                               clause_max_len=self.config.clause_max_len, max_len=self.config.max_len)
            best_model.cuda()
            doc_acc, sent_precision, sent_recall, sent_f1, subreddit_precisions,subreddit_recalls,subreddit_f1s = self.test(test_data_loader, best_model, test = True)
            run_doc_accs.append(doc_acc)
            run_precisions.append(sent_precision)
            run_recalls.append(sent_recall)
            run_f1s.append(sent_f1)
            for subreddit in self.subreddit2ids.keys():
                run_subreddit_precisions[subreddit].append(subreddit_precisions[subreddit])
                run_subreddit_recalls[subreddit].append(subreddit_recalls[subreddit])
                run_subreddit_f1s[subreddit].append(subreddit_f1s[subreddit])
            self.logging(
                "Run {} test performance, epoch: {:3d}, doc_acc: {:5.4f}, sent_precision: {:5.4f}, sent_recall: {:5.4f}, sent_f1: {:5.4f}".
                    format(run, best_epoch, doc_acc, sent_precision, sent_recall, sent_f1))
        avg_doc_acc = mean(run_doc_accs)
        avg_precision = mean(run_precisions)
        avg_recall = mean(run_recalls)
        avg_f1 = mean(run_f1s)
        self.logging(
            "Average performance, doc_acc: {:5.4f}, sent_precision: {:5.4f}, sent_recall: {:5.4f}, sent_f1: {:5.4f}".
                format(avg_doc_acc, avg_precision, avg_recall, avg_f1))
        avg_subreddit_precisions = {}
        avg_subreddit_recalls = {}
        avg_subreddit_f1s = {}
        for subreddit in self.subreddit2ids.keys():
            avg_subreddit_precisions[subreddit] = mean(run_subreddit_precisions[subreddit])
            avg_subreddit_recalls[subreddit] = mean(run_subreddit_recalls[subreddit])
            avg_subreddit_f1s[subreddit] = mean(run_subreddit_f1s[subreddit])
            self.logging(
                "Average performance, subreddit: {}, sent_precision: {:5.4f}, sent_recall: {:5.4f}, sent_f1: {:5.4f}". 
                    format(subreddit, avg_subreddit_precisions[subreddit], avg_subreddit_recalls[subreddit], avg_subreddit_f1s[subreddit])
            )

    def test(self, test_data_loader, model, test=False):
        model.eval()
        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        doc_correct = 0
        doc_all = 0
        id_ = 0
        pred_ids = set()
        pred_ids_0 = set()
        annot_ids = set()
        annot_ids_0 = set()
        clause_preds_list = []
        clause_labels_list = []
        subreddit_true_labels = {'android': [], 'apple': [], 'technology': [], 'dota2': [], 'playstation': [], 'movies': [], 'nba': [], 'steam': []}
        subreddit_pred_labels = {'android': [], 'apple': [], 'technology': [], 'dota2': [], 'playstation': [], 'movies': [], 'nba': [], 'steam': []}
        with torch.no_grad():
            while data is not None:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                input_mask = data["input_mask"]
                token_type_ids = data["token_type_ids"]
                position_ids = data["position_ids"]
                subreddit_input_ids = data["subreddit_input_ids"]
                subreddit_attention_mask = data["subreddit_attention_mask"]
                subreddit_token_type_ids = data["subreddit_token_type_ids"]
                subreddit_position_ids = data["subreddit_position_ids"]
                clause_idx = data["clause_idx"]
                clause_labels = data["clause_labels"]
                sarcasm_idx = data["sarcasm_idx"]
                subreddit_idx = data["subreddit_idx"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, input_mask=input_mask,
                                subreddit_input_ids=subreddit_input_ids,
                                subreddit_attention_mask=subreddit_attention_mask,
                                subreddit_token_type_ids=subreddit_token_type_ids,
                                subreddit_position_ids=subreddit_position_ids,
                                clause_idx=clause_idx,
                                sarcasm_idx=sarcasm_idx,
                                subreddit_idx=subreddit_idx,
                                clause_labels=clause_labels)
                clause_logits = outputs.clause_logits
                clause_preds = torch.argmax(F.softmax(clause_logits, dim=-1), dim=-1).detach().cpu().numpy().tolist()
                clause_labels = clause_labels.detach().cpu().numpy().tolist()
                for subreddit_id, clause_pred, clause_label in zip(subreddit_idx.detach().cpu().numpy().tolist(), clause_preds, clause_labels):
                    flag = True
                    doc_all += 1
                    for pred, label in zip(clause_pred, clause_label):
                        if label != -100:
                            id_ += 1
                            if pred == 1:
                                pred_ids.add(id_)
                            if label == 1:
                                annot_ids.add(id_)
                            if pred == 0:
                                pred_ids_0.add(id_)
                            if label == 0:
                                annot_ids_0.add(id_)
                            if pred != label:
                                flag = False
                            subreddit = self.id2subreddits[subreddit_id[0]]
                            subreddit_true_labels[subreddit].append(label)
                            subreddit_pred_labels[subreddit].append(pred)
                    if flag:
                        doc_correct += 1
                clause_preds_list.extend(clause_preds)
                clause_labels_list.extend(clause_labels)
                data = test_data_prefetcher.next()
            if test:
                with open(os.path.join(self.config.log_dir, "output.json"), "w", encoding="utf-8") as output_file:
                    json.dump({"preds":clause_preds_list, "ture":clause_labels_list}, output_file, indent=4)
            print("sents:", id_, "docs:", doc_all)
            sent_precision, sent_recall, sent_f1 = sent_metrics(pred_ids, annot_ids)
            sent_precision_0, sent_recall_0, sent_f1_0 = sent_metrics(pred_ids_0, annot_ids_0)
            subreddit_precision = {}
            subreddit_recall = {}
            subreddit_f1 = {}
            if test:
                for subreddit in self.subreddit2ids.keys():
                    subreddit_precision[subreddit], subreddit_recall[subreddit], subreddit_f1[subreddit] = sent_metrics(set([i for i, pred in enumerate(subreddit_pred_labels[subreddit]) if pred == 1]), set([i for i, label in enumerate(subreddit_true_labels[subreddit]) if label == 1]))
                    self.logging("subreddit: {}".format(subreddit))
                    self.logging("precision: {:4.5f}, recall: {:4.5f}, f1: {:4.5f}".format(subreddit_precision[subreddit], subreddit_recall[subreddit], subreddit_f1[subreddit]))
            self.logging("avg f1 {:4.5f}".format((sent_f1 + sent_f1_0) / 2))
            return doc_correct / doc_all, sent_precision, sent_recall, sent_f1, subreddit_precision, subreddit_recall, subreddit_f1

    def estimate_transformer_input_stats(self, test_data_loader, model):
        model.eval()
        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        max_norm = 0.0
        clause_batch_max_norm = 0.0
        con_feature_batch_max_norm = 0.0
        with torch.no_grad():
            while data is not None:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                input_mask = data["input_mask"]
                token_type_ids = data["token_type_ids"]
                position_ids = data["position_ids"]
                subreddit_input_ids = data["subreddit_input_ids"]
                subreddit_attention_mask = data["subreddit_attention_mask"]
                subreddit_token_type_ids = data["subreddit_token_type_ids"]
                subreddit_position_ids = data["subreddit_position_ids"]
                clause_idx = data["clause_idx"]
                clause_labels = data["clause_labels"]
                sarcasm_idx = data["sarcasm_idx"]
                subreddit_idx = data["subreddit_idx"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, input_mask=input_mask,
                                subreddit_input_ids=subreddit_input_ids,
                                subreddit_attention_mask=subreddit_attention_mask,
                                subreddit_token_type_ids=subreddit_token_type_ids,
                                subreddit_position_ids=subreddit_position_ids,
                                clause_idx=clause_idx,
                                sarcasm_idx=sarcasm_idx,
                                subreddit_idx=subreddit_idx,
                                clause_labels=clause_labels)

                input_features, subreddit_features = outputs.input_embeddings
                clause_embeddings = input_features
                subreddit_embeddings = subreddit_features
                for i in range(clause_embeddings.shape[0]):
                    clause_embedding = clause_embeddings[i]
                    clause_batch_max_norm = max(clause_batch_max_norm, torch.norm(clause_embedding, p=2, dim=-1).max().item())
                max_norm = max(max_norm, clause_batch_max_norm)
                for i in range(subreddit_embeddings.shape[0]):
                    subreddit_embedding = subreddit_embeddings[i]
                    clause_batch_max_norm = max(clause_batch_max_norm, torch.norm(subreddit_embedding, p=2, dim=-1).max().item())
                max_norm = max(max_norm, clause_batch_max_norm)
                data = test_data_prefetcher.next()
            return max_norm

    def estimate_transformer_output_stats(self, test_data_loader, model):
        model.eval()
        test_data_prefetcher = data_loader.DataPreFetcher(test_data_loader)
        data = test_data_prefetcher.next()
        max_norm = 0.0
        clause_batch_max_norm = 0.0
        with torch.no_grad():
            while data is not None:
                input_ids = data["input_ids"]
                attention_mask = data["attention_mask"]
                input_mask = data["input_mask"]
                token_type_ids = data["token_type_ids"]
                position_ids = data["position_ids"]
                subreddit_input_ids = data["subreddit_input_ids"]
                subreddit_attention_mask = data["subreddit_attention_mask"]
                subreddit_token_type_ids = data["subreddit_token_type_ids"]
                subreddit_position_ids = data["subreddit_position_ids"]
                clause_idx = data["clause_idx"]
                clause_labels = data["clause_labels"]
                sarcasm_idx = data["sarcasm_idx"]
                subreddit_idx = data["subreddit_idx"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                position_ids=position_ids, input_mask=input_mask,
                                subreddit_input_ids=subreddit_input_ids,
                                subreddit_attention_mask=subreddit_attention_mask,
                                subreddit_token_type_ids=subreddit_token_type_ids,
                                subreddit_position_ids=subreddit_position_ids,
                                clause_idx=clause_idx,
                                sarcasm_idx=sarcasm_idx,
                                subreddit_idx=subreddit_idx,
                                clause_labels=clause_labels)
                clause_embeddings,  = outputs.output_embeddings
                batch_max_norm = 0
                for i in range(clause_embeddings.shape[0]):
                    clause_embedding = clause_embeddings[i]
                    clause_batch_max_norm = max(clause_batch_max_norm, torch.norm(clause_embedding, p=2, dim=-1).max().item())
                    batch_max_norm = max(clause_batch_max_norm, clause_batch_max_norm)
                max_norm = max(max_norm, batch_max_norm)
                data = test_data_prefetcher.next()
            return max_norm






