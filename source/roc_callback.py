from sklearn.metrics import roc_auc_score, average_precision_score
import keras
import numpy as np



class ROC_Callback(keras.callbacks.Callback):
    def __init__(self, val_generator, fout=None):

        self.val_generator = val_generator
        self.fout = fout

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):

        b_pred = self.model.predict_generator(generator=self.val_generator)

        b_true = np.concatenate([self.val_generator.__getitem__(i)[-1] for i in range(len(self.val_generator))])

        if self.fout is not None:
            self.fout.write('\n------------End of an epoch--------\n')

        #        b_roc = roc_auc_score(b_true.reshape(-1), b_pred.reshape(-1))
        #        print('Boundary ROC: %.2f' %(b_roc))
        #        if self.fout is not None:
        #            self.fout.write('Boundary ROC: %.2f\n' %(b_roc))

        # if len(b_true.shape) > 1:
        #
        #     # with open("feature2id.json", "r") as fin:
        #     #     feature2id = json.load(fin)
        #     #
        #     # id2feature = {}
        #     # for k, v in feature2id.items():
        #     #     id2feature[v] = k
        #
        #     roc = np.empty(b_true.shape[1])
        #     for i in range(b_true.shape[1]):
        #         if len(set(b_true[:, i])) > 1:  # in case there is only one class during testing, validation
        #             roc[i] = roc_auc_score(b_true[:, i], b_pred[:, i])
        #             # print('feature: %s, class: %d, roc: %.2f' %(id2feature[i], i, roc[i]))
        #             # if self.fout is not None:
        #             #     self.fout.write('feature: %s, class: %d, roc: %.2f\n' % (id2feature[i], i, roc[i]))
        #
        #     print('meadian val_roc-auc: %.2f, min: %.2f, max: %.2f\n' % (
        #     np.median(roc[roc > 0]), np.min(roc[roc > 0]), np.max(roc[roc > 0])))
        #     if self.fout is not None:
        #         self.fout.write('meadian val_roc-auc: %.2f, min: %.2f, max: %.2f\n' % (
        #         np.median(roc[roc > 0]), np.min(roc[roc > 0]), np.max(roc[roc > 0])))
        #
        # else:
            roc = roc_auc_score(b_true.reshape(-1), b_pred.reshape(-1))
            pr_auc = average_precision_score(b_true.reshape(-1), b_pred.reshape(-1))

            print('val_roc-auc: %.4f, val_pr_auc: %.4f\n' % (roc, pr_auc))
            if self.fout is not None:
                self.fout.write ('val_roc-auc: %.4f, val_pr_auc: %.4f\n' % (roc, pr_auc))

        if self.fout is not None:
            self.fout.write('\n---------------------------\n')
            self.fout.flush()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


