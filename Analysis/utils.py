import pandas as pd
import numpy as np
from scipy.special import rel_entr, entr, xlogy
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
import seaborn as sns

# def laplace_pred_mat(P, lap_eps):
#     P+=lap_eps
#     row_sums = P.sum(axis=1)
#     normalized_P = P.div(row_sums, axis=0)
#     return normalized_P
    
# def calc_kl_np(P, Q, lap_eps=1e-5):
#     P = laplace_pred_mat(P, lap_eps)
#     Q = laplace_pred_mat(Q, lap_eps)
#     return np.sum(rel_entr(P, Q), axis=1)

# # Calculate Entropy
# def calc_entropy_np(P, lap_eps=1e-5):
#     P = pd.DataFrame(F.softmax(torch.tensor(P.values),dim=1).numpy())
#     P = laplace_pred_mat(P, lap_eps)
#     return np.sum(entr(P), axis=1)

# # Calculate Cross-Entropy
# def calc_cross_ent_np(P, Q, lap_eps=1e-5):
#     P = pd.DataFrame(F.softmax(torch.tensor(P.values),dim=1).numpy())
#     Q = pd.DataFrame(F.softmax(torch.tensor(Q.values),dim=1).numpy())
#     P = laplace_pred_mat(P, lap_eps)
#     Q = laplace_pred_mat(Q, lap_eps)
#     return (-np.sum(xlogy(P,Q), axis=1))

# def calc_nll_np(p, lap_eps=1e-5):
#     P = laplace_pred_mat(P, lap_eps)
#     return -np.log(p)

def calc_entr_torch(P):
    P_tensor = torch.tensor(P.values, dtype=torch.float32)
    P_softmax = F.softmax(P_tensor, dim=1)
    P_log_softmax = F.log_softmax(P_tensor, dim=1)
    elem_wise_entr = -torch.sum(P_softmax*P_log_softmax,dim=1)
    return elem_wise_entr.numpy()
    
def calc_cross_entr_torch(P, Q):
    P_tensor = torch.tensor(P.values, dtype=torch.float32)
    Q_tensor = torch.tensor(Q.values, dtype=torch.float32)
    
    # Calculate log softmax for Q
    P_softmax = F.softmax(P_tensor, dim=1)
    Q_log_softmax = F.log_softmax(Q_tensor, dim=1)
    
    elem_wise_cross_entr = -(P_softmax*Q_log_softmax).sum(dim=1)
    
    return elem_wise_cross_entr.numpy()


def calc_kl_torch(P, Q):
    return calc_cross_entr_torch(P,Q) - calc_entr_torch(P)

def softvote(pred_ls):
    res = pred_ls[0] / len(pred_ls)
    for i in range(len(pred_ls)-1):
        res += pred_ls[i+1] / len(pred_ls)
    return res

def eval_pred(pred):
    results = []
    for col in pred.columns.drop("target"):
        if isinstance(pred[col].iloc[0], str):
            acc = pred.apply(lambda row: str(row["target"]) in row[col], axis=1).mean()
        else:
            acc = (pred[col] == pred["target"]).mean()
        results.append({"Method": col, "Accuracy": acc})
    return pd.DataFrame(results)

def softmax(P):
    P_tensor = torch.tensor(P.values, dtype=torch.float32)
    P_softmax = F.softmax(P_tensor, dim=1)
    return P_softmax.numpy()
    
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def brier_score(y, p):
  """Compute the Brier score.

  Brier Score: see
  https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf,
  page 363, Example 1

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p: numpy array, size (?, num_classes)
       containing the output predicted probabilities
  Returns:
    bs: Brier score.
  """
  return np.mean(np.power(p - y, 2))


def calibration(y, p_mean, num_bins=10):
  """Compute the calibration.

  References:
  https://arxiv.org/abs/1706.04599
  https://arxiv.org/abs/1807.00263

  Args:
    y: one-hot encoding of the true classes, size (?, num_classes)
    p_mean: numpy array, size (?, num_classes)
           containing the mean output predicted probabilities
    num_bins: number of bins

  Returns:
    ece: Expected Calibration Error
    mce: Maximum Calibration Error
  """
  # Compute for every test sample x, the predicted class.
  class_pred = np.argmax(p_mean, axis=1)
  # and the confidence (probability) associated with it.
  conf = np.max(p_mean, axis=1)
  # Convert y from one-hot encoding to the number of the class
  y = np.argmax(y, axis=1)
  # Storage
  acc_tab = np.zeros(num_bins)  # empirical (true) confidence
  mean_conf = np.zeros(num_bins)  # predicted confidence
  nb_items_bin = np.zeros(num_bins)  # number of items in the bins
  tau_tab = np.linspace(0, 1, num_bins+1)  # confidence bins
  for i in np.arange(num_bins):  # iterate over the bins
    # select the items where the predicted max probability falls in the bin
    # [tau_tab[i], tau_tab[i + 1)]
    sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
    nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
    # select the predicted classes, and the true classes
    class_pred_sec, y_sec = class_pred[sec], y[sec]
    # average of the predicted max probabilities
    mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
    # compute the empirical confidence
    acc_tab[i] = np.mean(
        class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

  # Cleaning
  mean_conf = mean_conf[nb_items_bin > 0]
  acc_tab = acc_tab[nb_items_bin > 0]
  nb_items_bin = nb_items_bin[nb_items_bin > 0]

  # Expected Calibration Error
  ece = np.average(
      np.absolute(mean_conf - acc_tab),
      weights=nb_items_bin.astype(float) / np.sum(nb_items_bin))
  # Maximum Calibration Error
  mce = np.max(np.absolute(mean_conf - acc_tab))
  return ece, mce

def auroc(pred_df, unc_df, pred_vec, target_vec, unc_vec, ax):
    label = f"{unc_vec} on {pred_vec}"
    if isinstance(pred_df[pred_vec].iloc[0],str):
        is_correct = pred_df.apply(lambda row: str(row[target_vec]) not in row[pred_vec], axis=1)
    else:
        is_correct = pred_df[pred_vec]!=pred_df[target_vec]
    fpr, tpr, threshold = metrics.roc_curve(is_correct, unc_df[unc_vec])
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label = f'{label}: AUC = %0.3f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

def get_rank(unc_pred):
    rank = pd.DataFrame()
    for curr_metric in unc_pred.columns:
        rank[curr_metric] = unc_pred[curr_metric].rank()
    return rank

def acc_cov_tradeoff(pred_df, pred_vec, target_vec, unc_vec, rank, ax, cov_range, criteria="f1"):
    temp=pd.DataFrame()
    temp["coverage"] = cov_range
    coverage_ls = (cov_range+1)/100*pred_df.shape[0]
    for i, cov in enumerate(coverage_ls):
        cov_pred = pred_df[rank[unc_vec]<cov]
        temp.loc[i,"acc"] = np.mean(cov_pred[pred_vec]==cov_pred[target_vec])
        temp.loc[i,"f1"] = metrics.f1_score(cov_pred[target_vec], cov_pred[pred_vec], average='macro')
    area=np.sum(temp[criteria])
    label = f"(AUC: {area:.3f}) - {unc_vec} on {pred_vec}"
    sns.lineplot(temp,x="coverage",y=f"{criteria}",label=label,ax = ax)
    ax.set_ylabel(f'{criteria}')
    ax.set_xlabel('Coverage')
    ax.legend(loc = 'upper right')
    
def auroc_ood(pred_df, unc_df, pred_vec, unc_vec, ax):
    label = f"{unc_vec}"
    is_ood = np.where(pred_df["test_type"].isin([ "ood and cor", "ood and inc"]),True,False)
    fpr, tpr, threshold = metrics.roc_curve(is_ood, unc_df[unc_vec])
    roc_auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr, label = f'{label}: AUC = %0.3f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')