import pandas as pd
import numpy as np
from scipy.special import rel_entr, entr, xlogy
import torch
import torch.nn.functional as F
import sklearn.metrics as metrics
from sklearn.metrics import f1_score, roc_curve, auc
import seaborn as sns

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

def softmax_response_unc(P):
    P_softmax = softmax(P)
    P_softmax_response = np.max(P_softmax,axis=1)
    return 1-P_softmax_response
    
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
        rank.loc[:,curr_metric] = unc_pred[curr_metric].rank()
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
    
# Function to load predictions
# Takes in a list of models and return a list of predictions
def load_predictions(prefix, testtype, models):
    return [pd.read_csv(f"{prefix}{testtype}_{model}.csv") for model in models]

# Function to save ensemble predictions
def save_ensemble_predictions(prefix, testtype, name, predictions):
    predictions.to_csv(f"{prefix}{testtype}_{name}.csv", index=False)

# Function to generate ensemble name
def generate_ensemble_name(base_name, models):
    return f"{base_name}_{'_'.join(models)}"

# Function to evaluate models and store metrics
def evaluate_models(predictions, label, prefix, testtype, metrics_dict, category, num_classes):
    for method in predictions:
        pred_col_name = f"pred_{method}"
        pred = pd.read_csv(f"{prefix}{testtype}_{method}.csv")
        label[pred_col_name] = pred.idxmax(axis=1)
        label[pred_col_name] = label[pred_col_name].str.extract('(\d+)').astype(int)
        
        curr_acc = np.mean(label[pred_col_name]==label["target"])
        curr_f1 = f1_score(label["target"], label[pred_col_name], average='macro')
        curr_brier = brier_score(one_hot(np.array(label["target"]), num_classes), softmax(pred))
        curr_ece, curr_mce = calibration(one_hot(np.array(label["target"]), num_classes), softmax(pred))
        
        # Store metrics
        metrics_dict["Model"].append(method)
        metrics_dict["Test Set"].append(testtype)
        metrics_dict["Acc"].append(curr_acc)
        metrics_dict["F1"].append(curr_f1)
        metrics_dict["Brier"].append(curr_brier)
        metrics_dict["ECE"].append(curr_ece)
        metrics_dict["MCE"].append(curr_mce)
        metrics_dict["Predictor Category"].append(category)
    return metrics_dict

# Function to calculate AUROC
def calculate_auroc(pred_df, unc_df, pred_vec, target_vec, unc_vec):
    if isinstance(pred_df[pred_vec].iloc[0], str):
        is_correct = pred_df.apply(lambda row: str(row[target_vec]) not in row[pred_vec], axis=1)
    else:
        is_correct = pred_df[pred_vec] != pred_df[target_vec]
    fpr, tpr, _ = roc_curve(is_correct, unc_df[unc_vec])
    roc_auc = auc(fpr, tpr)
    return roc_auc

# Function to update AUROC results
def update_auroc_results(auroc_results, model, testtype, category, unc_measure, auroc_value):
    auroc_results["Model"].append(model)
    auroc_results["Test Set"].append(testtype)
    auroc_results["Uncertainty Measure"].append(unc_measure)
    auroc_results["Predictor Category"].append(category)
    auroc_results["AUROC"].append(auroc_value)
    return auroc_results

# Function to calculate F1-Coverage AUC
def calculate_f1_cov_auc(pred_df, unc_df, pred_vec, target_vec, unc_vec, cov_range):
    rank = get_rank(unc_df)
    temp = pd.DataFrame()
    temp["coverage"] = cov_range
    coverage_ls = (cov_range + 1) / 100 * pred_df.shape[0]
    for i, cov in enumerate(coverage_ls):
        cov_pred = pred_df.loc[rank[unc_vec] < cov]
        temp.loc[i, "f1"] = f1_score(cov_pred[target_vec], cov_pred[pred_vec], average='macro')
    area = np.sum(temp["f1"])
    return area

cov_range = np.arange(19, 100)

# Function to update F1-Coverage results
def update_f1_cov_results(f1_cov_results, model, testtype, category, unc_measure, f1_cov_auc_value):
    f1_cov_results["Model"].append(model)
    f1_cov_results["Test Set"].append(testtype)
    f1_cov_results["Uncertainty Measure"].append(unc_measure)
    f1_cov_results["Predictor Category"].append(category)
    f1_cov_results["F1-Cov AUC"].append(f1_cov_auc_value)
    return f1_cov_results

# Function to calculate AUROC for OOD detection
def calculate_auroc_ood(pred_df, unc_df, pred_vec, unc_vec):
    is_ood = np.where(pred_df["test_type"].isin(["ood"]), True, False)
    fpr, tpr, _ = roc_curve(is_ood, unc_df[unc_vec])
    roc_auc = auc(fpr, tpr)
    return roc_auc


# Function to update AUROC OOD results
def update_auroc_ood_results(auroc_ood_results, model, category, unc_measure, auroc_ood_value):
    auroc_ood_results["Model"].append(model)
    auroc_ood_results["Test Set"].append("combined")
    auroc_ood_results["Uncertainty Measure"].append(unc_measure)
    auroc_ood_results["Predictor Category"].append(category)
    auroc_ood_results["AUROC OOD"].append(auroc_ood_value)
    return auroc_ood_results

def generate_duos(pred_prefix, base_name, testtype, smaller_model_ls, larger_model_ls, predictor_categories):
    for smaller_model in smaller_model_ls:
        for larger_model in larger_model_ls:
            curr_duo_pred = softvote([pd.read_csv(f"{pred_prefix}{testtype}_{smaller_model}.csv"), 
                                            pd.read_csv(f"{pred_prefix}{testtype}_{larger_model}.csv")])
            curr_duo_name = generate_ensemble_name(base_name, [smaller_model, larger_model])
            save_ensemble_predictions(pred_prefix, testtype, curr_duo_name, curr_duo_pred)
            if not (curr_duo_name in predictor_categories[base_name]):
                predictor_categories[base_name].append(curr_duo_name)
    return predictor_categories

def generate_weighted_duos(target_prefix, val_set, pred_prefix, base_name, testtype, smaller_model_ls, larger_model_ls, predictor_categories):
    label = pd.read_csv(f"{target_prefix}{val_set}.csv")
    for smaller_model in smaller_model_ls:
        for larger_model in larger_model_ls:
            pred_smaller_val = pd.read_csv(f"{pred_prefix}{val_set}_{smaller_model}.csv")
            pred_larger_val = pd.read_csv(f"{pred_prefix}{val_set}_{larger_model}.csv")
            temp_f1_df = {"p": [],"f1": []}
            for p in np.arange(0,1.01,0.01):
                weighted_pred = (p*pred_smaller_val + (1-p)*pred_larger_val).idxmax(axis=1).str.extract('(\d+)').astype(int)
                curr_f1 = f1_score(label["target"], weighted_pred, average='macro')
                temp_f1_df["p"].append(p)
                temp_f1_df["f1"].append(curr_f1)
            f1_df = pd.DataFrame(temp_f1_df)
            opt_f1_index = f1_df["f1"].argmax()
            opt_p = f1_df["p"][opt_f1_index]
            pred_smaller = pd.read_csv(f"{pred_prefix}{testtype}_{smaller_model}.csv")
            pred_larger = pd.read_csv(f"{pred_prefix}{testtype}_{larger_model}.csv")
            curr_duo_pred = opt_p*pred_smaller + (1-opt_p)*pred_larger
            curr_duo_name = generate_ensemble_name(base_name, [smaller_model, larger_model])+f"_{opt_p}"
            save_ensemble_predictions(pred_prefix, testtype, curr_duo_name, curr_duo_pred)
            if not (curr_duo_name in predictor_categories[base_name]):
                predictor_categories[base_name].append(curr_duo_name)
    return predictor_categories