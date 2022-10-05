import os

self_path = os.path.split(__file__)[0]


def path_parse(target_path):
    return os.path.join(self_path, target_path)


# data set path

ETT_PATH_DIR = path_parse("../dataset/ETT-small")
ECL_PATH = path_parse("../dataset/ECL.csv")
WTH_PATH = path_parse("../dataset/WTH.csv")
KDD_PATH = path_parse("../dataset/wtbdata_245days.csv")

# result path

ECL_PRED_PATH = path_parse(
    "../submodule/informer2020/results/informer_ECL_ftM_sl96_ll48_pl24_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_test_0")
