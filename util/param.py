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
ECL_PRED = {
    "S": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ECL_ftS_sl168_ll168_pl168_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ECL_168_168_Autoformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "reformer": path_parse(
            "../submodule/autoformer/results/ECL_336_168_Reformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "prophet": None
    },
    "M": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ECL_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ECL_96_192_Autoformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0")
    }

}

WTH_PRED = {
    "S": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_WTH_ftS_sl168_ll168_pl168_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/WTH_168_168_Autoformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "reformer": path_parse(
            "../submodule/autoformer/results/WTH_336_168_Reformer_custom_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "prophet": None
    },
    "M": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_WTH_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/WTH_96_192_Autoformer_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0")
    }

}

ETTH1_PRED = {
    "S": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ETTh1_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ETTh1_168_168_Autoformer_ETTh1_ftS_sl168_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "reformer": path_parse(
            "../submodule/autoformer/results/ETTh1_336_168_Reformer_ETTh1_ftS_sl336_ll168_pl168_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0"),
        "prophet": None
    },
    "M": {
        "informer": path_parse(
            "../submodule/informer2020/results/informer_ETTh1_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_Exp_0"),
        "autoformer": path_parse(
            "../submodule/autoformer/results/ETTh1_96_192_Autoformer_ETTh1_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0")
    }

}
