"""Constants for electric truck data understanding.

This module defines vehicle identifiers, vehicle groups, vehicle-to-group
mappings, and signal column mappings used throughout the data understanding
pipeline.

Attributes:
    VEHICLE_IDS (list[str]): Supported vehicle identifiers.
    VEHICLE_GROUPS (dict[int, list[str]]): Mapping from group index to member vehicle IDs.
    VEHICLE_ID_TO_GROUP (dict[str, int]): Mapping from vehicle ID to its group index.
    SIGNAL_MAPPING (dict[str, int]): Expected signal names and their column indices used for schema validation and ordering.
"""

VEHICLE_IDS = [
    'v1', 
    'v2', 
    'v4', 
    'v10', 
    'v11', 
    'v12', 
    'v13', 
    'v14',
    'v15', 
    'v16', 
    'v17', 
    'v18', 
    'v19', 
    'v101', 
    'v102'
]

VEHICLE_GROUPS = {
    1: "v1_v2_v12", 
    2: "v4_v11_v13_v14_v16_v17", 
    3: "v10",
    4: "v15_v19", 
    5: "v18", 
    6: "v101_v102"
}

VEHICLE_ID_TO_GROUP = {
    'v1': 1, 
    'v2': 1, 
    'v4': 2, 
    'v10': 3, 
    'v11': 2, 
    'v12': 1, 
    'v13': 2, 
    'v14': 2,
    'v15': 4, 
    'v16': 2, 
    'v17': 2, 
    'v18': 5, 
    'v19': 4, 
    'v101': 6, 
    'v102': 6
}

SIGNAL_MAPPING = {
    "v_id": 0,
    "v_group": 1,
    "signal_time": 2,
    "signal_ts": 3,
    "accelpdlposn_cval": 4,
    "actdrvtrnpwrprc_cval": 5,
    "actualdcvoltage_pti1": 6,
    "actualspeed_pti1": 7,
    "actualtorque_pti1": 8,
    "airtempinsd_cval_hvac": 9,
    "airtempinsd_rq": 10,
    "airtempoutsd_cval_cpc": 11,
    "altitude_cval_ippc": 12,
    "brc_stat_brc1": 13,
    "brktempra_cval": 14,
    "bs_brk_cval": 15,
    "currpwr_contendrnbrkresist_cval": 16,
    "elcomp_pwrcons_cval": 17,
    "emot_pwr_cval": 18,
    "epto_pwr_cval": 19,
    "hirestotalvehdist_cval_icuc": 20,
    "hv_bat_dc_momvolt_cval_bms1": 21,
    "hv_bat_soc_cval_bms1": 22,
    "hv_batavcelltemp_cval_bms1": 23,
    "hv_batcurr_cval_bms1": 24,
    "hv_batisores_cval_e2e": 25,
    "hv_batmaxchrgpwrlim_cval_1": 26,
    "hv_batmaxdischrgpwrlim_cval_1": 27,
    "hv_batmomavldischrgen_cval_1": 28,
    "hv_batpwr_cval_bms1": 29,
    "hv_curr_cval_dcl1": 30,
    "hv_dclink_volt_cval_dcl1": 31,
    "hv_ptc_cabin1_pwr_cval": 32,
    "hv_pwr_cval_dcl1": 33,
    "latitude_cval_ippc": 34,
    "longitude_cval_ippc": 35,
    "lv_convpwr_cval_dcl1": 36,
    "maxrecuppwrprc_cval": 37,
    "maxtracpwrpct_cval": 38,
    "motortemperature_pti1": 39,
    "powerstagetemperature_pti1": 40,
    "rmsmotorcurrent_pti1": 41,
    "roadgrad_cval_pt": 42,
    "selgr_rq_pt": 43,
    "txoiltemp_cval_tcm": 44,
    "vehspd_cval_cpc": 45,
    "vehweight_cval_pt": 46,
}

