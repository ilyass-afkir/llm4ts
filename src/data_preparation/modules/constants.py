"""Constants for electric truck data preparation.
 
This module defines column names, label encoders, signal mappings,
weather labels, highway labels, highway speed ranges, CRS codes, and UEA dataset
settings used throughout the data preparation pipeline.

Example:
    >>> from constants import AIR_TEMPERATURE_LABEL_ENCODER
    >>> AIR_TEMPERATURE_LABEL_ENCODER["cold"]
    2

Attributes:
    WEATHER_COLUMNS (list[str]): Columns related to weather labeling.
    HIGHWAY_COLUMNS (list[str]): Columns related to highway labeling.
    AIR_TEMPERATURE_LABEL_ENCODER (dict[str, int]): Mapping of air temperature labels to integers.
    WEATHER_LABEL_ENCODER (dict[str, int]): Mapping of weather condition labels to integers.
    HIGHWAY_LABEL_ENCODER (dict[str, int]): Mapping of highway labels to integers.
    HIGHWAY_OLD_TO_NEW (dict[int, int]): Old-to-new highway label integer mapping.
    HIGHWAY_NEW_TO_OLD (dict[int, int]): New-to-old highway label integer mapping.
    WEATHER_OLD_TO_NEW (dict[int, int]): Old-to-new weather label integer mapping.
    WEATHER_NEW_TO_OLD (dict[int, int]): New-to-old weather label integer mapping.
    LABEL_ENCODERS (dict[str, dict[str, int]]): Label encoders for all relevant columns.
    LABEL_DECODERS (dict[str, dict[int, str]]): Label decoders for all relevant columns.
    MERGE_SIGNALS (list[str]): Signal column names used for merging datasets.
    SIGNAL_MAPPING (dict[str, int]): Mapping of all signal names to column indices.
    TRAIN_SIGNAL_MAPPING (dict[str, int]): Subset of signals used for model training.
    WEATHER_LABELING_SIGNALS (list[str]): Signal columns used for weather labeling.
    AIR_TEMPERATURE_LABELS (list[str]): Ordered list of air temperature label strings.
    WEATHER_LABELS (list[str]): Ordered list of weather condition label strings.
    METEOSTAT_CODES (dict[int, str]): 
        Mapping of Meteostat integer codes to descriptions. 
        Reference: https://dev.meteostat.net/formats.
    METEOSTAT_TO_WEATHER (dict[int, str]): Mapping of Meteostat codes to weather labels.
    HIGHWAY_LABELING_SIGNALS (list[str]): Signal columns used for highway labeling.
    HIGHWAY_LABELS (list[str]): 
        Ordered list of OSM highway label strings.
        Reference: https://wiki.openstreetmap.org/wiki/Key:highway.
    TRUCK_SPEED_RANGES (dict[str, tuple[int, int]]): 
        Speed ranges (min, max) per highway type.
        Reference: https://dhl-freight-connections.com/en/business/truck-speed-limits-europe/.
    HIGHWAY_PRIORITY (dict[str, int]): Priority ranking of OSM highway types.
    COUNTRY_CODE_TO_OSM_MAP (dict[str, str]): Mapping of ISO country codes to OSM PBF filenames.
    CRS_CODES (dict[str, str]): 
        EPSG CRS codes per country and multi-country key. 
        Reference: https://epsg.io/.
    UEA (dict[str, dict[str, int]]): 
        Per-dataset configuration for UEA archive datasets.
        Reference: https://arxiv.org/abs/1811.00075.
    """

WEATHER_COLUMNS = ["air_temperature_label", "surface_condition_label", "weather_label"]

HIGHWAY_COLUMNS = ["highway_label"]

AIR_TEMPERATURE_LABEL_ENCODER = {
    "extreme_cold": 0, 
    "freezing_cold": 1,
    "cold": 2,          
    "moderate": 3,      
    "warm": 4,          
    "hot": 5  
}

WEATHER_LABEL_ENCODER = {
    "mild": 0,
    "fog": 1,
    "hail": 2,
    "heavy_rain": 3,
    "heavy_snowfall": 4,
    "rain": 5,
    "sleet": 6,
    "heavy_sleet": 7,
    "snowfall": 8,
    "storm": 9
}

HIGHWAY_LABEL_ENCODER = {
    "living_street": 0,
    "motorway": 1,
    "motorway_link": 2,
    "primary": 3,
    "primary_link": 4,
    "residential": 5,
    "secondary": 6,
    "secondary_link": 7,
    "service": 8,
    "tertiary": 9,
    "tertiary_link": 10,
    "track": 11,
    "trunk": 12,
    "trunk_link": 13,
    "unclassified": 14,
    "mini_roundabout": 15
}

HIGHWAY_OLD_TO_NEW = {
    1: 0,   # motorway: 1 → 0
    3: 1,   # primary: 3 → 1
    5: 2,   # residential: 5 → 2
    6: 3,   # secondary: 6 → 3
    9: 4,   # tertiary: 9 → 4
    12: 5,  # trunk: 12 → 5
}

HIGHWAY_NEW_TO_OLD = {v: k for k, v in HIGHWAY_OLD_TO_NEW.items()}

WEATHER_OLD_TO_NEW = {
    0:0,
    1:1,
    3:2,
    4:3,
    5:4,
    6:5,
    7:6,
    8:7,
    9:8
}

WEATHER_NEW_TO_OLD = {v: k for k, v in WEATHER_OLD_TO_NEW.items()}

LABEL_ENCODERS = {
    "air_temperature_label": AIR_TEMPERATURE_LABEL_ENCODER,
    "weather_label": WEATHER_LABEL_ENCODER,
    "highway_label": HIGHWAY_LABEL_ENCODER,
}

LABEL_DECODERS = {
    key: {v: k for k, v in enc.items()}
    for key, enc in LABEL_ENCODERS.items()
}

MERGE_SIGNALS = [
    "signal_time",
    "hirestotalvehdist_cval_icuc",
    "latitude_cval_ippc",
    "longitude_cval_ippc"
]

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
    "weather_label": 47,
    "highway_label": 48,
    "air_temperature_label": 49,
    "surface_condition_label": 50
}

TRAIN_SIGNAL_MAPPING = {
    "accelpdlposn_cval": 0,
    "actdrvtrnpwrprc_cval": 1,
    "actualdcvoltage_pti1": 2,
    "actualspeed_pti1": 3,
    "actualtorque_pti1": 4,
    "airtempinsd_cval_hvac": 5,
    "airtempinsd_rq": 6,
    "airtempoutsd_cval_cpc": 7,
    "altitude_cval_ippc": 8,
    "brc_stat_brc1": 9,
    "brktempra_cval": 10,
    "bs_brk_cval": 11,
    "currpwr_contendrnbrkresist_cval": 12,
    "elcomp_pwrcons_cval": 13,
    "emot_pwr_cval": 14,
    "epto_pwr_cval": 15,
    "hirestotalvehdist_cval_icuc": 16,
    "hv_bat_dc_momvolt_cval_bms1": 17,
    "hv_bat_soc_cval_bms1": 18,
    "hv_batavcelltemp_cval_bms1": 19,
    "hv_batcurr_cval_bms1": 20,
    "hv_batisores_cval_e2e": 21,
    "hv_batmaxchrgpwrlim_cval_1": 22,
    "hv_batmaxdischrgpwrlim_cval_1": 23,
    "hv_batmomavldischrgen_cval_1": 24,
    "hv_batpwr_cval_bms1": 25,
    "hv_curr_cval_dcl1": 26,
    "hv_dclink_volt_cval_dcl1": 27,
    "hv_ptc_cabin1_pwr_cval": 28,
    "hv_pwr_cval_dcl1": 29,
    "latitude_cval_ippc": 30,
    "longitude_cval_ippc": 31,
    "lv_convpwr_cval_dcl1": 32,
    "maxrecuppwrprc_cval": 33,
    "maxtracpwrpct_cval": 34,
    "motortemperature_pti1": 35,
    "powerstagetemperature_pti1": 36,
    "rmsmotorcurrent_pti1": 37,
    "roadgrad_cval_pt": 38,
    "selgr_rq_pt": 39,
    "txoiltemp_cval_tcm": 40,
    "vehspd_cval_cpc": 41,
    "vehweight_cval_pt": 42
}

WEATHER_LABELING_SIGNALS = [
    "signal_time",
    "airtempoutsd_cval_cpc",
    "altitude_cval_ippc",
    "hirestotalvehdist_cval_icuc",
    "latitude_cval_ippc",
    "longitude_cval_ippc",
]

AIR_TEMPERATURE_LABELS = [
    "extreme_cold",  
    "freezing_cold", 
    "cold",          
    "moderate",      
    "warm",          
    "hot"  
]

WEATHER_LABELS = [
    "mild",
    "fog",
    "hail",
    "heavy_rain",
    "heavy_snowfall",
    "rain",
    "sleet",
    "heavy_sleet",
    "snowfall",
    "storm"
]

METEOSTAT_CODES = {
    1: "Clear",
    2: "Fair",
    3: "Cloudy",
    4: "Overcast",
    5: "Fog",
    6: "Freezing Fog",
    7: "Light Rain",
    8: "Rain",
    9: "Heavy Rain",
    10: "Freezing Rain",
    11: "Heavy Freezing Rain",
    12: "Sleet",
    13: "Heavy Sleet",
    14: "Light Snowfall",
    15: "Snowfall",
    16: "Heavy Snowfall",
    17: "Rain Shower",
    18: "Heavy Rain Shower",
    19: "Sleet Shower",
    20: "Heavy Sleet Shower",
    21: "Snow Shower",
    22: "Heavy Snow Shower",
    23: "Lightning",
    24: "Hail",
    25: "Thunderstorm",
    26: "Heavy Thunderstorm",
    27: "Storm"
}

METEOSTAT_TO_WEATHER = {
    1: "mild",               # Clear
    2: "mild",               # Fair
    3: "mild",               # Cloudy
    4: "mild",               # Overcast
    5: "fog",                # Fog
    6: "fog",                # Freezing Fog
    7: "rain",               # Light Rain
    8: "rain",               # Rain
    9: "heavy_rain",         # Heavy Rain
    10: "rain",              # Freezing Rain
    11: "heavy_rain",        # Heavy Freezing Rain
    12: "sleet",             # Sleet
    13: "heavy_sleet",       # Heavy Sleet
    14: "snowfall",          # Light Snowfall
    15: "snowfall",          # Snowfall
    16: "heavy_snowfall",    # Heavy Snowfall
    17: "rain",              # Rain Shower
    18: "heavy_rain",        # Heavy Rain Shower
    19: "sleet",             # Sleet Shower
    20: "heavy_sleet",       # Heavy Sleet Shower
    21: "snowfall",          # Snow Shower
    22: "heavy_snowfall",    # Heavy Snow Shower
    24: "hail",              # Hail
    25: "storm",             # Thunderstorm
    26: "storm",             # Heavy Thunderstorm
    27: "storm"              # Storm
}

HIGHWAY_LABELING_SIGNALS = [
    "signal_time",
    "hirestotalvehdist_cval_icuc",
    "latitude_cval_ippc",
    "longitude_cval_ippc",
    "vehspd_cval_cpc"
]

HIGHWAY_LABELS = [
    "living_street",
    "motorway",
    "motorway_link",
    "primary",
    "primary_link",
    "residential",
    "secondary",
    "secondary_link",
    "service",
    "tertiary",
    "tertiary_link",
    "track",
    "trunk",
    "trunk_link",
    "unclassified",
    "mini_roundabout"
]

TRUCK_SPEED_RANGES = {
    "motorway": (70, 90),
    "motorway_link": (40, 60),
    "trunk": (50, 70),
    "trunk_link": (40, 60),
    "primary": (50, 70),
    "primary_link": (40, 60),
    "secondary": (40, 60),
    "secondary_link": (30, 50),
    "tertiary": (30, 50),
    "tertiary_link": (20, 40),
    "unclassified": (20, 50),
    "residential": (15, 30),
    "living_street": (7, 20),
    "service": (10, 20),
    "track": (10, 15),
}

HIGHWAY_PRIORITY = {
    "motorway": 100,
    "trunk": 90,
    "motorway_link": 85,
    "trunk_link": 80,
    "primary": 70,
    "primary_link": 65,
    "secondary": 50,
    "secondary_link": 45,
    "tertiary": 30,
    "tertiary_link": 25,
    "unclassified": 20,
    "residential": 10,
    "service": 5,
    "living_street": 3,
    "track": 1,
}

COUNTRY_CODE_TO_OSM_MAP = {
    "AT": "austria-251013.osm.pbf",
    "BE": "belgium-251013.osm.pbf",
    "BA": "bosnia-herzegovina-251013.osm.pbf",
    "BG": "bulgaria-251013.osm.pbf",
    "DK": "denmark-251025.osm.pbf",
    "HK": "croatia-251013.osm.pbf",
    "HR": "croatia-251013.osm.pbf",
    "FI": "finland-251013.osm.pbf",
    "FR": "france-251013.osm.pbf",
    "DE": "germany-latest.osm.pbf",
    "GR": "greece-251013.osm.pbf",
    "IT": "italy-251013.osm.pbf",
    "LU": "luxembourg-251013.osm.pbf",
    "NL": "netherlands-251013.osm.pbf",
    "NO": "norway-251013.osm.pbf",
    "ES": "spain-251013.osm.pbf",
    "SE": "sweden-251013.osm.pbf",
    "CH": "switzerland-251013.osm.pbf",
    "SI": "slovenia-251013.osm.pbf",
    "RS": "serbia-251013.osm.pbf",
    "TR": "turkey-251013.osm.pbf",
    "GB": "united-kingdom-251013.osm.pbf",
    "EU": "europe-latest.osm.pbf"
}

CRS_CODES = {
    "AT": "EPSG:3416",   
    "BG": "EPSG:7801",    
    "DE": "EPSG:25832",  
    "DK": "EPSG:25832",   
    "ES": "EPSG:25830",  
    "FI": "EPSG:3067",   
    "FR": "EPSG:2154",   
    "IT": "EPSG:6875",  
    "NL": "EPSG:28992",  
    "RS": "EPSG:8682",  
    "SE": "EPSG:3006",   
    "SI": "EPSG:3794",   
    "TR": "EPSG:5253",   
    "EU": "EPSG:3035"    
}

UEA = {
    "ArticularyWordRecognition": {
        "num_channels": 9, "num_classes": 25, "sequence_length": 144,
        "patch_length": 20, "patch_stride": 10, "batch_size": 64
    },
    "AtrialFibrillation": {
        "num_channels": 2, "num_classes": 3, "sequence_length": 640,
        "patch_length": 50, "patch_stride": 25, "batch_size": 5
    },
    "BasicMotions": {
        "train_size": 40, "test_size": 40,
        "num_channels": 6, "sequence_length": 100, "num_classes": 4,
        "patch_length": 16, "patch_stride": 8, "batch_size": 10
    },
    "CharacterTrajectories": {
        "num_channels": 3, "sequence_length": 182, "num_classes": 20,
        "patch_length": 128, "patch_stride": 64, "batch_size": 256
    },
    "Cricket": {
        "num_channels": 6, "sequence_length": 1197, "num_classes": 12,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "DuckDuckGeese": {
        "train_size": 60, "test_size": 40,
        "num_channels": 1345, "sequence_length": 270, "num_classes": 5,
        "patch_length": 32, "patch_stride": 16, "batch_size": 10
    },
    "EigenWorms": {
        "num_channels": 6, "num_classes": 5, "sequence_length": 17984,
        "patch_length": 256, "patch_stride": 128, "batch_size": 2
    },
    "Epilepsy": {
        "num_channels": 3, "sequence_length": 206, "num_classes": 4,
        "patch_length": 16, "patch_stride": 8, "batch_size": 64
    },
    "EthanolConcentration": {
        "num_channels": 3, "sequence_length": 1751, "num_classes": 4,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "ERing": {
        "num_channels": 4, "sequence_length": 65, "num_classes": 6,
        "patch_length": 8, "patch_stride": 4, "batch_size": 8
    },
    "FaceDetection": {
        "num_channels": 144, "sequence_length": 62, "num_classes": 2,
        "patch_length": 8, "patch_stride": 4, "batch_size": 256
    },
    "FingerMovements": {
        "num_channels": 28, "sequence_length": 50, "num_classes": 2,
        "patch_length": 16, "patch_stride": 8, "batch_size": 64
    },
    "HandMovementDirection": {
        "num_channels": 10, "sequence_length": 400, "num_classes": 4,
        "patch_length": 64, "patch_stride": 32, "batch_size": 64
    },
    "Handwriting": {
        "num_channels": 3, "sequence_length": 152, "num_classes": 26,
        "patch_length": 32, "patch_stride": 16, "batch_size": 32
    },
    "Heartbeat": {
        "num_channels": 61, "sequence_length": 405, "num_classes": 2,
        "patch_length": 64, "patch_stride": 32, "batch_size": 32
    },
    "JapaneseVowels": {
        "num_channels": 12, "sequence_length": 29, "num_classes": 9,
        "patch_length": 8, "patch_stride": 4, "batch_size": 64
    },
    "Libras": {
        "num_channels": 2, "sequence_length": 45, "num_classes": 15,
        "patch_length": 8, "patch_stride": 4, "batch_size": 64
    },
    "LSST": {
        "num_channels": 6, "sequence_length": 36, "num_classes": 14,
        "patch_length": 2, "patch_stride": 2, "batch_size": 256
    },
    "InsectWingbeat": {
        "num_channels": 200, "sequence_length": 78, "num_classes": 10,
        "patch_length": 16, "patch_stride": 8, "batch_size": 128
    },
    "MotorImagery": {
        "num_channels": 64, "sequence_length": 3000, "num_classes": 2,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "NATOPS": {
        "num_channels": 24, "sequence_length": 51, "num_classes": 6,
        "patch_length": 16, "patch_stride": 8, "batch_size": 32
    },
    "PenDigits": {
        "num_channels": 2, "sequence_length": 8, "num_classes": 10,
        "patch_length": 2, "patch_stride": 1, "batch_size": 256
    },
    "PEMS-SF": {
        "num_channels": 963, "sequence_length": 144, "num_classes": 7,
        "patch_length": 16, "patch_stride": 8, "batch_size": 32
    },
    "Phoneme": {
        "num_channels": 11, "sequence_length": 217, "num_classes": 39,
        "patch_length": 16, "patch_stride": 8, "batch_size": 256
    },
    "RacketSports": {
        "num_channels": 6, "sequence_length": 30, "num_classes": 4,
        "patch_length": 16, "patch_stride": 8, "batch_size": 64
    },
    "SelfRegulationSCP1": {
        "num_channels": 6, "sequence_length": 896, "num_classes": 2,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "SelfRegulationSCP2": {
        "num_channels": 7, "sequence_length": 1152, "num_classes": 2,
        "patch_length": 128, "patch_stride": 64, "batch_size": 16
    },
    "SpokenArabicDigits": {
        "num_channels": 13, "sequence_length": 93, "num_classes": 10,
        "patch_length": 16, "patch_stride": 8, "batch_size": 256
    },
    "StandWalkJump": {
        "num_channels": 4, "sequence_length": 2500, "num_classes": 3,
        "patch_length": 128, "patch_stride": 64, "batch_size": 1
    },
    "UWaveGestureLibrary": {
        "num_channels": 3, "sequence_length": 315, "num_classes": 8,
        "patch_length": 32, "patch_stride": 16, "batch_size": 32
    }
}