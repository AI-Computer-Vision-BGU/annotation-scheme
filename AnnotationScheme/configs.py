# ---------------------------------------- External Drive Setup
SSD_NAME = 'AR-FOR WIND'
DATASET_PATH_SSD = f'/Volumes/{SSD_NAME}'
DATABASE_PATH_SDD = f'{DATASET_PATH_SSD}/maintenance_dataset.db'
DB_TABLE_NAME = 'MaintenanceActions_metadata'

# ---------------------------------------- weights configs
config_weights_mapping = {
    't': {'configs': 'configs/sam2.1/sam2.1_hiera_t.yaml', 'weights': 'segmentanything/checkpoints/sam2.1_hiera_tiny.pt'},  
    'b': {'configs': 'configs/sam2.1/sam2.1_hiera_b+.yaml', 'weights': 'segmentanything/checkpoints/sam2.1_hiera_base_plus.pt'},
    's': {'configs': 'configs/sam2.1/sam2.1_hiera_s.yaml', 'weights': 'segmentanything/checkpoints/sam2.1_hiera_small.pt'},
    'l': {'configs': 'configs/sam2.1/sam2.1_hiera_l.yaml', 'weights': 'segmentanything/checkpoints/sam2.1_hiera_large.pt'},
}



# ---------------------------------------- CHANGE based on the object one need to segment
OBJECT_TO_ANNOTATE = {'TOOL': 10, 'HAND': 20, 'DEVICE': 30}  # object name and unique id
OBJECT_COLORS = {
        'TOOL': (255, 255, 0),  # tool - blue
        'HAND': (255, 0, 255),  # hand - pink
        'DEVICE': (0, 255, 128)  # device - green
}
OBJECT_TO_ANNOTATE_REVERSED = {v: k for k, v in OBJECT_TO_ANNOTATE.items()}  # reverse mapping
OBJECT_WITH_BB = ['TOOL', 'DEVICE']                          # objects that need bounding box
TOOL_CATEGORIES = ['Hammer', 'Cut', 'Screw', 'Piping', 'Measure']
NONE_TOOL_CATEGORIES = ['Attach', 'Click', 'Cover', 'OpenClose', 'Plug']
category_id_to_name = {
                        0: "SL",
                        1: "adjustable spanner",
                        2: "allen",
                        3: "drill",
                        4: "hammer",
                        5: "plier",
                        6: "ratchet",
                        7: "screwdriver",
                        8: "tapemeasure",
                        9: "wrench",
                        10: "hands",
                        11: "saw",
                        12: "elctric screwdriver",
                        13: "None",
    }

device_to_id_mapping = {
    0: "PC",
    1: "Laptop",
    2: "Tablet",
    3: "Washer",
    4: "Dryer",
    5: "Refrigerator",
}

OBJECT_CLASSES = {
    'TOOL': category_id_to_name,
    'HAND': {},
    'DEVICE': device_to_id_mapping,
}

# ---------------------------------------- UI Setup
COLORS = {
    "menu_class"  : (0, 255, 128),   
    "tool_point"  : (255, 255, 0),   # cyan
    "hand_point"  : (255,   0, 255), # magenta
    "exclude"     : (0,   0, 255),   # red
    "cross_line"  : (255, 0,   0),
    "menu_bar"    : ()
}

x_offset = 20
winnsize = (1280, 720)  # width, height
