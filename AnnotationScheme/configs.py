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
TOOLS_CLASSES = ["SL", "adjustable spanner", "allen", "drill", "hammer", "plier", "ratchet", "screwdriver", "tapemeasure", "wrench", "hands", "saw", "elctric screwdriver", "blade", "scissor", "None"]
DEVICE_CLASSES = ["PC", "Laptop", "Washer", "Dryer", "Refrigerator", "microwave", "scaffold board", "vibratory plate", "monitor", "TV", "Fan", "Air intaker", "Electronics", "Printer", "None"]
OBJECT_CLASSES = {
    'TOOL': {id_: name for id_, name in enumerate(TOOLS_CLASSES)},
    'HAND': {},
    'DEVICE': {id_: name for id_, name in enumerate(DEVICE_CLASSES)},
}

# ---------------------------------------- UI Setup
COLORS = {
    "menu_class"  : (255, 240, 191),   
    "tool_point"  : (255, 255, 0),   # cyan
    "hand_point"  : (255,   0, 255), # magenta
    "exclude"     : (0,   0, 255),   # red
    "cross_line"  : (255, 0,   0),
    "panel_color" : (40,40,40),
    "classes_background": (10, 10,  10),  # dark gray
}
LINE_GAP = 1.5
x_offset = 20
winnsize = (1600, 900)  # width, height
