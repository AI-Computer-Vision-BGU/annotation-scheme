# ---------------------------------------- External Drive Setup
SSD_NAME = 'AR-FOR WIND'
DATASET_PATH_SSD = f'/Volumes/{SSD_NAME}'
DATABASE_PATH_SDD = f'{DATASET_PATH_SSD}/maintenance_dataset.db'
DB_TABLE_NAME = 'MaintenanceActions_metadata'

# ---------------------------------------- weights path
sam2_checkpoint_video = 'segmentanything/checkpoints/sam2.1_hiera_tiny.pt'
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
sam_checkpoint_image = 'segmentanything/checkpoints/sam_vit_h_4b8939.pth'


# ---------------------------------------- Change based on the object one need to segment
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
