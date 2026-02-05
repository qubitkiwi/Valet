IMG_WIDTH = 224
IMG_HEIGHT = 224

NUM_CLASSES = 2
CLASS_COL_NAME = 'sign_class'

# 카메라별 Crop 설정
CROP_SETTINGS = {
    'front_cam': (0, 480, 0, 640),      
    'rear_cam':  (0, 480, 0, 640),      
    'left_cam':  (0, 300, 100, 640),    
    'right_cam': (0, 300, 0, 540)      
}