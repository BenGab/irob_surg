from ToolSegment import TootlSegment

segment = TootlSegment('/pretrined_models/UNET12_BCE_G_C3.h5')
segment.apply_video('/datasets/miccai/videoframes/Segmentation_Robotic_Training/Training/Dataset2/Video.avi', '/datasets/overlayed.avi')