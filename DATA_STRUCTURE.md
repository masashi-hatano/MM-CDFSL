# Data Structure
The data should be organized as follow.

```
Data
|
└── ego4d_data/v1
|   |
|   └── image_frame
|   │   └── clip_uid1
|   │       └── 00001.jpg
|   |       └── ...
|   |
|   └── optical_flow
|   │   └── clip_uid1/npy
|   │       └── 000001.npy
|   |       └── ...
|   |    
|   └── hand_pose/heatmap
|       └── clip_uid1/npy
|           └── 000001.npy
|           └── ...
|
└── EPIC-KITCHENS-55/frames
|   |
|   └── rgb/train
|   │   └── participant_number
|   │       └── video_id
|   |           └── frame_0000000001.jpg
|   |           └── ...
|   |
|   └── flow/train
|   │   └── participant_number
|   │       └── video_id
|   |           └── frame_0000000001.npy
|   |           └── ...
|   |    
|   └── pose/train
|       └── participant_number
|           └── video_id
|               └── frame_0000000001.npy
|               └── ...
|
└── MECCANO
|   |
|   └── RGB_frames
|   │   └── video_id
|   |       └── 00001.jpg
|   |       └── ...
|   |
|   └── flow_frames
|   │   └── video_id
|   |       └── 00001.npy
|   |       └── ...
|   |    
|   └── hand-pose/heatmap
|       └── video_id
|           └── 00001.npy
|           └── ...
|
└── WEAR
    |
    └── RGB_frames
    │   └── video_id
    |       └── 000001.jpg
    |       └── ...
    |
    └── flow_frames
    │   └── video_id
    |       └── 000001.npy
    |       └── ...
    |    
    └── hand-pose/heatmap
        └── video_id
            └── 000001.npy
            └── ...
```