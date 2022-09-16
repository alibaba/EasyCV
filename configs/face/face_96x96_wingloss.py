# model settings
POINT_NUMBER = 106
MEAN_FACE = [
    0.05486667535113006, 0.24441904048908245, 0.05469932714062696,
    0.30396829196709935, 0.05520653400164321, 0.3643191463607746,
    0.05865501342257397, 0.42453849020500306, 0.0661603899137523,
    0.48531377442945767, 0.07807677169271177, 0.5452126843738523,
    0.09333319368757653, 0.6047840615432064, 0.11331425394034209,
    0.6631144309665994, 0.13897813867699352, 0.7172296230155276,
    0.17125811033538194, 0.767968859462583, 0.20831698519371536,
    0.8146603379935117, 0.24944621000897876, 0.857321261721953,
    0.2932993820558674, 0.8973900596678597, 0.33843820185594653,
    0.9350576242126986, 0.38647802623495553, 0.966902971122812,
    0.4411974776504609, 0.9878629960611088, 0.5000390697219397,
    0.9934886214875595, 0.5588590024515473, 0.9878510782414189,
    0.6135829360035883, 0.9668655595323074, 0.6616294188166414,
    0.9350065330378543, 0.7067734980023662, 0.8973410411573094,
    0.7506167730772516, 0.8572957679511382, 0.7917579157122047,
    0.8146281598803492, 0.8288026446367324, 0.7679019642224981,
    0.8610918526053805, 0.7171624168757985, 0.8867491048162915,
    0.6630344261248556, 0.9067293813428708, 0.6047095492618413,
    0.9219649147678989, 0.5451295187190602, 0.9338619041815587,
    0.4852292097262674, 0.9413455695142587, 0.424454780475834,
    0.9447753107545577, 0.3642347111991026, 0.9452649776939869,
    0.30388458223793025, 0.9450854849661369, 0.24432737691068557,
    0.1594802473020129, 0.17495177946520288, 0.2082918411850002,
    0.12758378330875153, 0.27675902873293057, 0.11712230823088154,
    0.34660582049732336, 0.12782553369032904, 0.4137234315527489,
    0.14788458441422778, 0.4123890243720449, 0.18814226684806626,
    0.3498927810760776, 0.17640650480816664, 0.28590212091591866,
    0.16895271174960227, 0.22193967489846017, 0.16985862149585013,
    0.5861805004572298, 0.147863456192582, 0.6532904167464643,
    0.12780412047734288, 0.723142364263288, 0.11709102395419578,
    0.7916076475508984, 0.12753867695205595, 0.8404440227263494,
    0.17488715120168932, 0.7779848023963316, 0.1698261195288917,
    0.7140264757991571, 0.1689377237959271, 0.650024882334848,
    0.17640581823811927, 0.5875270068157493, 0.18815421057605972,
    0.4999687027691624, 0.2770570778583906, 0.49996466107378934,
    0.35408433007759227, 0.49996725190415664, 0.43227025345368053,
    0.49997367716346774, 0.5099309118810921, 0.443147025685285,
    0.2837021691260901, 0.4079306716593004, 0.4729519900478952,
    0.3786223176615041, 0.5388017782630576, 0.4166237366074797,
    0.5822229552544941, 0.4556754522760756, 0.5887956328134262,
    0.49998730493119997, 0.5951855531982454, 0.5443300921009105,
    0.5887796732983633, 0.5833722476054509, 0.582200985012979,
    0.6213509190608012, 0.5387760772258134, 0.5920137550293199,
    0.4729325070035326, 0.5567854054587345, 0.28368589871138317,
    0.23395988420439123, 0.275313734012504, 0.27156519109550253,
    0.2558735678926061, 0.31487949633428597, 0.2523033259214858,
    0.356919009399118, 0.2627342680634766, 0.3866625969903256,
    0.2913618036573405, 0.3482919069920915, 0.3009936818974329,
    0.3064437008415846, 0.3037349617842158, 0.26724000706363993,
    0.2961896087804692, 0.3135744691699477, 0.27611103614975246,
    0.6132904312551143, 0.29135144033587107, 0.6430396927648264,
    0.2627079452269443, 0.6850713556136455, 0.2522730391144915,
    0.728377707003201, 0.25583118190779625, 0.7660035591791254,
    0.27526375689471777, 0.7327054300488236, 0.2961495286346863,
    0.6935171517115648, 0.3036951925380769, 0.6516533228539426,
    0.3009921014909089, 0.6863983789278025, 0.2760904908649394,
    0.35811903020866753, 0.7233174007629063, 0.4051199834269763,
    0.6931800846807724, 0.4629631471997891, 0.6718031951363689,
    0.5000016063148277, 0.6799150331999366, 0.5370506360177653,
    0.6717809139952097, 0.5948714927411151, 0.6931581144392573,
    0.6418878095835022, 0.7232890570786875, 0.6088129582142587,
    0.7713407215524752, 0.5601450388292929, 0.8052499757498277,
    0.5000181358125715, 0.8160749831906926, 0.4398905591799545,
    0.8052697696938342, 0.39120318265892984, 0.771375905028864,
    0.36888771299734613, 0.7241751210643214, 0.4331097084010058,
    0.7194543690519717, 0.5000188612450743, 0.7216823277180712,
    0.566895861884284, 0.7194302225129479, 0.631122598507516,
    0.7241462073974219, 0.5678462302796355, 0.7386355816766528,
    0.5000082906571756, 0.7479600838019628, 0.43217532542902076,
    0.7386538729390463, 0.31371761254774383, 0.2753328284323114,
    0.6862487843823917, 0.2752940437017121
]
IMAGE_SIZE = 96

loss_config = dict(
    num_points=POINT_NUMBER,
    left_eye_left_corner_index=66,
    right_eye_right_corner_index=79,
    points_weight=1.0,
    contour_weight=1.5,
    eyebrow_weight=1.5,
    eye_weight=1.7,
    nose_weight=1.3,
    lip_weight=1.7,
    omega=10,
    epsilon=2)

model = dict(
    type='FaceKeypoint',
    backbone=dict(
        type='FaceKeypointBackbone',
        in_channels=3,
        out_channels=48,
        residual_activation='relu',
        inverted_activation='half_v2',
        inverted_expand_ratio=2,
    ),
    keypoint_head=dict(
        type='FaceKeypointHead',
        in_channels=48,
        out_channels=POINT_NUMBER * 2,
        input_size=IMAGE_SIZE,
        inverted_expand_ratio=2,
        inverted_activation='half_v2',
        mean_face=MEAN_FACE,
        loss_keypoint=dict(type='WingLossWithPose', **loss_config),
    ),
    pose_head=dict(
        type='FacePoseHead',
        in_channels=48,
        out_channels=3,
        inverted_expand_ratio=2,
        inverted_activation='half_v2',
        loss_pose=dict(type='FacePoseLoss', pose_weight=0.01),
    ),
)

train_pipeline = [
    dict(type='FaceKeypointRandomAugmentation', input_size=IMAGE_SIZE),
    dict(type='FaceKeypointNorm', input_size=IMAGE_SIZE),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.4076, 0.458, 0.485],
        std=[1.0, 1.0, 1.0]),
    dict(
        type='Collect',
        keys=[
            'img', 'target_point', 'target_point_mask', 'target_pose',
            'target_pose_mask'
        ])
]

val_pipeline = [
    dict(type='FaceKeypointNorm', input_size=IMAGE_SIZE),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.4076, 0.458, 0.485],
        std=[1.0, 1.0, 1.0]),
    dict(
        type='Collect',
        keys=[
            'img', 'target_point', 'target_point_mask', 'target_pose',
            'target_pose_mask'
        ])
]
test_pipeline = val_pipeline

data_root = 'path/to/face_landmark_data/'

data_cfg = dict(
    data_root=data_root,
    input_size=IMAGE_SIZE,
)

data = dict(
    imgs_per_gpu=512,
    workers_per_gpu=2,
    train=dict(
        type='FaceKeypointDataset',
        data_source=dict(
            type='FaceKeypintSource',
            train=True,
            data_range=[0, 30000],  # [0,30000]  [0,478857]
            data_cfg=data_cfg,
        ),
        pipeline=train_pipeline),
    val=dict(
        type='FaceKeypointDataset',
        data_source=dict(
            type='FaceKeypintSource',
            train=False,
            data_range=[478857, 488857],
            # data_range=[478857, 478999], #[478857, 478999] [478857, 488857]
            data_cfg=data_cfg,
        ),
        pipeline=val_pipeline),
    test=dict(
        type='FaceKeypointDataset',
        data_source=dict(
            type='FaceKeypintSource',
            train=False,
            data_range=[478857, 488857],
            # data_range=[478857, 478999], #[478857, 478999] [478857, 488857]
            data_cfg=data_cfg,
        ),
        pipeline=test_pipeline),
)

# runtime setting
optimizer = dict(
    type='Adam',
    lr=0.005,
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.00001,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.001,
    warmup_by_epoch=True,
    by_epoch=True)

total_epochs = 1000
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=5, hooks=[
        dict(type='TextLoggerHook'),
    ])

predict = dict(type='FaceKeypointsPredictor')

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

evaluation = dict(interval=1, metric=['NME'], save_best='NME')

eval_config = dict(interval=1)
evaluator_args = dict(metric_names='ave_nme')
eval_pipelines = [
    dict(
        mode='test',
        data=dict(**data['val'], imgs_per_gpu=1),
        evaluators=[dict(type='FaceKeypointEvaluator', **evaluator_args)])
]
