dataset_config = {
    'train_im_sets': ['data/VOC2007', 'data/VOC2012'],
    'test_im_sets': ['data/VOC2007-test'],
    'num_classes': 21,
    'bg_class_idx': 0,
    'im_size': 640,
    'VOC2007_TRAINVAL_URL': "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    'VOC2007_TEST_URL': "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    'VOC2012_URL': "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
    'BASE_DIR': 'data',
    'num_workers': 10,
}

model_config = {
    'im_channels': 3,
    'backbone_dim': 2048, # Resnet 50: 2048, Resnet34: 512,,
    'backbone_model': 101,
    'model_dim': 256,
    'num_queries': 25,
    'freeze_backbone': True,
    'encoder_layers': 4,
    'encoder_heads': 8,
    'decoder_layers': 4,
    'decoder_heads': 8,
    'dropout_prob': 0.1,
    'mlp_inner_dim': 2048,
    'cls_cost_weight': 1.0,
    'l1_cost_weight': 5.0,
    'giou_cost_weight': 2.0,
    'bg_class_weight': 0.1,
    'nms_threshold': 0.5,
    
}

train_config = {
    'task_name': 'voc',
    'eval_score_threshold': 0.0,
    'infer_score_threshold': 0.5,
    'use_nms_eval': False,
    'use_nms_infer': True,
    'num_epochs': 30,
    'batch_size': 4*15,# 4
    'lr_steps': [20], #200
    'lr': 0.0001, #0.0001,
    'log_steps': 40,#500,
    'ckpt_name': 'detr_voc2007.pth',
}
