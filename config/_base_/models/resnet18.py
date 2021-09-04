model = 'resnet18'
pretrained = True

# get_loss
loss1 = dict(type='CrossEntropyLoss')
# loss1 = dict(type='CrossEntropyLabelSmooth', epsilon=0.1)
loss2 = dict(type='TripletLoss', margin=0.3)

bnn_neck = False

