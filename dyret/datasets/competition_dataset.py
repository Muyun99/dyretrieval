import os.path
from .base_dataset import BaseDataset
from .builder import DATASETS
from torch.cuda.amp import autocast
from tools.torch_utils import *

@DATASETS.register_module()
class competition_dataset(BaseDataset):
    def load_annotations(self):
        self.data_csv_path = os.path.join(self.data_prefix, self.ann_file)
        self.meta_txt = os.path.join(self.data_prefix, 'classmap.txt')
        self.df = pd.read_csv(self.data_csv_path)

        if not self.test_mode:
            self.num_classes = len(self.df['label'].unique())

        self.imgs = []
        self.gt_labels = []


        for index in range(len(self.df)):
            img = mmcv.imread(self.df['filename'].iloc[index], channel_order='rbg')
            self.imgs.append(img)

            if not self.test_mode:
                label = np.array(self.df['label'].iloc[index])
                self.gt_labels.append(label)

        data_infos = []
        if not self.test_mode:
            for img, gt_label in zip(self.imgs, self.gt_labels):
                gt_label = np.array(gt_label, dtype=np.int64)
                info = {'img': img, 'gt_label': gt_label}
                data_infos.append(info)
        else:
            for img in self.imgs:
                info = {'img': img}
                data_infos.append(info)

        return data_infos

    def evaluate(self, cfg, model, valid_dataloader):
        model.eval()
        valid_features = []
        with torch.no_grad():
            for step, data in enumerate(valid_dataloader):
                images, labels = data['img'], data['gt_label']
                images, labels = images.cuda(), labels.cuda()

                if cfg.fp16:
                    with autocast():
                        features = model(images)
                else:
                    features = model(images)
                valid_features.append(features.data.cpu().numpy())

        valid_features = np.vstack(valid_features)
        val_distance = []
        for feature in valid_features:
            dis = np.dot(feature, valid_features.T)
            val_distance.append(dis)

        best_threahold, best_f1 = 0, 0
        for threahold in np.linspace(0.5, 0.99, 20):
            valid_submit = []
            for dis in val_distance[:]:
                pred = np.where(dis > threahold)[0]
                if len(pred) == 1:
                    ids = dis.argsort()[::-1]
                    pred = [x for x in ids[dis[ids] > 0.1]][:2]

                valid_submit.append(pred)

            val_f1s = []
            for x, pred in zip(self.gt_labels, valid_submit):
                label = np.where(self.gt_labels == x)[0]
                val_f1 = len(set(pred) & set(label)) / len(set(pred) | set(label))
                val_f1s.append(val_f1)

            if best_f1 < np.mean(val_f1s):
                best_f1 = np.mean(val_f1s)
                best_threahold = threahold
        model.train()
        return best_threahold, best_f1

    # def _inference(engine, cfg, model, valid_dataloader):
    #     model.eval()
    #     valid_features = []
    #     with torch.no_grad():
    #         for step, data in enumerate(valid_dataloader):
    #             images, labels = data['img'], data['gt_label']
    #             images, labels = images.cuda(), labels.cuda()
    #
    #             if cfg.fp16:
    #                 with autocast():
    #                     features = model(images)
    #             else:
    #                 features = model(images)
    #             valid_features.append(features.data.cpu().numpy())