import os
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
from PIL import Image


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        # assert 0,"You should specify the folder of your dataset"
        train_dir = '/home/wangyabin/workspace/data/train'
        test_dir = '/home/wangyabin/workspace/data/val'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = '/home/wangyabin/workspace/datasets/imagenet100/train/'
        test_dir = '/home/wangyabin/workspace/datasets/imagenet100/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCore50(iData):

    use_path = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(8*50).tolist()

    def download_data(self):
        from utils.datautils.core50.core50data import CORE50
        datagen = CORE50(root='/home/wangyabin/workspace/datasets/core50/data/core50_128x128', scenario="ni")

        dataset_list = []
        for i, train_batch in enumerate(datagen):
            imglist, labellist = train_batch
            labellist += i*50
            imglist = imglist.astype(np.uint8)
            dataset_list.append([imglist, labellist])
        train_x = np.concatenate(np.array(dataset_list)[:, 0])
        train_y = np.concatenate(np.array(dataset_list)[:, 1])
        self.train_data = train_x
        self.train_targets = train_y

        test_x, test_y = datagen.get_test_set()
        test_x = test_x.astype(np.uint8)
        self.test_data = test_x
        self.test_targets = test_y
        # import pdb;pdb.set_trace()



class iDomainnetCIL(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    def download_data(self):
        rootdir = '/home/wangyabin/workspace/datasets/domainnet'

        train_txt = './utils/datautils/domainnet/train.txt'
        test_txt = './utils/datautils/domainnet/test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (value, key) = line.strip().split(' ')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels








class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    # first we need get the pre-processed txt files which containing the taining and text split of DualPrompt.
    # the origin data structure is tfds, but we extract its data to a txt file
    """ 
    we use following code to extract split
    def get_stats(split_name, filepath):
        stats = []
        ds = dataset_builder.as_dataset(split=split_name)
        label_list = []
        for batch in ds:
            label_list.append(int(batch["label"]))
        label_list = list(set(label_list))
        data_dict = {i:[] for i in label_list}
        for batch in ds:
            data_dict[int(batch["label"])].append(batch["file_name"].numpy())
        print(len(label_list))
        label_list.sort()
        with open(filepath, 'w') as f:
            for i in label_list:
                for line in data_dict[i]:
                    f.write(str(IR_LABEL_MAP[i])  + "\t" + line.decode("utf-8") +"\n")
        return data_dict
    train_stats = get_stats("test[:80%]", "train.txt")
    test_stats = get_stats("test[80%:]", "text.txt")
    """

    def download_data(self):
        rootdir = '/home/wangyabin/workspace/datasets/imagenet-r'

        train_txt = './utils/datautils/imagenet-r/train.txt'
        test_txt = './utils/datautils/imagenet-r/test.txt'

        train_images = []
        train_labels = []
        with open(train_txt, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split('\t')
                train_images.append(os.path.join(rootdir, value))
                train_labels.append(int(key))
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)

        test_images = []
        test_labels = []
        with open(test_txt, 'r') as dict_file:
            for line in dict_file:
                (key, value) = line.strip().split('\t')
                test_images.append(os.path.join(rootdir, value))
                test_labels.append(int(key))
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

        self.train_data = train_images
        self.train_targets = train_labels
        self.test_data = test_images
        self.test_targets = test_labels

class iCIFAR100_vit(iData):
    use_path = False
    train_trsf = [
        transforms.Resize(256),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(100).tolist()
    # class_order = [51, 48, 73, 93, 39, 67, 29, 49, 57, 33, 4, 32, 5, 75, 63, 7, 61, 36, 69, 62, 46, 30, 25, 47, 12, 11, 94, 18, 27, 88, 0, 99, 21, 87, 34, 24, 86, 35, 22, 42, 66, 64, 2, 97, 98, 96, 71, 14, 95, 37, 54, 31, 10, 20, 52, 79, 60, 72, 41, 91, 44, 15, 16, 83, 59, 6, 82, 45, 81, 13, 53, 28, 50, 17, 19, 85, 1, 77, 70, 58, 38, 43, 80, 26, 9, 55, 92, 3, 89, 40, 76, 74, 65, 90, 84, 23, 8, 78, 56, 68]

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class i5Datasets_vit(iData):
    use_path = False
    train_trsf = [
        transforms.Resize(224),
        # transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]


    class_order = np.arange(50).tolist()


    def download_data(self):
        img_size=64
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)

        trainlist = []
        testlist = []
        train_label_list = []
        test_label_list = []

        # cifar10
        cifar10_train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        cifar10_test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        for img, target in zip(cifar10_train_dataset.data, cifar10_train_dataset.targets):
            trainlist.append(np.array(Image.fromarray(img).resize((img_size, img_size))))
            train_label_list.append(target)
        for img, target in zip(cifar10_test_dataset.data, cifar10_test_dataset.targets):
            testlist.append(np.array(Image.fromarray(img).resize((img_size, img_size))))
            test_label_list.append(target)

        # MNIST
        minist_train_dataset = datasets.MNIST('./data', train=True, download=True)
        minist_test_dataset = datasets.MNIST('./data', train=False, download=True)
        for img, target in zip(minist_train_dataset.data.numpy(), minist_train_dataset.targets.numpy()):
            trainlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            train_label_list.append(target+10)
        for img, target in zip(minist_test_dataset.data.numpy(), minist_test_dataset.targets.numpy()):
            testlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            test_label_list.append(target+10)

        # notMNIST
        classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        tarin_dir = "./data/notMNIST_large"
        test_dir = "./data/notMNIST_small"
        for idx, cls in enumerate(classes):
            image_files = os.listdir(os.path.join(tarin_dir, cls))
            for img_path in image_files:
                try:
                    image = np.array(Image.open(os.path.join(tarin_dir, cls, img_path)).resize((img_size, img_size)).convert('RGB'))
                    trainlist.append(image)
                    train_label_list.append(idx+20)
                except:
                    print(os.path.join(tarin_dir, cls, img_path))
            image_files = os.listdir(os.path.join(test_dir, cls))
            for img_path in image_files:
                try:
                    image = np.array(Image.open(os.path.join(test_dir, cls, img_path)).resize((img_size, img_size)).convert('RGB'))
                    testlist.append(image)
                    test_label_list.append(idx+20)
                except:
                    print(os.path.join(test_dir, cls, img_path))


        # Fashion-MNIST
        fminist_train_dataset = datasets.FashionMNIST('./data', train=True, download=True)
        fminist_test_dataset = datasets.FashionMNIST('./data', train=False, download=True)
        for img, target in zip(fminist_train_dataset.data.numpy(), fminist_train_dataset.targets.numpy()):
            trainlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            train_label_list.append(target+30)
        for img, target in zip(fminist_test_dataset.data.numpy(), fminist_test_dataset.targets.numpy()):
            testlist.append(np.array(Image.fromarray(img).resize((img_size, img_size)).convert('RGB')))
            test_label_list.append(target+30)

        # SVHN
        svhn_train_dataset = datasets.SVHN('./data', split='train', download=True)
        svhn_test_dataset = datasets.SVHN('./data', split='test', download=True)
        for img, target in zip(svhn_train_dataset.data, svhn_train_dataset.labels):
            trainlist.append(np.array(Image.fromarray(img.transpose(1,2,0)).resize((img_size, img_size))))
            train_label_list.append(target+40)
        for img, target in zip(svhn_test_dataset.data, svhn_test_dataset.labels):
            testlist.append(np.array(Image.fromarray(img.transpose(1,2,0)).resize((img_size, img_size))))
            test_label_list.append(target+40)


        train_dataset.data = np.array(trainlist)
        train_dataset.targets = np.array(train_label_list)
        test_dataset.data = np.array(testlist)
        test_dataset.targets = np.array(test_label_list)

        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iGanFake(object):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_order = args["class_order"]
        self.class_order = class_order

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.args["task_name"]):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = os.listdir(root_) if self.args["multiclass"][id] else ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)

class iGanClass(object):
    use_path = True
    train_trsf = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    def __init__(self, args):
        self.args = args
        class_number = args["class_number"]
        self.class_order = [i for i in range(2*class_number)]
        self.task_name = [str(i) for i in range(class_number)]

    def download_data(self):

        train_dataset = []
        test_dataset = []
        for id, name in enumerate(self.task_name):
            root_ = os.path.join(self.args["data_path"], name, 'train')
            sub_classes = ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    train_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    train_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        for id, name in enumerate(self.task_name):
            root_ = os.path.join(self.args["data_path"], name, 'val')
            sub_classes = ['']
            for cls in sub_classes:
                for imgname in os.listdir(os.path.join(root_, cls, '0_real')):
                    test_dataset.append((os.path.join(root_, cls, '0_real', imgname), 0 + 2 * id))
                for imgname in os.listdir(os.path.join(root_, cls, '1_fake')):
                    test_dataset.append((os.path.join(root_, cls, '1_fake', imgname), 1 + 2 * id))

        self.train_data, self.train_targets = split_images_labels(train_dataset)
        self.test_data, self.test_targets = split_images_labels(test_dataset)