import os
import os.path as osp
import random


class CommonFolderScanner(object):
    """CommonFolderScanner

        Arguments:
            root_folder {string} -- which folder to scan

        Keyword Arguments:
            max_depth {int} -- the max_depth to scan (default: {1})
            classification_depth {int} -- classifcate by folder name in what depth (default: {1})
            file_type {list} -- default filter function used to filter file type (default: {['jpg']}) 
                                if file_filter is not none this para will be ignored
            file_filter{function} -- you can impl a filter function instead the default one,recieve a fullpath filename and return a bool (default: None)
            combine_same_dir_as_one_class {bool} -- whether combine the same folder of different father folder (default: {True})
    """

    def __init__(self, root_folder, max_depth=1, classification_depth=1, file_type=['jpg'], file_filter=None, combine_same_dir_as_one_class=True):

        self.root_folder = os.path.normpath(root_folder)
        self.max_depth = max_depth
        self.classification_depth = classification_depth
        self.file_type = file_type
        self.filter = file_filter
        self.combine = combine_same_dir_as_one_class

    def scan_folder(self):
        """Scan the root folder and find out needed files
           and labeled it with folders' names

        Returns:
            [dict] -- [{'classname':file list}]
        """
        assert self.max_depth >= self.classification_depth, 'max_depth should be larger than classification_depth'
        assert osp.exists(
            self.root_folder), '%s is not a directory' % self.root_folder
        g = os.walk(self.root_folder, topdown=True)
        res_for_imglist = []
        res_for_classification = {}
        for _, (path, _, files) in enumerate(g):
            new_path = path[len(self.root_folder):]
            depth = new_path.count(os.path.sep)
            if depth > self.max_depth:
                continue
            if self.classification_depth <= depth:
                if self.combine:
                    classname = new_path.split(os.path.sep)[
                        self.classification_depth]
                else:
                    classname = new_path.rsplit(
                        osp.sep, depth-self.classification_depth)[0]
                    classname = classname.replace(osp.sep, '_')
            else:
                continue
            for f in files:
                fullpath_f = osp.join(new_path, f)
                if self.filter_file(fullpath_f):
                    res_for_imglist.append(fullpath_f)
            if classname in res_for_classification:
                res_for_classification[classname] += res_for_imglist
            else:
                res_for_classification[classname] = res_for_imglist
            res_for_imglist = []
        return res_for_classification

    def create_trainvaltest(self, dst_path=None, rate=(3, 1, 1)):
        """Create train val test txt files using result returned by scan_folder

        Keyword Arguments:
            dst_path {string} -- where the txt files to save (default: {None})
            rate {tuple} -- train:val:test (default: {(3, 1, 1)})
        """
        if dst_path is None:
            save_root = '.'
        else:
            save_root = dst_path
        assert osp.exists(save_root), '%s does\'nt exist' % save_root
        res = self.scan_folder()
        val_rate = rate[1] / sum(rate)
        test_rate = rate[2] / sum(rate)
        for i, classname in enumerate(res):
            class_i_imgs = res[classname]
            random.shuffle(class_i_imgs)
            total = len(class_i_imgs)
            val_count = int(total*val_rate)
            test_count = int(total*test_rate)
            class_i_imgs_for_val = class_i_imgs[0:val_count]
            class_i_imgs_for_test = class_i_imgs[val_count:val_count+test_count]
            class_i_imgs_for_train = class_i_imgs[val_count+test_count:]
            label = i
            with open(osp.join(save_root, 'train.txt'), 'a') as f:
                f.writelines(['%s %d\n' % (x, label)
                              for x in class_i_imgs_for_train])
            with open(osp.join(save_root, 'val.txt'), 'a') as f:
                f.writelines(['%s %d\n' % (x, label)
                              for x in class_i_imgs_for_val])
            with open(osp.join(save_root, 'test.txt'), 'a') as f:
                f.writelines(['%s %d\n' % (x, label)
                              for x in class_i_imgs_for_test])
            with open(osp.join(save_root, 'classnames.txt'), 'a') as f:
                f.write('%d\t%-s\n' % (label, classname))
            with open(osp.join(save_root, 'trainvaltest_all.txt'), 'a') as f:
                f.writelines(['%s %d\n' % (x, label) for x in class_i_imgs])

    def filter_file(self, filename):
        '''To filter the file, eg. filetype or check the file validation

        Arguments:
            filename {string} -- a filename full path string

        Returns:
            bool -- whether this file is OK or not
        '''
        if self.filter is not None:
            assert hasattr(self.filter,'__call__'),'filter need to be a function'
            res = self.filter(filename)
            assert isinstance(res,bool),'filter must return a bool'
            return res
        for _type in self.file_type:
            if filename.endswith(_type):
                return True
        return False



def main():
    scanner = CommonFolderScanner(
        './', max_depth=5, classification_depth=1, combine_same_dir_as_one_class=True)
    # res = scanner.scan_folder(combine_same_dir_as_one_class=True)
    scanner.create_trainvaltest()


if __name__ == '__main__':
    main()
