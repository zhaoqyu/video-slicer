import argparse
import cv2
import os, sys
from pathlib import Path
import glob
from tqdm import tqdm

FILE = Path(__file__).resolve()  # 当前文件所在绝对路径
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 当前文件所在相对路径

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']
VID_MKFORMATS = ['avi', 'mp4']
VID = ['.' + x for x in VID_FORMATS]


class Frame(object):
    def __init__(
            self,
            video_path: ROOT,
            step: int = None,
            fps: int = 25,
            start: int = None,
            end: int = None,
            use_file_name: bool = False,
            img_format: str = 'jpg'
    ) -> None:
        # 参数含义同video_to_image()
        self.step = step if step is not None else 1
        self.fps = fps
        self.start = start
        self.end = end
        self.use_file_name = use_file_name
        self.img_format = img_format

        p = str(Path(video_path).resolve())
        if os.path.isdir(p):
            dir_file = os.listdir(video_path)
            files_v = [pa for pa in dir_file if os.path.splitext(pa)[-1] in VID]
            files = [os.path.join(p, file) for file in files_v]
        elif os.path.isfile(p):
            assert os.path.splitext(p)[-1] in [x for x in VID], f'Supported formats are:\nimages: {VID_FORMATS}'
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')

        self.nv = len(files)
        self.count = 0
        self.videos = files
        self.pps = 0  # 当前帧所处位号
        self.frame = 0

        assert self.nv > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nvideos: {VID_FORMATS}'
        assert self.img_format in IMG_FORMATS, f'Supported formats are:\nimages: {IMG_FORMATS}'
        assert self.step > 0, f'FPS {self.step} is illegal'
        assert self.fps > 0, f'FPS {self.fps} is illegal'

        print(f'INFORMATION\ntotal find {self.nv} files:\n{self.videos}')

        self.new_video(self.videos[0])

        if self.start is not None:
            self.start = self.time_str_to_sec(self.start) * self.fps
        else:
            self.start = 0

        if self.end is not None:
            self.end = self.time_str_to_sec(self.end) * self.fps
        else:
            self.end = self.frames

        if self.use_file_name:
            self.file_name = Path(self.videos[0]).stem + '_'
        else:
            self.file_name = ''

        print(f'make images\t{self.start} --> {self.end}')

    def __iter__(self):
        return self

    def __next__(self):
        if self.count == self.nv:
            raise StopIteration
        path = self.videos[self.count]
        ret, img = self.cap.read()
        if not ret:
            self.check_and_get_new()
            ret, img = self.cap.read()

        if self.frames < self.end:
            print('Video duration is too short')
            self.check_and_get_new()
            ret, img = self.cap.read()

        while self.pps < self.start:
            print(f'Loading Video File {path}', end='\r')
            self.pps += 1
            ret, img = self.cap.read()
            if not ret:
                self.check_and_get_new()
                ret, img = self.cap.read()
        if self.pps == self.start:
            print('\nLoad Complete\nSaving Images\nPlease Waiting···')

        while self.pps % self.step != 0:
            self.pps += 1
            ret, img = self.cap.read()
            if not ret:
                self.check_and_get_new()
                ret, img = self.cap.read()

        if self.pps >= self.start and self.pps < self.end:
            self.frame += 1
            self.pps += 1
            img_name = '{:0>8d}.{}'.format(self.frame - 1, self.img_format)
            img_name = self.file_name + img_name
            print(f'video {self.count + 1}/{self.nv} ({self.frame}/{(self.end - self.start) / self.fps}) {img_name}: ',
                  end='')
            return path, img, self.frame, img_name
        else:
            print()
            return self.check_next()

    def check_next(self):
        # 获取下一个视频中符合要求的第一帧图片
        self.check_and_get_new()
        path = self.videos[self.count]
        ret, img = self.cap.read()
        if not ret:
            self.check_and_get_new()
            ret, img = self.cap.read()

        if self.frames < self.end:
            print('Video duration is too short')
            self.check_and_get_new()
            ret, img = self.cap.read()

        while self.pps < self.start:
            print(f'Loading Video File {path}', end='\r')
            self.pps += 1
            ret, img = self.cap.read()
            if not ret:
                self.check_and_get_new()
                ret, img = self.cap.read()
        if self.pps == self.start:
            print('\nLoad Complete\nSaving Images\nPlease Waiting···')

        while self.pps % self.step != 0:
            self.pps += 1
            ret, img = self.cap.read()
            if not ret:
                self.check_and_get_new()
                ret, img = self.cap.read()

        if self.pps >= self.start and self.pps < self.end:
            self.frame += 1
            self.pps += 1
            img_name = '{:0>8d}.{}'.format(self.frame - 1, self.img_format)
            img_name = self.file_name + img_name
            print(f'video {self.count + 1}/{self.nv} ({self.frame}/{(self.end - self.start) / self.fps}) {img_name}: ',
                  end='')
            return path, img, self.frame, img_name
        else:
            return self.check_next()

    def check_and_get_new(self):
        # 检查是否已经读取完成，如果队列中还有视频，则开始读取新的视频
        self.count += 1
        self.cap.release()
        if self.count == self.nv:
            raise StopIteration
        else:
            path = self.videos[self.count]
            if self.use_file_name:
                self.file_name = Path(path).stem + '_'
            self.new_video(path)

    def time_str_to_sec(self, time):
        # 转换时间格式
        time = time.replace('：', ':')
        time_list = time.split(':')
        if len(time_list) == 1:
            return int(time_list[0])
        elif len(time_list) == 2:
            return int(time_list[0]) * 60 + int(time_list[1])
        elif len(time_list) == 3:
            return int(time_list[0]) * 3600 + int(time_list[1]) * 60 + int(time_list[2])

    def new_video(self, path):
        # 读取一个视频文件
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.pps = 0
        print(f'\nOpen Video {path}')

    def __len__(self):
        return self.nv


class Video():

    def __init__(
            self,
            images: ROOT,
            video_name: str = None,
            video_format: str = 'mp4',
            image_format: str = 'jpg',
            size: list = None,
            fps: int = 25,
            output: str = './output'
    ):

        '''
            images: 图片所在目录  
            video_name: 保存的视频名
            video_format: 保存的视频格式 
            image_format: 选取的图片格式
            size: 合并图片的resize大小
            fps: 合成视频的帧率
            output: 视频输出目录      
        '''

        print('Checking···\nINFORMATION')
        images = Path(images)
        size = size if size is not None else ['auto']

        assert os.path.isdir(images), 'Images path is invalid'
        # assert video_name is not None, 'Video name is None'
        assert video_format in VID_FORMATS, f'Supported video formats are:\nimages: {VID_MKFORMATS}'
        assert image_format in IMG_FORMATS, f'Supported images formats are:\nimages: {IMG_FORMATS}'

        if video_name is None:
            video_name = os.path.split(Path(images).resolve())[-1]

        imgs = os.listdir(images)
        imgs = [img for img in imgs if os.path.splitext(img)[-1] == ('.' + image_format)]
        imgs.sort()
        self.imgs = [os.path.join(Path(images), img) for img in imgs]

        self.size = size
        self.fps = fps
        self.output = output

        fourcc = {'mp4': 'mp4v', 'avi': 'DIVX'}
        self.fourcc = fourcc[video_format]

        if os.path.splitext(video_name)[-1] == ('.' + video_format):
            self.video = os.path.join(Path(output), video_name)
        else:
            video_name += '.' + video_format
            self.video = os.path.join(Path(output), video_name)

        print(
            f'Images from {images}\tVideo name: {video_name}\nImage Format: {image_format}\tVideo Format: {video_format}\nImage Size: {size}\tFPS: {fps}')
        print(f'Video will Save to {self.video}')

    def make_video(self):
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        fourcc = cv2.VideoWriter_fourcc(*self.fourcc)
        if self.size[0] == 'auto':
            img = self.imgs[0]
            self.size = cv2.imread(img).shape[::-1][1:]
        else:
            assert len(self.size) == 2, 'Size of video is illegal'

        video = cv2.VideoWriter(self.video, fourcc, self.fps, self.size)
        print('pls waiting···')
        with tqdm(self.imgs, desc='making video', ncols=80, unit='img') as t:
            for item in t:
                img = cv2.imread(item)
                video.write(img)
        video.release()
        cv2.destroyAllWindows()
        print('video Done!\nsave to {}'.format(self.video))


def video_to_image(
        video_path: ROOT,
        step: int = None,
        fps: int = 25,
        start: int = None,
        end: int = None,
        use_file_name: bool = False,
        img_format: str = 'jpg',
        save_path: ROOT = './images'
):
    # 视频拆帧函数
    '''
        video_path: ROOT -> 视频路径（或视频所在文件目录）
        step: int -> 间隔帧率，默认不间隔取帧
        fps: int -> 视频帧率，默认25帧
        start: str -> 开始时间（00:00:00），默认视频开始时间
        end: str -> 结束时间（00:00:00），默认视频结束时间
        use_file_name: bool -> 是否使用视频文件名作为命名规范
        img_format: str -> 保存的图片格式
        save_path: ROOT -> 保存的文件路径
    '''

    # 实例化迭代器
    video = Frame(video_path, step=step, fps=fps, start=start, end=end, use_file_name=use_file_name,
                  img_format=img_format)

    for path, img, frame, name in video:
        dir_path = os.path.join(Path(save_path), Path(path).stem)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_img_path = os.path.join(dir_path, name)
        cv2.imwrite(save_img_path, img)
        print(f'Save to {save_img_path}', end='\r')
    print('\nSave Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', type=str, help='input')
    parser.add_argument('-o', type=str, help='output')

    args = parser.parse_args()

    video_to_image(video_path=args.i, step=1, save_path=args.o)  # 视频拆帧

    # images_path = './images/ppelabt'
    # video = Video(images_path, video_format='avi')	# 实例化Video
    # video.make_video() # 视频组帧
    # 测试代码时实现了将视频mp4格式转化为avi格式
