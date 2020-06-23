import glob
import cv2
import warnings
import numpy as np
warnings.simplefilter("always")


class VideoWriter:
    def __init__(self, name, width, height, fps=25):
        # type: (str, int, int, int) -> None
        if not name.endswith('.mp4'):  # 保证文件名的后缀是.mp4
            name += '.mp4'
            warnings.warn('video name should ends with ".mp4"')
        self.__name = name          # 文件名
        self.__height = height      # 高
        self.__width = width        # 宽
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 如果是mp4视频，编码需要为mp4v
        self.__writer = cv2.VideoWriter(name, fourcc, fps, (width, height))

    def write(self, frame):
        if frame.dtype != np.uint8:  # 检查frame的类型
            raise ValueError('frame.dtype should be np.uint8')
        # 检查frame的大小
        row, col, _ = frame.shape
        if row != self.__height or col != self.__width:
            warnings.warn('长和宽不等于创建视频写入时的设置，此frame不会被写入视频')
            return
        self.__writer.write(frame)

    def close(self):
        self.__writer.release()


def write_video(path, glob_name_list, videoname='test.mp4', size=(512,512)):
    width = size[0]
    height = size[1]
    vw = VideoWriter(videoname, width, height)

    images_name_list = [glob.glob(path + glob_name) for glob_name in glob_name_list]

    for i in range(len(images_name_list[0])):
        img = None
        for imgs in images_name_list:
            if img is None:
                img = cv2.imread(imgs[i])
            else:
                img_new = cv2.imread(imgs[i])
                img = np.concatenate((img, img_new), axis=1)

        print('{} / {}'.format(i, len(images_name_list[0])))
        frame = img
        # 随机生成一幅图像
        # frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        # 写入图像
        vw.write(frame)
    # 关闭
    vw.close()


if __name__ == '__main__':

    path = "D:\\paper\\human_body_reconstruction\\code\\master\\exp\\demo\\model_best_hand_4_hip_0.05_hum_13_3dpw_13_3dpw\\debug\\images"

    glob_name_list = ['/*blend_smpl.jpg', '/*pred_box.jpg']
    video_name = 'test.mp4'
    write_video(path, glob_name_list, videoname=video_name,size=(1024,512))

    print('Done')

