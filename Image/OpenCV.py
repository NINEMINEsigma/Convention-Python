from ..Runtime.Config   import *

try:
    import cv2              as     cv2
    import cv2.data         as     cv2data
    from cv2.typing         import *
except ImportError as e:
    ImportingThrow(e, "OpenCV", ["opencv-python", "opencv-python-headless"])
try:
    import numpy            as     np
except ImportError as e:
    ImportingThrow(e, "OpenCV", ["numpy"])
try:
    from PIL                import ImageFile as ImageFile
    from PIL                import Image as Image
except ImportError as e:
    ImportingThrow(e, "OpenCV", ["pillow"])

from ..Runtime.File     import ToolFile

_Unwrapper2Str = lambda x: str(x)
_Wrapper2File = lambda x: ToolFile(x)

VideoWriter = cv2.VideoWriter
def mp4_with_MPEG4_fourcc() -> int:
    return VideoWriter.fourcc(*"mp4v")
def avi_with_Xvid_fourcc() -> int: 
    return VideoWriter.fourcc(*"XVID")
def avi_with_DivX_fourcc() -> int:
    return VideoWriter.fourcc(*"DIVX")
def avi_with_MJPG_fourcc() -> int:
    return VideoWriter.fourcc(*"MJPG")
def mp4_or_avi_with_H264_fourcc() -> int:
    return VideoWriter.fourcc(*"X264")
def avi_with_H265_fourcc() -> int:
    return VideoWriter.fourcc(*"H264")
def wmv_with_WMV1_fourcc() -> int:
    return VideoWriter.fourcc(*"WMV1")
def wmv_with_WMV2_fourcc() -> int:
    return VideoWriter.fourcc(*"WMV2")
def oggTheora_with_THEO_fourcc() -> int:
    return VideoWriter.fourcc(*"THEO")
def flv_with_FLV1_fourcc() -> int:
    return VideoWriter.fourcc(*"FLV1")
class VideoWriterInstance(VideoWriter):
    def __init__(
        self, 
        file_name:  Union[ToolFile, str], 
        fourcc:     int,
        fps:        float, 
        frame_size: tuple[int, int],
        is_color:   bool = True
        ):
        super().__init__(_Unwrapper2Str(file_name), fourcc, fps, frame_size, is_color)
    def __del__(self):
        self.release()

def wait_key(delay:int):
    return cv2.waitKey(delay)
def until_esc():
    return wait_key(0)

def is_current_key(key:str, *, wait_delay:int = 1):
    return wait_key(wait_delay) & 0xFF == ord(key[0])

class BasicViewable:
    def __init__(self, filename_or_index:Union[str, ToolFile, int]):
        self._capture: cv2.VideoCapture     = None
        self.stats:     bool                = True
        self.Retarget(filename_or_index)
    def __del__(self):
        self.Release()
    
    def __bool__(self):
        return self.stats
    
    def IsOpened(self):
        return self._capture.isOpened()
        
    def Release(self):
        if self._capture is not None:
            self._capture.release()
    def Retarget(self, filename_or_index:Union[str, ToolFile, int]):
        self.Release()
        if isinstance(filename_or_index, int):
            self._capture = cv2.VideoCapture(filename_or_index)
        else:
            self._capture = cv2.VideoCapture(_Unwrapper2Str(filename_or_index))
        return self
    
    def NextFrame(self) -> MatLike:
        self.stats, frame =self._capture.read()
        if self.stats:
            return frame
        else:
            return None
    
    def GetCaptrueInfo(self, id:int):
        return self._capture.get(id)
    def GetPropPosMsec(self):
        return self.GetCaptrueInfo(0)
    def GetPropPosFrames(self):
        return self.GetCaptrueInfo(1)
    def GetPropAviRatio(self):
        return self.GetCaptrueInfo(2)
    def GetPropFrameWidth(self):
        return self.GetCaptrueInfo(3)
    def GetPropFrameHeight(self):
        return self.GetCaptrueInfo(4)
    def GetPropFPS(self):
        return self.GetCaptrueInfo(5)
    def GetPropFourcc(self):
        return self.GetCaptrueInfo(6)
    def GetPropFrameCount(self):
        return self.GetCaptrueInfo(7)
    def GetPropFormat(self):
        return self.GetCaptrueInfo(8)
    def GetPropMode(self):
        return self.GetCaptrueInfo(9)
    def GetPropBrightness(self):
        return self.GetCaptrueInfo(10)
    def GetPropContrast(self):
        return self.GetCaptrueInfo(11)
    def GetPropSaturation(self):
        return self.GetCaptrueInfo(12)
    def GetPropHue(self):
        return self.GetCaptrueInfo(13)
    def GetPropGain(self):
        return self.GetCaptrueInfo(14)
    def GetPropExposure(self):
        return self.GetCaptrueInfo(15)
    def GetPropConvertRGB(self):
        return self.GetCaptrueInfo(16)
        
    def SetupCapture(self, id:int, value):
        self._capture.set(id, value)
        return self
    def SetPropPosMsec(self, value:int):
        return self.SetupCapture(0, value)
    def SetPropPosFrames(self, value:int):
        return self.SetupCapture(1, value)
    def SetPropAviRatio(self, value:float):
        return self.SetupCapture(2, value)
    def SetPropFrameWidth(self, value:int):
        return self.SetupCapture(3, value)
    def SetPropFrameHeight(self, value:int):
        return self.SetupCapture(4, value)
    def SetPropFPS(self, value:int):
        return self.SetupCapture(5, value)
    def SetPropFourcc(self, value):
        return self.SetupCapture(6, value)
    def SetPropFrameCount(self, value):
        return self.SetupCapture(7, value)
    def SetPropFormat(self, value):
        return self.SetupCapture(8, value)
    def SetPropMode(self, value):
        return self.SetupCapture(9, value)
    def SetPropBrightness(self, value):
        return self.SetupCapture(10, value)
    def SetPropContrast(self, value):
        return self.SetupCapture(11, value)
    def SetPropSaturation(self, value):
        return self.SetupCapture(12, value)
    def SetPropHue(self, value):
        return self.SetupCapture(13, value)
    def SetPropGain(self, value):
        return self.SetupCapture(14, value)
    def SetPropExposure(self, value):
        return self.SetupCapture(15, value)
    def SetPropConvertRGB(self, value:int):
        return self.SetupCapture(16, value)
    def SetPropRectification(self, value:int):
        return self.SetupCapture(17, value)
    
    @property
    def FrameSize(self) -> Tuple[float, float]:
        return self.GetPropFrameWidth(), self.GetPropFrameHeight()
    
class BasicCamera(BasicViewable):
    def __init__(self, index:int = 0):
        self.writer:    VideoWriter = None
        super().__init__(int(index))
    
    @override
    def Release(self):
        super().Release()
        if self.writer is not None:
            self.writer.release()
    
    def CurrentFrame(self):
        return self.NextFrame()
    
    def recording(
        self, 
        stop_pr:    Callable[[], bool], 
        writer:     VideoWriter,
        ):
        self.writer = writer
        while self.IsOpened():
            if stop_pr():
                break
            frame = self.CurrentFrame()
            cv2.imshow("__recording__", frame)
            writer.write(frame)
        cv2.destroyWindow("__recording__")
        return self

class ImageObject:
    def __init__(
        self,
        image:          Optional[Union[
            str,
            Self,
            BasicCamera,
            ToolFile, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]],
        flags:          int             = -1):
        self.__image:   MatLike         = None
        self.__camera:  BasicCamera = None
        self.current:   MatLike         = None
        if isinstance(image, BasicCamera):
            self.lock_from_camera(image)
        else:
            self.load_image(image, flags)

    @property
    def camera(self) -> BasicCamera:
        if self.__camera is None or self.__camera.IsOpened() is False:
            return None
        else:
            return self.__camera
    @property
    def image(self) -> MatLike:
        if self.current is not None:
            return self.current
        elif self.camera is None:
            return self.__image
        else:
            return self.__camera.CurrentFrame()

    @image.setter
    def image(self, image:          Optional[Union[
            str,
            Self,
            ToolFile, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]]):
        self.load_image(image)

    def load_from_nparray(
        self,
        array_: np.ndarray,
        code:   int = cv2.COLOR_RGB2BGR,
        *args, **kwargs
        ):
        self.__image = cv2.cvtColor(array_, code, *args, **kwargs)
        return self
    def load_from_PIL_image(
        self,
        image:  Image.Image,
        code:   int = cv2.COLOR_RGB2BGR,
        *args, **kwargs
    ):
        self.load_from_nparray(np.array(image), code, *args, **kwargs)
        return self
    def load_from_PIL_ImageFile(
        self,
        image:  ImageFile.ImageFile,
        rect:   Optional[Tuple[float, float, float, float]] = None
    ):
        return self.load_from_PIL_image(image.crop(rect))
    def load_from_cv2_image(self, image:  MatLike):
        self.__image = image
        return self
    def lock_from_camera(self, camera: BasicCamera):
        self.__camera = camera
        return self

    @property
    def dimension(self) -> int:
        return self.image.ndim
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        '''height, width, depth'''
        return self.image.shape
    @property
    def height(self) -> int:
        return self.shape[0]
    @property
    def width(self) -> int:
        return self.shape[1]

    def is_enable(self):
        return self.image is not None
    def is_invalid(self):
        return self.is_enable() is False
    def __bool__(self):
        return self.is_enable()
    def __MatLike__(self):
        return self.image

    def load_image(
        self, 
        image:          Optional[Union[
            str,
            ToolFile, 
            Self,
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]],
        flags:          int = -1
        ):
        """加载图片"""
        if image is None:
            self.__image = None
            return self
        elif isinstance(image, type(self)):
            self.__image = image.image
        elif isinstance(image, MatLike):
            self.__image = image
        elif isinstance(image, np.ndarray):
            self.load_from_nparray(image, flags)
        elif isinstance(image, ImageFile.ImageFile):
            self.load_from_PIL_ImageFile(image, flags)
        elif isinstance(image, Image.Image):
            self.load_from_PIL_image(image, flags)
        else:
            self.__image = cv2.imread(_Unwrapper2Str(image), flags)
        return self
    def save_image(self, save_path:Union[str, ToolFile], is_path_must_exist = False):
        """保存图片"""
        if is_path_must_exist:
            _Wrapper2File(save_path).try_create_parent_path()
        if self.is_enable():
            cv2.imwrite(_Unwrapper2Str(save_path), self.image)
        return self

    def show_image(
        self, 
        window_name:        str                         = "Image", 
        delay:              Union[int,str]              = 0,
        image_show_func:    Callable[[Self], None]      = None,
        *args, **kwargs
        ):
        """显示图片"""
        if self.is_invalid():
            return self
        if self.camera is not None:
            while (wait_key(1) & 0xFF != ord(str(delay)[0])) and self.camera is not None:
                # dont delete this line, self.image is camera flame now, see<self.current = None>
                self.current = self.image
                if image_show_func is not None:
                    image_show_func(self)
                if self.current is not None:
                    cv2.imshow(window_name, self.current)
                # dont delete this line, see property<image>
                self.current = None
        else:
            cv2.imshow(window_name, self.image)
            cv2.waitKey(delay = int(delay), *args, **kwargs)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(window_name)
        return self

    # 分离通道
    def split(self):
        """分离通道"""
        return cv2.split(self.image)
    def split_to_image_object(self):
        """分离通道"""
        return [ImageObject(channel) for channel in self.split()]
    @property
    def channels(self):
        return self.split()
    @property
    def blue_channel(self):
        return self.channels[0]
    @property
    def green_channel(self):
        return self.channels[1]
    @property
    def red_channel(self):
        return self.channels[2]
    @property
    def alpha_channel(self):
        return self.channels[3]
    def get_blue_image(self):
        return ImageObject(self.blue_channel)
    def get_green_image(self):
        return ImageObject(self.green_channel)
    def get_red_image(self):
        return ImageObject(self.red_channel)
    def get_alpha_image(self):
        return ImageObject(self.alpha_channel)

    # 混合通道
    def merge_channels_from_list(self, channels:List[MatLike]):
        """合并通道"""
        self.image = cv2.merge(channels)
        return self
    def merge_channels(self, blue:MatLike, green:MatLike, red:MatLike):
        """合并通道"""
        return self.merge_channels_from_list([blue, green, red])
    def merge_channel_list(self, bgr:List[MatLike]):
        """合并通道"""
        return self.merge_channels_from_list(bgr)

    # Transform
    def get_resize_image(self, width:int, height:int):
        if self.is_enable():
            return cv2.resize(self.image, (width, height))
        return None
    def get_rotate_image(self, angle:float):
        if self.is_invalid():
            return None
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.image, M, (w, h))
    def resize_image(self, width:int, height:int):
        """调整图片大小"""
        new_image = self.get_resize_image(width, height)
        if new_image is not None:
            self.image = new_image
        return self
    def rotate_image(self, angle:float):
        """旋转图片"""
        new_image = self.get_rotate_image(angle)
        if new_image is not None:
            self.image = new_image
        return self
    
    # 图片翻折
    def flip(self, flip_code:int):
        """翻转图片"""
        if self.is_enable():
            self.image = cv2.flip(self.image, flip_code)
        return self
    def horizon_flip(self):
        """水平翻转图片"""
        return self.flip(1)
    def vertical_flip(self):
        """垂直翻转图片"""
        return self.flip(0)
    def both_flip(self):
        """双向翻转图片"""
        return self.flip(-1)

    # 色彩空间猜测
    def guess_color_space(self) -> Optional[str]:
        """猜测色彩空间"""
        if self.is_invalid():
            return None
        image = self.image
        # 计算每个通道的像素值分布
        hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

        # 计算每个通道的像素值总和
        sum_b = np.sum(hist_b)
        sum_g = np.sum(hist_g)
        sum_r = np.sum(hist_r)

        # 根据像素值总和判断色彩空间
        if sum_b > sum_g and sum_b > sum_r:
            #print("The image might be in BGR color space.")
            return "BGR"
        elif sum_g > sum_b and sum_g > sum_r:
            #print("The image might be in GRAY color space.")
            return "GRAY"
        else:
            #print("The image might be in RGB color space.")
            return "RGB"

    # 颜色转化
    def get_convert(self, color_convert:int):
        """颜色转化"""
        if self.is_invalid():
            return None
        return cv2.cvtColor(self.image, color_convert)
    def convert_to(self, color_convert:int):
        """颜色转化"""
        if self.is_invalid():
            return None
        self.image = self.get_convert(color_convert)
    
    def is_grayscale(self):
        return self.dimension == 2
    def get_grayscale(self):
        if self.is_invalid():
            return None
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    def convert_to_grayscale(self):
        """将图片转换为灰度图"""
        self.image = self.get_grayscale()
        return self

    def get_convert_flag(
        self, 
        targetColorTypeName:Literal[
            "BGR", "RGB", "GRAY", "YCrCb"
            ]
        ) -> Optional[int]:
        """获取颜色转化标志"""
        flag = self.guess_color_space()
        if flag is None:
            return None
        
        if targetColorTypeName == "BGR":
            if flag == "RGB":
                return cv2.COLOR_RGB2BGR
            elif flag == "GRAY":
                return cv2.COLOR_GRAY2BGR
            elif flag == "YCrCb":
                return cv2.COLOR_YCrCb2BGR
        elif targetColorTypeName == "RGB":
            if flag == "BGR":
                return cv2.COLOR_BGR2RGB
            elif flag == "GRAY":
                return cv2.COLOR_GRAY2RGB
            elif flag == "YCrCb":
                return cv2.COLOR_YCrCb2RGB
        elif targetColorTypeName == "GRAY":
            if flag == "RGB":
                return cv2.COLOR_RGB2GRAY
            elif flag == "RGB":
                return cv2.COLOR_BGR2GRAY
        return None

    # 原址裁切
    def sub_image(self, x:int, y:int ,width:int ,height:int):
        """裁剪图片"""
        if self.is_invalid():
            return self
        self.image = self.image[y:y+height, x:x+width]
        return self

    # 直方图
    def equalizeHist(self, is_cover = False) -> MatLike:
        """直方图均衡化"""
        if self.is_invalid():
            return self
        result:MatLike = cv2.equalizeHist(self.image)
        if is_cover:
            self.image = result
        return result
    def calcHist(
        self, 
        channel:    Union[List[int], int],
        mask:       Optional[MatLike]       = None,
        hist_size:  Sequence[int]           = [256],
        ranges:     Sequence[float]         = [0, 256]
        ) -> MatLike:
        """计算直方图"""
        if self.is_invalid():
            return None
        return cv2.calcHist(
            [self.image],
            channel if isinstance(channel, list) else [channel],
            mask, 
            hist_size, 
            ranges)

    # 子集操作
    def sub_image_with_rect(self, rect:Tuple[float, float, float, float]):
        """裁剪图片"""
        if self.is_invalid():
            return self
        self.image = self.image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return self
    def sub_image_with_box(self, box:Tuple[float, float, float, float]):
        """裁剪图片"""
        if self.is_invalid():
            return self
        self.image = self.image[box[1]:box[3], box[0]:box[2]]
        return self
    def sub_cover_with_rect(self, image:Union[Self, MatLike], rect:Tuple[float, float, float, float]):
        """覆盖图片"""
        if self.is_invalid():
            raise ValueError("Real Image is none")
        if isinstance(image, MatLike):
            image = ImageObject(image)
        self.image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = image.image
        return self
    def sub_cover_with_box(self, image:Union[Self, MatLike], box:Tuple[float, float, float, float]):
        """覆盖图片"""
        if self.is_invalid():
            raise ValueError("Real Image is none")
        if isinstance(image, MatLike):
            image = ImageObject(image)
        self.image[box[1]:box[3], box[0]:box[2]] = image.image
        return self

    def operator_cv(self, func:Callable[[MatLike], Any], *args, **kwargs):
        func(self.image, *args, **kwargs)
        return self

    def stack(self, *args:Self, **kwargs) -> Self:
        images = [ image for image in args]
        images.append(self)
        return ImageObject(np.stack([np.uint8(image.image) for image in images], *args, **kwargs))
    def vstack(self, *args:Self) -> Self:
        images = [ image for image in args]
        images.append(self)
        return ImageObject(np.vstack([np.uint8(image.image) for image in images]))
    def hstack(self, *args:Self) -> Self:
        images = [ image for image in args]
        images.append(self)
        return ImageObject(np.hstack([np.uint8(image.image) for image in images]))
    
    def merge_with_blending(self, other:Self, weights:Tuple[float, float]):
        return ImageObject(cv2.addWeighted(self.image, weights[0], other.image, weights[1], 0))
    
    def add(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = cv2.add(self.image, image_or_value)
        else:
            self.image = cv2.add(self.image, image_or_value.image)
        return self
    def __add__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).add(image_or_value)
    def subtract(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = cv2.subtract(self.image, image_or_value)
        else:
            self.image = cv2.subtract(self.image, image_or_value.image)
        return self
    def __sub__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).subtract(image_or_value)
    def multiply(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = cv2.multiply(self.image, image_or_value)
        else:
            self.image = cv2.multiply(self.image, image_or_value.image)
        return self
    def __mul__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).multiply(image_or_value)
    def divide(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = cv2.divide(self.image, image_or_value)
        else:
            self.image = cv2.divide(self.image, image_or_value.image)
        return self
    def __truediv__(self, image_or_value:Union[Self, int]):
        return ImageObject(self.image.copy()).divide(image_or_value)
    def bitwise_and(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = cv2.bitwise_and(self.image, image_or_value)
        else:
            self.image = cv2.bitwise_and(self.image, image_or_value.image)
        return self
    def bitwise_or(self, image_or_value:Union[Self, int]):
        if isinstance(image_or_value, int):
            self.image = cv2.bitwise_or(self.image, image_or_value)
        else:
            self.image = cv2.bitwise_or(self.image, image_or_value.image)
        return self
    def bitwise_xor(self, image_or_value:Union[Self]):
        if isinstance(image_or_value, int):
            self.image = cv2.bitwise_xor(self.image, image_or_value)
        else:
            self.image = cv2.bitwise_xor(self.image, image_or_value.image)
        return self
    def bitwise_not(self):
        self.image = cv2.bitwise_not(self.image)
        return self
    def __neg__(self):
        return ImageObject(self.image.copy()).bitwise_not()
    
class NoiseImageObject(ImageObject):
    def __init__(
        self,
        height:     int,
        weight:     int,
        *,
        mean:       float   = 0,
        sigma:      float   = 25,
        dtype               = np.uint8
        ):
        super().__init__(NoiseImageObject.get_new_noise(
            None, height, weight, mean=mean, sigma=sigma, dtype=dtype
            ))
    
    @classmethod
    def get_new_noise(
        raw_image:  Optional[MatLike],
        height:     int,
        weight:     int,
        *,
        mean:       float   = 0,
        sigma:      float   = 25,
        dtype               = np.uint8
        ) -> MatLike:
        noise = raw_image
        if noise is None:
            noise = np.zeros((height, weight), dtype=dtype)
        cv2.randn(noise, mean, sigma)
        return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)

def Unwrapper(image:Optional[Union[
            str,
            ImageObject,
            ToolFile, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]]) -> MatLike:
    return image.image if isinstance(image, ImageObject) else ImageObject(image).image

def Wrapper(image:Optional[Union[
            str,
            ImageObject,
            ToolFile, 
            MatLike, 
            np.ndarray, 
            ImageFile.ImageFile,
            Image.Image
            ]]) -> ImageObject:
    return ImageObject(image)

class light_cv_window:
    def __init__(self, name:str):
        self.__my_window_name = name
        cv2.namedWindow(self.__my_window_name)
    def __del__(self):
        self.destroy()

    def show_image(self, image:Union[ImageObject, MatLike]):
        if self.__my_window_name is None:
            self.__my_window_name = "window"
        if isinstance(image, ImageObject):
            image = image.image
        cv2.imshow(self.__my_window_name, image)
        return self
    def destroy(self):
        if self.__my_window_name is not None and cv2.getWindowProperty(self.__my_window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow(self.__my_window_name)
        return self
    
    @property
    def window_rect(self):
        return cv2.getWindowImageRect(self.__my_window_name)
    @window_rect.setter
    def window_rect(self, rect:Tuple[float, float, float, float]):
        self.set_window_rect(rect[0], rect[1], rect[2], rect[3])
    
    def set_window_size(self, weight:int, height:int):
        cv2.resizeWindow(self.__my_window_name, weight, height)
        return self
    def get_window_size(self) -> Tuple[float, float]:
        rect = self.window_rect
        return rect[2], rect[3]
    
    def get_window_property(self, prop_id:int):
        return cv2.getWindowProperty(self.__my_window_name, prop_id)
    def set_window_property(self, prop_id:int, prop_value:int):
        cv2.setWindowProperty(self.__my_window_name, prop_id, prop_value)
        return self
    def get_prop_frame_width(self):
        return self.window_rect[2]
    def get_prop_frame_height(self):
        return self.window_rect[3]
    def is_full_window(self):
        return cv2.getWindowProperty(self.__my_window_name, cv2.WINDOW_FULLSCREEN) > 0
    def set_full_window(self):
        cv2.setWindowProperty(self.__my_window_name, cv2.WINDOW_FULLSCREEN, 1)
        return self
    def set_normal_window(self):
        cv2.setWindowProperty(self.__my_window_name, cv2.WINDOW_FULLSCREEN, 0)
        return self
    def is_using_openGL(self):
        return cv2.getWindowProperty(self.__my_window_name, cv2.WINDOW_OPENGL) > 0
    def set_using_openGL(self):
        cv2.setWindowProperty(self.__my_window_name, cv2.WINDOW_OPENGL, 1)
        return self
    def set_not_using_openGL(self):
        cv2.setWindowProperty(self.__my_window_name, cv2.WINDOW_OPENGL, 0)
        return self
    def is_autosize(self):
        return cv2.getWindowProperty(self.__my_window_name, cv2.WINDOW_AUTOSIZE) > 0
    def set_autosize(self):
        cv2.setWindowProperty(self.__my_window_name, cv2.WINDOW_AUTOSIZE, 1)
        return self
    def set_not_autosize(self):
        cv2.setWindowProperty(self.__my_window_name, cv2.WINDOW_AUTOSIZE, 0)
        return self
    
    def set_window_rect(self, x:int, y:int, weight:int, height:int):
        cv2.moveWindow(self.__my_window_name, x, y)
        return self.set_window_size(weight, height)

    def set_window_pos(self, x:int, y:int):
        cv2.moveWindow(self.__my_window_name, x, y)
        return self

    def wait_key(self, wait_time:int=0):
        return cv2.waitKey(wait_time)

def get_haarcascade_frontalface(name_or_default:Optional[str]=None):
    if name_or_default is None:
        name_or_default = "haarcascade_frontalface_default"
    return cv2.CascadeClassifier(cv2data.haarcascades+'haarcascade_frontalface_default.xml')

def detect_human_face(
    image:          ImageObject,
    detecter:       cv2.CascadeClassifier, 
    scaleFactor:    float                   = 1.1,
    minNeighbors:   int                     = 4,
    *args, **kwargs):
    '''return is Rect[]'''
    return detecter.detectMultiScale(image.image, scaleFactor, minNeighbors, *args, **kwargs)

class internal_detect_faces_oop(Callable[[ImageObject], None]):
    def __init__(self):
        self.face_cascade = get_haarcascade_frontalface()
    def __call__(self, image:ImageObject):
        gray = image.convert_to_grayscale()
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            image.operator_cv(cv2.rectangle,(x,y),(x+w,y+h),(255,0,0),2)
    
def easy_detect_faces(camera:BasicCamera):
    ImageObject(camera).show_image("window", 'q', internal_detect_faces_oop())
    
# 示例使用
if __name__ == "__main__":
    img_obj = ImageObject("path/to/your/image.jpg")
    img_obj.show_image()
    img_obj.resize_image(800, 600)
    img_obj.rotate_image(45)
    img_obj.convert_to_grayscale()
    img_obj.save_image("path/to/save/image.jpg")

# Override tool_file to tool_file_ex

class tool_file_cvex(ToolFile):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @override
    def load_as_image(self) -> ImageObject:
        self.data = ImageObject(self.get_path())
        return self.data
    
    @override
    def save(self, path = None):
        image:ImageObject   = self.data
        image.save_image(path if path is not None else self.get_path())
        return self

