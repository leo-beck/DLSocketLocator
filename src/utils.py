import numpy as np
import copy
import cv2 as cv
from scipy.spatial.transform import Rotation
import os
import imgaug.augmenters as iaa
import torch
import matplotlib.pyplot as plt
from imgaug.augmentables import Keypoint, KeypointsOnImage
from scipy.spatial.transform import Rotation as scipy_rot
import time


class ImgProcessingPipeline:
    def __init__(self, rotate=None, h_flip_idx=None, v_flip_idx=None, rot90_idx=None, buffer_distances=None,
                 crop_size_wh=None, resize_wh=None, background_paths=None, basic_aug=True, affine_aug=True):
        """
        ImageProcessingPipeline for both training and evaluation. Contains image augmentation methods and normalization
        :param rotate:              Angle in degrees for image augmentation
        :param h_flip_idx:          Indices of keypoints after flipping
        :param v_flip_idx:          Indices of keypoints after flipping
        :param rot90_idx:           Indices of keypoints after rotating 90 degrees
        :param buffer_distances:    Minimum distance between image boundaries and keypoints
        :param crop_size_wh:        Size of the image in pixels for cropping the original image
        :param resize_wh:           Final size of the image (=size of model input)
        :param background_paths:    Path to a folder of images used as random backgrounds
        :param basic_aug:           Augmentations that do not effect the location of keypoints in the image
        :param affine_aug:          Scaling, Rotating and Shearing the image
        """
        # Affine transformations
        if rotate:
            affine = iaa.Affine(scale=(0.9, 1.1), rotate=(-rotate, rotate), shear=(-3, 3))
        else:
            affine = iaa.Affine(scale=(0.9, 1.1))
        if affine_aug:
            self.trfm_affine = [TransformContainerIAASeq(iaa_seq=affine)]
        else:
            self.trfm_affine = None
        # Rot90 before flip
        if rot90_idx is not None:
            self.trfm_affine.append(TransformRot90(indices=rot90_idx))
        if v_flip_idx is not None or h_flip_idx is not None:
            self.trfm_affine.append(TransformFlip(v_idx=v_flip_idx, h_idx=h_flip_idx))
        # Crop
        if buffer_distances is not None:
            self.fn_crop = TransformCrop(buffer_distances=buffer_distances, crop_size_wh=crop_size_wh, resize_wh=resize_wh)
        # Background transformation
        self.trfm_bckgnd = None
        if background_paths is not None:
            self.trfm_bckgnd = TransformBackground(background_paths, resize_wh=resize_wh)
        # Basic Augmentations (keypoints stay the same)
        self.trfm_basic = []
        if basic_aug:
            trfm_basic = iaa.Sequential([
                iaa.Sometimes(0.3, iaa.Multiply((0.8, 1.2))),
                iaa.Sometimes(0.3, iaa.LinearContrast((0.8, 1.2))),
                iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(0, 0.04 * 255))),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=0.3)),
                iaa.Sometimes(0.3, iaa.Add((-10, 10))),
                iaa.Sometimes(0.3, iaa.Sharpen(alpha=0.04)),
                iaa.Sometimes(0.5, iaa.CoarseDropout(p=(0.02, 0.04), size_percent=(0.1, 0.2)))])
            self.trfm_basic.append(TransformContainerIAASeq(trfm_basic))
        # Normalize Pts
        self.fn_normalize = TransformPtNormalization(img_size=resize_wh)

    def __call__(self, img, pts):
        # Affine Augmentations
        if self.trfm_affine:
            for fn in self.trfm_affine:
                img, pts = fn(img, pts)
        # Crop
        img, pts, offsets = self.fn_crop(img, pts)
        # Background
        if self.trfm_bckgnd is not None:
            img = self.trfm_bckgnd(img, pts)
        # Basic Augmentations (keypoints stay the same)
        for fn in self.trfm_basic:
            img, pts = fn(img, pts)
        # Normalize images and points
        pts = self.fn_normalize(pts)
        img = img / 255
        return img, pts, offsets


class TransformFlip:
    def __init__(self, v_idx=None, h_idx=None, p_v=0.5, p_h=0.5):
        """
        :param v_idx:   Indices of keypoints after vertical flipping
        :param h_idx:   Indices of keypoints after horizontal flipping
        :param p_v:     Probability of vertical flip
        :param p_h:     Probability of horizontal flip
        """
        self.v_idx = v_idx
        self.h_idx = h_idx
        self.p_v = p_v
        self.p_h = p_h

    def __call__(self, img, kps=None):
        if kps is None:
            if np.random.random() < self.p_v:
                img = np.flipud(img)
            if np.random.random() < self.p_h:
                img = np.fliplr(img)
            return img
        h, w = np.shape(img)
        if self.v_idx is not None:
            if np.random.random() < self.p_v:
                img = np.flipud(img)
                kps[:, 1] = h - kps[:, 1]
                kps = kps[self.v_idx]
        if self.h_idx is not None:
            if np.random.random() < self.p_h:
                img = np.fliplr(img)
                kps[:, 0] = w - kps[:, 0]
                kps = kps[self.h_idx]
        return img, kps


class TransformRot90:
    def __init__(self, indices=None, p=0.5):
        """
        :param indices:     Indices of keypoints after a 90 degree rotation
        :param p:           Probability
        """
        self.p = p
        self.indices = indices

    def __call__(self, img, kps=None):
        if kps is None:
            if np.random.random() < self.p:
                img = np.rot90(img)
            return img
        h, w = np.shape(img)
        if np.random.random() < self.p:
            img = np.rot90(img)
            # Rot kps
            new_kps = []
            for kp in kps:
                # Keypoint is [x, y]
                y = w - kp[0]
                x = kp[1]
                new_kps.append([x, y])
            kps = np.array(new_kps)[self.indices]
        return img, kps


class TransformContainerIAASeq:
    """
    Transforms images and keypoints to the data types needed for the imgaug pipeline and retransforms them back...
    ... to numpy arrays
    """
    def __init__(self, iaa_seq):
        self.iaa_seq = iaa_seq

    def __call__(self, img, kps=None):
        if len(np.shape(img)) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        if kps is None:
            img_aug = self.iaa_seq(image=img)
            return cv.cvtColor(img_aug, cv.COLOR_RGB2GRAY)
        kps_imgaug = self._labels_to_kpts(kps, np.shape(img))
        img_aug, kps_aug = self.iaa_seq(image=img, keypoints=kps_imgaug)
        kps, _, _ = self._kpts_to_labels(kps_aug)
        return cv.cvtColor(img_aug, cv.COLOR_RGB2GRAY), np.reshape(kps, (-1, 2))

    @staticmethod
    def _labels_to_kpts(pts, img_shape):
        """
        Numpy arrays to imgaug keypoint class
        """
        imgaug_kps = []
        for x, y in pts:
            imgaug_kps.append(Keypoint(x=x, y=y))
        kps = KeypointsOnImage(imgaug_kps, shape=img_shape)
        return kps

    @staticmethod
    def _kpts_to_labels(kps):
        """
        Imgaug keypoint class to numpy array
        """
        result_batch_element = []
        result_x = []
        result_y = []
        for kp in kps:
            result_batch_element.append(kp.x)
            result_batch_element.append(kp.y)
            result_x.append(kp.x)
            result_y.append(kp.y)
        return result_batch_element, result_x, result_y


class TransformCrop:
    def __init__(self, buffer_distances=None, crop_size_wh=None, resize_wh=None):
        """
        Crops the image and resizes it
        :param buffer_distances:    The minimum amount of pixels between a keypoint and the image border
        :param crop_size_wh:        Width, height of crop from the original image
        :param resize_wh:           Final size of image (=model input)
        """
        if buffer_distances is None:
            buffer_distances = [30, 30, 30, 30]
        if crop_size_wh is None:
            crop_size_wh = [300, 300]
        if resize_wh is None:
            resize_wh = [100, 100]
        self.dist = buffer_distances
        self.crop = crop_size_wh
        self.resize = resize_wh
        self.offsets = None
        self.last_offsets = None

    def __call__(self, img, kps):
        h, w = np.shape(img)
        # Get roi from kps
        roi_le = np.min(kps[:, 0]) - self.dist[0]
        roi_up = np.min(kps[:, 1]) - self.dist[1]
        roi_ri = np.max(kps[:, 0]) + self.dist[2]
        roi_lo = np.max(kps[:, 1]) + self.dist[3]
        # Create boundaries for random vars
        le_min = np.max([0, roi_ri - self.crop[0]])
        le_max = np.max([0, np.min([roi_le, w - self.crop[0]])])
        up_min = np.max([0, roi_lo - self.crop[1]])
        up_max = np.max([0, np.min([roi_up, h - self.crop[1]])])
        # Get rands
        x = int(np.random.rand() * (le_max - le_min) + le_min)
        y = int(np.random.rand() * (up_max - up_min) + up_min)
        offset = np.array([x, y])
        # Crop
        img_crop = img[y: y + self.crop[1], x: x + self.crop[0]]
        kps_crop = kps - offset
        # Resize img, kp
        img_resized = cv.resize(img_crop, self.resize)
        kps_resized = kps_crop * self.resize / self.crop
        return img_resized, kps_resized, offset


class TransformBackground:
    def __init__(self, background_paths, resize_wh, buffer_min=None, p=0.3):
        """
        :param background_paths:    Paths of images used as random backgrounds
        :param resize_wh:           Output size of the background images
        :param buffer_min:          The minimum amount of pixels between a keypoint and the random background
        :param p:                   Probability
        """
        self.p = p
        if buffer_min is None:
            buffer_min = [5, 5, 5, 5]
        self.buffer_min = buffer_min
        self.resize_wh = resize_wh
        self.imgs = []
        for p in background_paths:
            self.imgs.append(cv.imread(p))
        # Augmentation pipeline
        affine = iaa.Affine(scale=(0.5, 2), rotate=(-180, 180), shear=(-45, 45))
        self.trfm_affine = [TransformContainerIAASeq(iaa_seq=affine),
                            TransformRot90(),
                            TransformFlip()]

    def __call__(self, img, kps):
        if np.random.random() < self.p:
            return img
        h, w = np.shape(img)
        # Get roi from kps
        le = 0
        up = 0
        lo = h
        ri = w
        # Make sure buffer is regarded
        le_high = np.max([0, np.min(kps[:, 0]) - self.buffer_min[0]]).astype(int)
        if le_high != 0:
            le = np.random.randint(low=0, high=le_high)
        up_high = np.max([0, np.min(kps[:, 1]) - self.buffer_min[1]]).astype(int)
        if up_high != 0:
            up = np.random.randint(low=0, high=up_high)
        lo_low = np.min([h, np.max(kps[:, 1]) + self.buffer_min[1]]).astype(int)
        if lo_low != h:
            lo = np.random.randint(low=lo_low, high=h)
        ri_low = np.min([w, np.max(kps[:, 0]) + self.buffer_min[0]]).astype(int)
        if ri_low != w:
            ri = np.random.randint(low=ri_low, high=w)
        # Load background
        new_img = copy.deepcopy(self.imgs[np.random.randint(low=0, high=len(self.imgs)-1)])
        # Affine background
        for fn in self.trfm_affine:
            new_img = fn(new_img)
        # Resize
        new_img = cv.resize(new_img, self.resize_wh)
        # Insert roi in background
        new_img[up:lo, le:ri] = img[up:lo, le:ri]
        return new_img


class TransformPtNormalization:
    """
    Normalizes keypoints from pixel coordinates to a range of [-1;1]
    """
    def __init__(self, img_size):
        self.img_size = np.array(img_size)

    def __call__(self, keypoints):
        kps_norm = keypoints / self.img_size * 2 - 1
        return kps_norm


class Dataset:
    def __init__(self, img_paths, transform=None, kpts2d=None, rob_poses=None, augment_onthefly=True):
        """
        :param img_paths:           Paths of all the images of the dataset
        :param transform:           An instance of the ImgProcessingPipeline class
        :param kpts2d:              Keypoints for each image in order of the img_paths
        :param rob_poses:           Robot poses for each image in order of the img_paths
        :param augment_onthefly:    Augments images on-the-fly (=New unique augmentation of the image...
                                    ...for every __getitem__ call)
        """
        if rob_poses is None:
            self.train_mode = True
        else:
            self.train_mode = False
        self.images = [cv.imread(e, cv.IMREAD_GRAYSCALE) for e in img_paths]
        self.rob_poses = rob_poses
        self.transform = transform
        self.pts2d = kpts2d
        self.augment_onthefly = augment_onthefly
        # Do augmentation once during init if on-the-fly augmentation isnt wanted
        if not self.augment_onthefly:
            imgs_aug, pts_aug, offsets = [], [], []
            for img, pt in zip(self.images, self.pts2d):
                img_aug, pt_aug, offset = self.transform(img, pt)
                imgs_aug.append(img_aug), pts_aug.append(pt_aug), offsets.append(offset)
            self.images = imgs_aug
            self.pts2d = pts_aug
            self.offsets = offsets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # Load image, label
        img = copy.deepcopy(self.images[item])
        kps = copy.deepcopy(self.pts2d[item])
        # Transform
        if self.augment_onthefly:
            img, kps, offset = self.transform(img, kps)
        else:
            offset = self.offsets[item]
        # To batch dimensions and correct data types
        img = torch.tensor(img).unsqueeze(0).to(torch.float)
        kps = torch.tensor(np.array(kps).flatten())
        # Return either kpts (during training) or rob_pose (during val & test)
        if self.train_mode:
            return img, kps, offset
        else:
            rob_pose = self.rob_poses[item]
            return img, kps, offset, rob_pose


class WorldPoseTransformer:
    def __init__(self, pts_3d, h_cam_intern, rvec, tvec, h_from_rob_pose_function):
        """
        This class is crucial for the whole procedure.
            - It is used to calculate the object's pose relativ to the camera frame as well as the world frame
            - It is used to reproject points using the mean pose of the object to pseudo-label unlabeled training images
            - It is used to calculate the evaluation error from predicted keypoints by comparing the

        :param pts_3d:                      3D points of the real object [in mm]
        :param h_cam_intern:                Internal Calibration matrix of the camera
        :param rvec:                        Rotation vector used as initial guess for the PnP algorithm
        :param tvec:                        Translation vector used as initial guess for the PnP algorithm [in mm]
        :param h_from_rob_pose_function:    This function is used to load ./data/robot_poses.json. The result must be a
                                            homogenous 4x4 transformation matrix. See h_from_ur_pose as an example

        Nomenclature of 4x4 homogenous transformation matrices:
            cHo:    Object's pose relative to the camera frame (Received from keypoints, camera calibration with PnP)
            wHc:    Camera's pose relative to the world frame (Received from the robot's control system)
            wHo:    Object's pose relative to the world frame (Received from wHc * cHo)
        """
        self.pts_3d = np.array(pts_3d).astype(float)
        self.h_cam_intern = np.array(h_cam_intern).astype(float)
        self.r_vec = np.array(rvec).astype(float)
        self.t_vec = np.array(tvec).astype(float)
        self.h_pred = []
        self.h_from_rob_pose_fn = h_from_rob_pose_function
        self.chos = []

    def __call__(self, pts_2d, rob_pose, epnp=False):
        """
        :param pts_2d:                  Labeled keypoints of the images
        :param rob_pose:                Robot pose
        :param epnp:                    Use EPnP instead of Iterative PnP if True
        """
        if isinstance(pts_2d, torch.Tensor):
            pts_2d = pts_2d.view(-1, 2).detach().numpy()
        if epnp:
            pnp_flag = cv.SOLVEPNP_EPNP
        else:
            pnp_flag = cv.SOLVEPNP_ITERATIVE
        success, r, t = cv.solvePnP(objectPoints=self.pts_3d,
                                    imagePoints=np.array(pts_2d).astype(float),
                                    cameraMatrix=self.h_cam_intern,
                                    distCoeffs=np.array([]),
                                    flags=pnp_flag,
                                    useExtrinsicGuess=True,
                                    rvec=self.r_vec,
                                    tvec=self.t_vec)
        # r, t to cHo (object in camera) matrix
        cHo = self.h_from_cv_pose(t=t, r=r)
        self.chos.append(cHo)
        # wHc (camera in world) from rob_pose
        wHc = self.h_from_rob_pose_fn(rob_pose)
        # wHo (object in world) matrix
        wHo = np.dot(wHc, cHo)
        self.h_pred.append(wHo)

    def reset_log(self):
        self.h_pred = []
        self.chos = []

    def _get_diffs(self):
        mean_t = np.mean(np.array(self.h_pred)[:, :3, 3], axis=0)
        diffs = np.abs(np.array(self.h_pred)[:, :3, 3] - mean_t)
        return diffs

    def get_mean_xyz_error(self):
        diffs = self._get_diffs()
        mean_diffs = np.mean(diffs, axis=0)
        return mean_diffs

    def get_max_xyz_error(self):
        diffs = self._get_diffs()
        max_diff_xy = np.max(diffs, axis=0)
        return max_diff_xy

    def reproject_points(self, rob_poses, pts_2d, ignore_rotation=False):
        """
        :param rob_poses:           Robot pose
        :param ignore_rotation:     Only needed for rotationally symmetrical objects where extreme points are labeled...
                                    ... in this case the rotation of the object relative to the camera is ignored
        :return:
        """
        # Log poses with labeled keypoints
        for key in pts_2d.keys():
            self.__call__(pts_2d[key], rob_poses[key], epnp=True)
        # Get mean H
        wHo = np.eye(4)
        wHo[:3, :3] = scipy_rot.from_matrix(np.array(self.h_pred)[:, :3, :3]).mean().as_matrix()
        wHo[:3, 3] = np.mean(np.array(self.h_pred)[:, :3, 3], axis=0)
        # Reproject 3d keypoints into 2d images
        all_reprojected_points = {}
        for key in rob_poses.keys():
            # Use labeled if labeled
            if key in pts_2d.keys():
                all_reprojected_points[key] = pts_2d[key]
                continue
            # Reproject keypoints using cHo = oHw * wHc
            wHc = self.h_from_rob_pose_fn(rob_poses[key])
            cHo = np.dot(np.linalg.inv(wHc), wHo)
            if ignore_rotation:
                r, _ = cv.Rodrigues(scipy_rot.from_matrix(np.array(self.chos)[:, :3, :3]).mean().as_matrix())
            else:
                r, _ = cv.Rodrigues(cHo[:3, :3])
            t = cHo[:3, 3]
            # Get rob pose as
            proj_points2d, _ = cv.projectPoints(objectPoints=np.array(self.pts_3d).astype(float),
                                                cameraMatrix=self.h_cam_intern, distCoeffs=np.array([]), tvec=t, rvec=r)
            proj_points2d = np.reshape(proj_points2d, (-1, 2))
            all_reprojected_points[key] = proj_points2d
        return all_reprojected_points

    @staticmethod
    def h_from_cv_pose(t, r):
        """
        :param t:   Translation vector [in mm]
        :param r:   Rotation vector [rodriguez]
        :return:    4x4 homogenous transformatio matrix
        """
        h = np.eye(4)
        r_mx, _ = cv.Rodrigues(r)
        h[:3, :3] = r_mx
        if np.shape(t) == (3, 1):
            h[:3, 3] = t[:, 0]
        else:
            h[:3, 3] = t
        return h


def h_from_ur_pose(pose):
    """
    :param pose:    Robot Pose as stored in data/robot_poses.json
    :return:        Must return a 4x4 homogenous transformation matrix
    """
    rot = pose[3:]
    trans = np.array(pose[:3])*1000
    rmx = Rotation.from_rotvec(rot).as_matrix()
    r_t = np.hstack((rmx, np.array(trans).reshape((3, 1))))
    return np.vstack((r_t, [0, 0, 0, 1]))


def get_h_cam_in_pinhole(params):
    """
    :param params:  Dictionary of internal camera parameters. Must contain the keys "f" in mm, "ux", "uy" in...
                    ...pixels and "pixel_size" in mm
    :return:        3x3 camera calibration matrix
    """
    # Pinhole camera
    cam = np.array([[params['f']/params['pixel_size'], 0, params['ux']],
                    [0, params['f']/params['pixel_size'], params['uy']],
                    [0, 0, 1]])
    return cam


def dsnt_denorm(pts_2d, img_size_wh):
    """
    Transforms the keypoints from the range [-1;1] to [0, w] resp. [0, h] where w, h is the images width and height
    :param pts_2d:          Torch.tensor of 2D keypoints
    :param img_size_wh:     Width, Height of image
    :return:
    """
    pts_2d_shape = pts_2d.shape
    pts_2d_denorm = (pts_2d.view(pts_2d_shape[0], -1, 2) + 1) * torch.tensor(img_size_wh).to(torch.float) / 2
    return pts_2d_denorm.view(pts_2d_shape)


def decode_pred_pts(pts_batch, resize_wh, cropsize_wh, offsets=None):
    """
    :param pts_batch:       Predicted keypoints normalized to the rang eof [-1, 1]
    :param resize_wh:       Width, Height of resized image
    :param cropsize_wh:     Width, Height of cropped image
    :param offsets:         Offsets [width, height] from the image crop transformation
    :return:                Returns xy coordinates of the predicted keypoints in the *original image*
    """
    resize_wh = np.array([resize_wh[1], resize_wh[0]])
    cropsize_wh = np.array(cropsize_wh)
    #  0-1 decoded points after DSNT to 0-w/h pixels
    pts_denorm = dsnt_denorm(pts_batch, resize_wh)
    if offsets is None:
        return pts_denorm
    #  Undo resizing
    pts_desized = pts_denorm * cropsize_wh / resize_wh
    #  Undo cropping by adding coordinate offsets to get original coordinates
    pts_decrop = pts_desized + offsets
    return pts_decrop


def plot_loss_curves(train_loss, val_loss, val_loss_i, file_path):
    """
    :param train_loss:  Training loss
    :param val_loss:    Validation loss
    :param val_loss_i:  Validation loss indices
    :param file_path:   Path of the resulting image of the loss curves
    :return:
    """
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Averaged (Euclidean, Jensen-Shannon) Training Loss', color=color)
    ax1.plot(train_loss, color=color)
    ax1.set_ylim([0, 2])
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Average Validation error in mm', color=color)  # we already handled the x-label with ax1
    ax2.plot(val_loss_i, val_loss, color=color)
    ax2.set_ylim([0, 2])
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(file_path)
    plt.show()


def print_net_info(net, size=(100, 100)):
    """
    :param net:     pytorch model
    :param size:    input image size
    """
    # Info
    trainables = sum(p.numel() for p in net.parameters() if p.requires_grad)
    all = sum(p.numel() for p in net.parameters())
    print(f"Start Training with {int(trainables / 1e3)}k trainable params")
    test_input = torch.rand(1, 1, size[0], size[1])
    t = time.perf_counter()
    test = net(test_input)
    t_fwd_pass = np.round(time.perf_counter() - t, 3)
    print(f"A forward pass with 1 image takes {t_fwd_pass} sec")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return all, trainables, t_fwd_pass


def draw_cross(img, middle, size=None, color=None, thickness=1, notation=None, size_text=1, thickness_notation=2):
    """
    Annotates keypoints in images by drawing a cross
    """
    result = copy.copy(img)
    if len(np.shape(result)) == 2:
        result = cv.cvtColor(result, cv.COLOR_GRAY2RGB)
    if size is None:
        size = [20, 20]
    if color is None:
        color = [255, 0, 0]
    # Round middle
    middle = np.round(np.array(middle)).astype(int)
    # Vertical line
    cv.line(result, (middle[0], middle[1]-size[1]), (middle[0], middle[1]+size[1]), color, thickness)
    # Horizontal line
    cv.line(result, (middle[0]-size[0], middle[1]), (middle[0]+size[0], middle[1]), color, thickness)
    if notation is not None:
        cv.putText(img=result, text=str(notation), org=middle-[32, 6], fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=size_text, color=color, thickness=thickness_notation)
    return result


def save_prediction(img, pts, pts_2=None, path_n_name=None):
    """
    :param img:                 Image torch.tensor of shape B,C,H,W (pytorch format)
    :param pts:                 Keypoints as torch.tensor
    :param pts_2:               Keypoints as torch.tensor
    :param path_n_name:         Path to save images, if None images will be displayed via cv.imshow
    :return:
    """
    b, c, h, w = img.shape
    pts = np.reshape(pts.detach().numpy(), (b, -1, 2))
    if pts_2 is not None:
        pts_2 = np.reshape(pts_2.detach().numpy(), (b, -1, 2))
    for i in range(len(img)):
        img_np = img.detach().numpy()[i]
        img_np = np.uint8(img_np * 255)
        img_np = np.reshape(img_np, (h, w, c))
        if c == 1:
            img_np = cv.cvtColor(img_np, cv.COLOR_GRAY2RGB)
        for n, pt in enumerate(pts[i]):
            img_np = draw_cross(img_np, middle=pt, color=[0, 255, 0])
        if pts_2 is not None:
            for n, pt in enumerate(pts_2[i]):
                img_np = draw_cross(img_np, middle=pt, color=[255, 0, 0])
        if path_n_name is not None:
            cv.imwrite(path_n_name, img_np)
        else:
            cv.imshow('pred', img_np)
            cv.waitKey(0)


def denormalize_kpts(pts, img_size):
    """ Changes keypoints from [-1;1] range to [0; h/w]
    :param pts:         Torch.tensor of points
    :param img_size:    Size of image
    :return:
    """
    tensor_shape = pts.shape
    tensor_denorm = (pts.view(tensor_shape[0], -1, 2) + 1) * torch.tensor(img_size).to(torch.float) / 2
    return tensor_denorm.view(tensor_shape)


def create_results_paths(dataset_name):
    """
    :param dataset_name:    Name of the dataset in the results folder
    :return:
    """
    path_results = os.path.join('results', dataset_name)
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    path_model = os.path.join(path_results, 'weights.pth')
    path_loss = os.path.join(path_results, 'loss.png')
    path_imgs = os.path.join(path_results, 'test_images')
    if not os.path.exists(path_imgs):
        os.makedirs(path_imgs)
    return path_model, path_loss, path_imgs
