#!/usr/bin/env python3

# modified significantly from
# https://github.com/opencv/opencv/blob/master/samples/python/stitching_detailed.py
# https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp

import cv2
import numpy as np
from pcv.vidIO import VideoReader
import functools
import time

def sharpness(image):
    return cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                         cv2.CV_64F).var()

def log(func):
    ''' A logging wrapper to print timing of each logged function call. '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        print(f'{func.__name__.replace("_"," ").strip()+"...":<30}', end=' ',
              flush=True)
        output = func(*args, **kwargs)
        print(f'{time.perf_counter() - start:.3f}s')
        return output

    return wrapper


class Stitcher:
    feature_detector_options = dict()
    for option, constructor in (
            ('orb'  , 'cv2.ORB_create'),
            ('surf' , 'cv2.xfeatures2d_SURF.create'),
            ('brisk', 'cv2.BRISK_create'),
            ('akaze', 'cv2.AKAZE_create'),
            ('sift' , 'cv2.xfeatures2d_SIFT.create'),
            ):
        try:
            feature_detector_options[option] = eval(constructor)
        except AttributeError:
            print(f'{option.upper()} not available')

    matcher_options = ('homography', 'affine')

    estimator_options = {
        'homography' : cv2.detail_HomographyBasedEstimator,
        'affine'     : cv2.detail_AffineBasedEstimator,
    }

    bundle_adjustment_options = {
        'ray'    : cv2.detail_BundleAdjusterRay,
        'reproj' : cv2.detail_BundleAdjusterReproj,
        'affine' : cv2.detail_BundleAdjusterAffinePartial,
        'no'     : cv2.detail_NoBundleAdjuster,
    }

    wave_correction_options = {
        'vertical'   : cv2.detail.WAVE_CORRECT_VERT,
        'no'         : None,
        'horizontal' : cv2.detail.WAVE_CORRECT_HORIZ,
    }

    exposure_compensation_options = {
        'gain_blocks'    : cv2.detail.ExposureCompensator_GAIN_BLOCKS,
        'gain'           : cv2.detail.ExposureCompensator_GAIN,
        'channel'        : cv2.detail.ExposureCompensator_CHANNELS,
        'channel_blocks' : cv2.detail.EXPOSURE_COMPENSATOR_CHANNELS_BLOCKS,
        'no'             : cv2.detail.ExposureCompensator_NO,
    }

    warp_options = (
        'spherical',
        'plane',
        'affine',
        'cylindrical',
        'fisheye',
        'stereographic',
        'compressedPlaneA2B1',
        'compressedPlaneA1.5B1',
        'compressedPlanePortraitA2B1',
        'compressedPlanePortraitA1.5B1',
        'paniniA2B1',
        'paniniA1.5B1',
        'paniniPortraitA2B1',
        'paniniPortraitA1.5B1',
        'mercator',
        'transverseMercator', # equirectangular?
    )

    seam_find_options = {
        'gc_color'     : cv2.detail_GraphCutSeamFinder('COST_COLOR'),
        'gc_colorgrad' : cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD'),
        'dp_color'     : cv2.detail_DpSeamFinder('COLOR'),
        'dp_colorgrad' : cv2.detail_DpSeamFinder('COLOR_GRAD'),
        'voronoi'      : cv2.detail.SeamFinder_createDefault(
            cv2.detail.SeamFinder_VORONOI_SEAM),
        'no' : cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO),
    }

    timelapse_options = {
        'no'    : None,
        'as_is' : cv2.detail.Timelapser_AS_IS,
        'crop'  : cv2.detail.TIMELAPSER_CROP,
    }

    blend_options = ('multiband', 'feather', 'no')

    def __init__(self, images, megapixels=dict(), feature_detector=None,
                 matcher_details=dict(), confidence_threshold=1.0,
                 estimator=None, bundle_adjustment=dict(),
                 wave_correction=None, exposure_compensation=dict(),
                 warp_type=None, seam_finder=None, timelapse=None,
                 blend_type=None, blend_strength=5, result_name='result.jpg'):
        ''' Stitches 'images' as specified.

        'images' is an iterable of images or image filenames to stitch.
        'megapixels' is a dictionary of 'registration', 'composition', and
            'seam' megapixel counts, defaulting to 0.6, -1, and 0.1
             respectively if unspecified. Set any to -1 to use image as given.
        'feature_detector' is a key from self.feature_detector_options.
            Defaults to 'orb' for efficiency and robustness.
        'matcher_details' details the feature matching functionality to use.
            It is a dictionary with keys 'matcher', 'match_confidence',
            'try_cuda', and 'rangewidth'. Defaults to a 'homography' matcher
            with 'rangewidth' of 2 for fast sequential frame processing.
            Detailed descriptions are in self._match_features docs.
        'confidence_threshold' is a float specifying the required confidence
            for an image to be considered part of the stitching. Defaults to
            1.0. TODO check if correct or if actually the confidence of the
            user that all images are valid parts of the stitching.
        'estimator' is one of self.estimator_options. Defaults to 'homography'.
            It is strongly suggested to have the same estimator type as the
            matcher.
        'bundle_adjustment' details which bundle adjustment to use. It is a
            dictionary with keys 'method' (defaults to 'ray' if unspecified)
            and 'refine_params'. Details are in self._bundle_adjustment docs.
        'wave_correction' is the orientation to try to set the stitched result
            to. Defaults to 'vertical' from self.wave_correction_options.
        'exposure_compensation' is a dictionary specifying 'type', 'feeds', and
            'block_size' (as appropriate). Detailed specification is provided
            in self._set_exposure_compensator. Default 'gain_blocks' type.
        'warp_type' is one of self.warp_options. Defaults to 'spherical'.
        'seam_finder' is one of self.seam_find_options. Defaults to 'gc_color'.
        'timelapse' is one of self.timelapse_options. Defaults to 'no' for a
            stitched and blended result instead of saving individual warped
            images.
        'blend_type' is the type of blending applied between overlapping
            images, from self.blending_options. Defaults to 'multiband'.
        'blend_strength' is a number specifying how strongly to apply blending.
        'result_name' is the filename of the generated output image. Ignored
            if 'timelapse' is not 'no'.

        '''
        self._megapixels = {
            'registration' : 0.6,
            'composition'  : -1,
            'seam'         : 0.1,
            **megapixels
        }

        self._set_images(images)
        self._detect_features(feature_detector)
        self._match_features(**matcher_details)
        # self._save_graph(graph_filename)
        self._remove_bad_images(confidence_threshold)
        self._estimate_camera_parameters(estimator)
        self._bundle_adjustment(**bundle_adjustment)
        self._wave_correction(wave_correction)
        self._exposure_compensation(exposure_compensation, warp_type)
        self._find_seams(seam_finder)
        if self._compose(timelapse, blend_type, blend_strength, result_name):
            cv2.waitKey(0) # wait for user to close image

    @log
    def _set_images(self, images):
        self._images = []
        for image in images:
            if isinstance(image, str):
                image = cv2.imread(image)
                if image is None:
                    print(f"Cannot read image {image} - skipping")
                    continue
            self._images.append(image)

    @log
    def _detect_features(self, feature_detector):
        ''' Detects and stores the features of the stored images.

        'feature_detector' should be a key from self.feature_detector_options.
            Defaults to 'orb' for efficieny and robustness.

        '''
        self._feature_detector = feature_detector or \
                self.default(self.feature_detector_options)
        detector = self.feature_detector_options[self._feature_detector]()

        # images from video are all the same size
        # NOTE if intending to use images of different sizes, perform these
        #    steps for each image and store the results for later reference
        # -> may be worth implementing a dictionary by image size
        self._full_size = width, height = self.get_size(self._images[0])
        image_megapixels = width * height / 1e6

        if self._megapixels['registration'] < 0:
            self._registration_scale = 1
        else:
            self._registration_scale = min(1.0, np.sqrt(
                self._megapixels['registration'] / image_megapixels))
        self._seam_scale = min(1.0, np.sqrt(self._megapixels['seam'] /
                                            image_megapixels))
        self._seam_registration_aspect = \
            self._seam_scale / self._registration_scale

        features_seams = []
        for image in self._images:
            features_seams.append(
                self.detect_features(image, detector, self._registration_scale,
                                     self._seam_scale))

        self._features = []
        self._seam_images = []
        for features, seams in features_seams:
            self._features.append(features)
            self._seam_images.append(seams)

    @staticmethod
    def detect_features(full_image, detector, registration_scale, seam_scale):
        ''' Returns the features and seam-scaled image of 'image'. '''
        rescale = lambda image, scale: image if scale == 1 else \
            cv2.resize(image, dsize=None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LINEAR_EXACT)

        registration_image = rescale(full_image, registration_scale)

        features = cv2.detail.computeImageFeatures2(detector,
                                                    registration_image)

        seam_image = rescale(full_image, seam_scale)

        return features, seam_image

    @log
    def _match_features(self, matcher=None, match_confidence=None,
                        try_cuda=False, range_width=2):
        ''' Matches features as specified.

        'matcher' can be 'homography' or 'affine' (default 'homography').
        'match_confidence' is a float specifying how the confidence required
            to consider a pair of features as a match. Defaults to 0.3 if
            using the 'orb' feature detector or 0.65 otherwise.
        'try_cuda' is a boolean specifying if CUDA optimisations are available
            and should be used. NOTE: tends not to work
        'range_width' specifies the matching range for a given image (ie how
            many subsequent frames should be compared to for matching) when
            using the 'homography' matcher ('affine' ignores this parameter).
            Defaults to 2 (matching only with the image after a frame), to
            minimise matching time and maximise success on video input. Set to
            -1 to match each image against all others.

        '''
        matcher = matcher or self.default(self.matcher_options)
        if match_confidence is None:
            match_confidence = 0.3 if self._feature_detector == 'orb' else 0.65
        if matcher == 'affine':
            matcher = cv2.detail_AffineBestOf2NearestMatcher(
                False, try_cuda, match_confidence)
        elif range_width == -1:
            matcher = cv2.detail.BestOf2NearestMatcher_create(
                try_cuda, match_confidence)
        else:
            matcher = cv2.detail_BestOf2NearestRangeMatcher(
                range_width, try_cuda, match_confidence)

        self._matches = matcher.apply2(self._features)
        matcher.collectGarbage()

    # don't have image names -> can't save meaningful graph
    """
    @log
    def _save_graph(self, filename):
        ''' Save a graph of matches by image names and confidence_threshold. '''
        if save_graph:
            with open(filename, 'w') as graph:
                graph.write(cv2.detail.matchesGraphAsString(self._image_names,
                            self._matches, self._confidence_threshold))
    """

    @log
    def _remove_bad_images(self, confidence_threshold):
        ''' Removes images that don't seem to fit in the stitching.

        'confidence_threshold' is a float specifying the confidence required
            to keep an image in the stitching. TODO check precise meaning

        '''
        self._confidence_threshold = confidence_threshold
        indices = cv2.detail.leaveBiggestComponent(
                self._features, self._matches, self._confidence_threshold)

        remaining_images = []
        remaining_seam_images = []
        for index_ in indices:
            index = index_[0]
            remaining_images.append(self._images[index])
            remaining_seam_images.append(self._seam_images[index])

        self._images = remaining_images
        self._seam_images = remaining_seam_images

        self._num_images = len(self._images)
        if self._num_images < 2:
            raise ValueError("Insufficient matched images - Need more images.")

    @log
    def _estimate_camera_parameters(self, estimator):
        ''' Provides rough estimation of camera parameters.

        'estimator' specifies which estimator to use, as a key from
            self.estimator_options. Defaults to 'homography'.

        '''
        # TODO possibly replace with something that estimates translation
        #   AND rotation instead of just rotation
        estimator = estimator or self.default(self.estimator_options)
        estimator = self.estimator_options[estimator]()

        successful, self._cameras = estimator.apply(self._features,
                                                    self._matches, None)
        if not successful:
            raise ValueError("Homography estimation failed.")

        for cam in self._cameras:
            cam.R = cam.R.astype(np.float32)

    @log
    def _bundle_adjustment(self, method=None, refine_params=dict()):
        ''' Perform bundle-adjustment of camera parameters.

        'method' is a key from self.bundle_adjustment_options. Defaults to
            'ray'.
        'refine_params' is a dictionary specifying which camera parameters
            should be refined. Defaults to True for all of 'fx', 'skew',
            'ppx', 'aspect', and 'ppy' (explicitly set to False to skip).

        '''
        # TODO replace estimation with something that estimates translation
        #   AND rotation instead of just rotation
        method = method or self.default(self.bundle_adjustment_options)
        refine_params = {
            **{parameter: True for parameter in
               ('fx', 'skew', 'ppx', 'aspect', 'ppy')},
            **refine_params,
        }
        bundle_adjuster = self.bundle_adjustment_options[method]()
        bundle_adjuster.setConfThresh(self._confidence_threshold)

        refinement_mask = np.zeros((3, 3), np.uint8)
        for index, param in enumerate(('fx','skew','ppx','NO','aspect','ppy')):
            if refine_params.get(param, False):
                refinement_mask[divmod(index, 3)] = 1
        bundle_adjuster.setRefinementMask(refinement_mask)

        successful, self._cameras = \
            bundle_adjuster.apply(self._features, self._matches, self._cameras)

        if not successful:
            raise ValueError("Camera parameter adjustment failed.")

    @log
    def _wave_correction(self, wave_correction):
        ''' Tries to make the panorama more horizontal or vertical.

        'wave_correction' specifies which way to attempt to correct to, as a
            string from self.wave_correction_options. Defaults to 'vertical'.

        '''
        wave_correction = wave_correction or \
            self.default(self.wave_correction_options)
        wave_correction = self.wave_correction_options[wave_correction]

        if wave_correction is None:
            return
        rmats = [np.copy(cam.R) for cam in self._cameras]
        rmats = cv2.detail.waveCorrect(rmats, wave_correction)
        for index, cam in enumerate(self._cameras):
            cam.R = rmats[index]

    @log
    def _exposure_compensation(self, exposure_compensation, warp_type):
        ''' Defines an exposure compensator over the stitching images.

        'exposure_compensation' is a dictionary specifying 'type', 'feeds', and
            'block_size' (as appropriate). Detailed specification is provided
            in self._set_exposure_compensator.

        'warp_type' is the type of warping to apply to each image within the
            stitching. It should be one of self.warp_options, and defaults to
            'spherical'.

        '''
        self._set_exposure_compensator(exposure_compensation)
        self._warp_type = warp_type or self.default(self.warp_options)

        focals = sorted(cam.focal for cam in self._cameras)
        middle = len(focals) // 2
        self._warped_image_scale = focals[middle] if len(focals) % 2 == 1 \
                else (focals[middle] + focals[middle-1]) / 2

        warped_images        = []
        self._warped_masks   = []
        self._warped_corners = []
        # NOTE assumes images are the same size
        base_mask = cv2.UMat(255 * np.ones(self._seam_images[0].shape[:2],
                                           np.uint8))
        warper = cv2.PyRotationWarper(self._warp_type,
                self._warped_image_scale * self._seam_registration_aspect)
        warp = lambda image, K, R, border : \
                warper.warp(image, K, R, cv2.INTER_LINEAR, border)

        for index, seam_image in enumerate(self._seam_images):
            cam = self._cameras[index]
            K = cam.K().astype(np.float32)
            K[[0,0,1,1],[0,2,1,2]] *= self._seam_registration_aspect
            warp2 = lambda image, border : warp(image, K, cam.R, border)
            image_corners, warped_image = warp2(seam_image, cv2.BORDER_REFLECT)
            warped_images.append(warped_image)
            self._warped_corners.append(image_corners)
            _, warped_mask = warp2(base_mask, cv2.BORDER_CONSTANT)
            self._warped_masks.append(warped_mask)

        # store for seam finding
        self._warped_images = [img.astype(np.float32) for img in warped_images]

        self._compensator.feed(corners=self._warped_corners,
                               images=warped_images, masks=self._warped_masks)

    def _set_exposure_compensator(self, exposure_compensation):
        ''' Sets the exposure compensator.

        'exposure_compensation' is a dictionary specifying 'type', 'feeds', and
            'block_size' (as appropriate). 'type' should be one of
            self.exposure_compensation_options, and defaults to 'gain_blocks'.
            'feeds' is used for 'channels' type compensation, and both 'feeds'
            and 'block_size' are used for 'channel_blocks' compensation. The
            defaults are 2 for 'feeds' and 32 for 'channel_blocks'.

        '''
        compensation = {
            'type'       : self.default(self.exposure_compensation_options),
            'feeds'      : 2,
            'block_size' : 32,
            **exposure_compensation
        }

        if compensation['type'] == 'channels':
            self._compensator = cv2.detail_ChannelsCompensator(
                compensation['feeds']
            )
        elif compensation['type'] == 'channel_blocks':
            self._compensator = cv2.detail_BlocksChannelsCompensator(
                compensation['block_size'], compensation['block_size'],
                compensation['feeds']
            )
        else:
            self._compensator = cv2.detail.ExposureCompensator_createDefault(
                self.exposure_compensation_options[compensation['type']])

    @log
    def _find_seams(self, seam_finder):
        ''' Find seams in the image for seam blending.

        'seam_finder' should be a key from self.seam_find_options.
            Defaults to 'gc_color'.

        '''
        seam_finder = seam_finder or self.default(self.seam_find_options)
        seam_finder = self.seam_find_options[seam_finder]
        seam_finder.find(self._warped_images, self._warped_corners,
                         self._warped_masks)

    @log
    def _compose(self, timelapse, blend_type, blend_strength, result_name):
        ''' Compose the images into a single stitching.

        'timelapse' is one of self.timelapse_options. Defaults to 'no' for a
            stitched and blended result instead of saving individual warped
            images.
        'blend_type' is the type of blending applied between overlapping
            images, from self.blending_options. Defaults to 'multiband'.
        'blend_strength' is a number specifying how strongly to apply blending.
        'result_name' is the filename of the generated output image. Ignored
            if 'timelapse' is not 'no'.

        '''
        sizes              = []
        corners            = []
        blender            = None
        timelapser         = None
        compose_scale      = 1
        compose_megapixels = self._megapixels['composition']

        # NOTE assumes images are the same size
        #   -> for different sized images, need to calculate for each size
        image_size = self.get_size(self._images[0])
        if compose_megapixels > 0:
            compose_scale = min(1.0, np.sqrt(compose_megapixels))
        composed_size = tuple(np.array(image_size) * compose_scale)
        compose_registration_aspect = compose_scale / self._registration_scale
        self._warped_image_scale *= compose_registration_aspect
        warper = cv2.PyRotationWarper(self._warp_type, self._warped_image_scale)
        warp = lambda img, K, R, border : \
                warper.warp(img, K, R, cv2.INTER_LINEAR, border)[1]
        if abs(compose_scale - 1) > 1e-1:
            rescale = lambda image: cv2.resize(image, dsize=None,
                    fx=compose_scale, fy=compose_scale,
                    interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            rescale = lambda image: image
        base_mask = 255 * np.ones(rescale(self._images[0]).shape[:2], np.uint8)

        for cam in self._cameras:
            cam.focal *= compose_registration_aspect
            cam.ppx   *= compose_registration_aspect
            cam.ppy   *= compose_registration_aspect
            K = cam.K().astype(np.float32)
            roi = warper.warpRoi(composed_size, K, cam.R)
            corners.append(roi[0:2])
            sizes.append(roi[2:4])

        timelapse = timelapse or self.default(self.timelapse_options)
        if timelapse != 'no':
            timelapser = self._create_timelapser(timelapse, corners, sizes)
        else: # blending
            blender = self._create_blender(blend_type, blend_strength,
                                           corners, sizes)

        for index, image in enumerate(self._images):
            cam = self._cameras[index]
            image = rescale(image)
            warp2 = lambda image, border : \
                warp(image, cam.K().astype(np.float32), cam.R, border)

            warped_image = warp2(image, cv2.BORDER_REFLECT)
            warped_mask  = warp2(base_mask, cv2.BORDER_CONSTANT)

            self._compensator.apply(index, corners[index],
                                    warped_image, warped_mask)

            warped_image = warped_image.astype(np.int16)
            dilated_mask = cv2.dilate(self._warped_masks[index], None)
            seam_mask = cv2.resize(dilated_mask, self.get_size(warped_mask),
                                   0, 0, cv2.INTER_LINEAR_EXACT)
            warped_mask = cv2.bitwise_and(seam_mask, warped_mask)

            if True: # timelapse == 'no':
                blender.feed(cv2.UMat(warped_image), warped_mask, corners[index])
            else: # timelapse (not possible without filenames)
                """
                ma_tones = np.ones(warped_image.shape[:2], np.uint8)
                timelapser.process(warped_image, ma_tones, corners[index])
                image_path = image_names[index]
                name_position = image_path.rfind('/') + 1
                if name_position == 0: # no preceding directories
                    fixed_file_name = 'fixed_' + image_path
                else:
                    fixed_file_name = image_path[:name_position] + 'fixed_'\
                                     + image_path[name_position:]
                cv2.imwrite(fixed_file_name, timelapser.getDst())
                """
                pass

        if timelapse == 'no':
            result, result_mask = blender.blend(None, None)
            cv2.imwrite(result_name, result)
            zoom_x = 600.0 / result.shape[1]
            dst = cv2.normalize(src=result, dst=None, alpha=255.,
                                norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            dst = cv2.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
            cv2.imshow(result_name, dst)
            cv2.waitKey(1)
            return True

    def _create_timelapser(self, timelapse, corners, sizes):
        ''' Creates, initialises, and returns a timelapser. '''
        timelapser = cv2.detail.Timelapser_createDefault(
                self.timelapse_options[timelapse])

        timelapser.initialise(corners, sizes)
        return timelapser

    def _create_blender(self, blend_type, blend_strength, corners, sizes):
        ''' Creates, initialises, and returns a blender. '''
        blend_type = blend_type or self.default(self.blend_options)
        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)

        dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100

        if blend_width < 1: pass # not helpful to blend, so don't
        elif blend_type == 'multiband':
            blender = cv2.detail_MultiBandBlender()
            num_bands = (np.log(blend_width) / np.log(2.) - 1.).astype(np.int64)
            blender.setNumBands(num_bands)
        elif blend_type == 'feather':
            blender = cv2.detail_FeatherBlender()
            blender.setSharpness(1. / blend_width)

        blender.prepare(dst_sz)
        return blender

    @staticmethod
    def default(options):
        ''' Efficiently retrieves the first key from a dictionary or first
            value from a list/tuple.

        NOTE Assumes >= Python 3.7, from which dictionaries were defined as
            maintaining insertion order.

        '''
        return next(iter(options))

    @staticmethod
    def get_size(image):
        ''' Returns the width and height of 'image' as a tuple (w,h).

        Note the difference to default array.shape ordering, which outputs
            (h,w,d).

        '''
        return image.shape[:2][::-1]

    @staticmethod
    @log
    def images_from_video(filename, start=None, end=None, k=30, save_path=''):
        ''' Get the sharpest of every k frames between start and end times. '''
        images = []
        with VideoReader(filename, start=start, end=end, auto_delay=False) \
                as vid:
            for index, (status, img) in enumerate(vid):
                state = index % k
                if state == 0:
                    sharpest = None
                    max_sharpness = 0
                img_sharpness = sharpness(img)
                if img_sharpness > max_sharpness:
                    sharpest = img
                    max_sharpness = img_sharpness
                    if save_path:
                        frame = vid._frame
                if state == k-1:
                    images.append(img)
                    if save_path:
                        cv2.imwrite(f'{save_path}/{frame}.png', img)
        return images

if __name__ == '__main__':
    path = 'home2'#'images'
    #images = Stitcher.images_from_video('2020-07-15_16.56.54.mp4',
    #        start='1:50', end='2:22', k=30, save_path=path)
    images = [cv2.imread(f'{path}/{img}.png') for img in range(7906,7919)]#(50,70)]
    #import os
    #images = sorted([int(image[:-4]) for image in os.listdir(path)
    #                 if image.endswith('.png')])
    #images = [f'{path}/{img}.png' for img in images[7:12]]
    from pcv.process import downsize
    images = [downsize(img, 4) for img in images]

    Stitcher(images)
