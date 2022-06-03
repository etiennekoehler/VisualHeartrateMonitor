class Pipeline():
    """ 
    This class runs the pyVHR pipeline on a single video or dataset
    """

    def __init__(self):
        pass

    def run_on_video(self, videoFileName, cuda=False, roi_method='convexhull', roi_approach='holistic',
                     method='cupy_POS', bpm_type='welch', pre_filt=False, post_filt=True, verb=True):
        """ 
        Runs the pipeline on a specific video file.

        Args:
            videoFileName:
                - The path to the video file to analyse
            cuda:
                - True - Enable computations on GPU
                - False - Use CPU only
            roi_method:
                - 'convexhull' - Uses MediaPipe's lanmarks to compute the convex hull of the face and segment the skin
                - 'faceparsing' - Uses BiseNet to parse face components and segment the skin
            roi_approach:
                - 'holistic' - Use the Holistic approach (one single ROI defined as the whole face skin region of the subject)
                - 'patches' - Use multiple patches as Regions of Interest
            method:
                - One of the rPPG methods defined in pyVHR
            bpm_type:
                - the method for computing the BPM estimate on a time window
            pre_filt:
                - True - Use Band pass filtering on the windowed RGB signal
                - False - No pre-filtering
            post_filt:
                - True - Use Band pass filtering on the estimated BVP signal
                - False - No post-filtering
            verb:
               - False - not verbose
               - True - show the main steps  
        """
        ldmks_list = [2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54, 58, 67, 68, 69, 71, 92, 93,
                      101, 103, 104, 108, 109, 116, 117, 118, 123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151,
                      152, 182, 187, 188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251, 262, 265,
                      266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330, 332, 333, 337, 338, 345, 346, 361,
                      363, 364, 367, 368, 371, 377, 379, 411, 412, 417, 421, 425, 426, 427, 430, 432, 436]
        assert os.path.isfile(videoFileName), "\nThe provided video file does not exists!"

        sig_processing = SignalProcessing()
        av_meths = getmembers(pyVHR.BVP.methods, isfunction)
        available_methods = [am[0] for am in av_meths]

        assert method in available_methods, "\nrPPG method not recognized!!"

        if cuda:
            sig_processing.display_cuda_device()
            sig_processing.choose_cuda_device(0)

        # set skin extractor
        target_device = 'GPU' if cuda else 'CPU'
        if roi_method == 'convexhull':
            sig_processing.set_skin_extractor(
                SkinExtractionConvexHull(target_device))
        elif roi_method == 'faceparsing':
            sig_processing.set_skin_extractor(
                SkinExtractionFaceParsing(target_device))
        else:
            raise ValueError("Unknown 'roi_method'")

        assert roi_approach == 'patches' or roi_approach == 'holistic', "\nROI extraction approach not recognized!"

        # set patches
        if roi_approach == 'patches':
            # ldmks_list = ast.literal_eval(landmarks_list)
            # if len(ldmks_list) > 0:
            sig_processing.set_landmarks(ldmks_list)
            # set squares patches side dimension
            sig_processing.set_square_patches_side(28.0)

        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = 75
        SignalProcessingParams.RGB_HIGH_TH = 230
        SkinProcessingParams.RGB_LOW_TH = 75
        SkinProcessingParams.RGB_HIGH_TH = 230

        if verb:
            print('\nProcessing Video: ' + videoFileName)
        fps = get_fps(videoFileName)
        sig_processing.set_total_frames(0)

        # -- ROI selection
        sig = []
        if roi_approach == 'holistic':
            # SIG extraction with holistic
            sig = sig_processing.extract_holistic(videoFileName)
        elif roi_approach == 'patches':
            # SIG extraction with patches
            sig = sig_processing.extract_patches(videoFileName, 'squares', 'mean')

        # -- sig windowing
        windowed_sig, timesES = sig_windowing(sig, 6, 1, fps)

        # -- PRE FILTERING
        filtered_windowed_sig = windowed_sig

        # -- color threshold - applied only with patches
        if roi_approach == 'patches':
            filtered_windowed_sig = apply_filter(windowed_sig,
                                                 rgb_filter_th,
                                                 params={'RGB_LOW_TH': 75,
                                                         'RGB_HIGH_TH': 230})

        if pre_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps = apply_filter(filtered_windowed_sig,
                                method_to_call,
                                fps=fps,
                                params={'minHz': 0.65, 'maxHz': 4.0, 'fps': 'adaptive', 'order': 6})
        if verb:
            print("\nBVP extraction with method: %s" % (method))

        # -- BVP Extraction
        module = import_module('pyVHR.BVP.methods')
        method_to_call = getattr(module, method)
        if 'cpu' in method:
            method_device = 'cpu'
        elif 'torch' in method:
            method_device = 'torch'
        elif 'cupy' in method:
            method_device = 'cuda'

        if 'POS' in method:
            pars = {'fps': 'adaptive'}
        elif 'PCA' in method or 'ICA' in method:
            pars = {'component': 'all_comp'}
        else:
            pars = {}

        bvps = RGB_sig_to_BVP(windowed_sig, fps,
                              device_type=method_device, method=method_to_call, params=pars)

        # -- POST FILTERING
        if post_filt:
            module = import_module('pyVHR.BVP.filters')
            method_to_call = getattr(module, 'BPfilter')
            bvps = apply_filter(bvps,
                                method_to_call,
                                fps=fps,
                                params={'minHz': 0.65, 'maxHz': 4.0, 'fps': 'adaptive', 'order': 6})

        if verb:
            print("\nBPM estimation with: %s" % (bpm_type))
        # -- BPM Estimation
        if bpm_type == 'welch':
            if cuda:
                bpmES = BVP_to_BPM_cuda(bvps, fps, minHz=0.65, maxHz=4.0)
            else:
                bpmES = BVP_to_BPM(bvps, fps, minHz=0.65, maxHz=4.0)
        elif bpm_type == 'clustering':
            if cuda:
                bpmES = BVP_to_BPM_PSD_clustering_cuda(bvps, fps, minHz=0.65, maxHz=4.0)
            else:
                bpmES = BVP_to_BPM_PSD_clustering(bvps, fps, minHz=0.65, maxHz=4.0)
        else:
            raise ValueError("Unknown 'bpm_type'")

        # median BPM from multiple estimators BPM
        median_bpmES, mad_bpmES = multi_est_BPM_median(bpmES)

        if verb:
            print('\n...done!\n')

        return timesES, median_bpmES, mad_bpmES

