import cv2, utils_realtime, time
import numpy as np
from scipy import signal
from sig_processing import *
from skin_extraction_methods import *
from sig_extraction_methods import *
from utils import *
from filters import *

class Realtime_Pipeline():

    def get_mean_RGB_holistic(self, frame):

        sig_processing = SignalProcessing()

        sig_processing.set_skin_extractor(SkinExtractionFaceParsing('CPU'))

        # set sig-processing and skin-processing params
        SignalProcessingParams.RGB_LOW_TH = 75
        SignalProcessingParams.RGB_HIGH_TH = 230
        SkinProcessingParams.RGB_LOW_TH = 75
        SkinProcessingParams.RGB_HIGH_TH = 230

        sig_processing.set_total_frames(1)  # 0 means all frames are being processed

        # 0.3 s until this point
        # 0.15 s for the rest
        # -- ROI selection
        return sig_processing.extract_holistic_frame(frame)

    def run_realtime_pipeline(self, is_test, output, advanced_skin_extraction):

        # %% User Settings
        use_prerecorded = True
        fs = 30  # Sampling Frequency
        use_POS = False

        # %% Parameters
        # is_test: only executes 60 frames
        # output: 'mean_colors', 'skin_extraction_execution_times', 'bpm'
        haar_cascade_path = "/Users/etienne/opt/miniconda3/envs/pyvhr/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        tracker = cv2.legacy.TrackerMOSSE_create()
        cap = utils_realtime.RecordingReader() if use_prerecorded else cv2.VideoCapture(0)

        window = 300  # Number of samples to use for every measurement
        skin_vec = [0.3841, 0.5121, 0.7682]
        B, G, R = 0, 1, 2

        found_face = False
        initialized_tracker = False
        face_box = []
        mean_colors = []
        timestamps = []
        bpm_list = []
        mean_colors_resampled = np.zeros((3, 1))
        frames_not_found = 0

        # Testing
        skin_extraction_exection_times = []


        # %% Main loop
        while True:

            success, frame = cap.read()
            if not success:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Try to update face location using tracker
            if found_face and initialized_tracker:
                #print("Tracking")
                found_face, face_box = tracker.update(frame)
                #if not found_face:
                    #print("Lost Face")

            # Try to detect new face
            if not found_face:
                initialized_tracker = False
                #print("Redetecing")
                faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5)
                found_face = len(faces) > 0

            # Reset tracker
            if found_face and not initialized_tracker:
                face_box = faces[0]
                tracker = cv2.legacy.TrackerMOSSE_create()
                tracker.init(frame, tuple(face_box))
                initialized_tracker = True

            # Measure face color
            if found_face:
                face = utils_realtime.crop_to_boundingbox(face_box, frame)
                if face.shape[0] > 0 and face.shape[1] > 0:
                    if advanced_skin_extraction:
                        st = time.time()
                        mean_colors += [(self.get_mean_RGB_holistic(frame))[0]]
                        et = time.time()
                        elapsed_time = et - st
                        skin_extraction_exection_times.append(elapsed_time)
                    else:
                        st = time.time()
                        mean_colors += [face.mean(axis=0).mean(axis=0)] # 3-item-list with mean RGB values of current frame is added to mean_colors
                        et = time.time()
                        elapsed_time = et - st
                        skin_extraction_exection_times.append(elapsed_time)
                    timestamps += [time.time()] # time that frame was recorded in seconds
                    utils_realtime.draw_face_roi(face_box, frame)
                    t = np.arange(timestamps[0], timestamps[-1], 1 / fs) # get evenly spaced list of times, with length: num of frames
                    mean_colors_resampled = np.zeros((3, t.shape[0]))

                    for color in [B, G, R]:
                        resampled = np.interp(t, timestamps, np.array(mean_colors)[:, color]) # get list of mean colors that are evenly spaced temporally (with interpolation)
                        mean_colors_resampled[color] = resampled

            # Perform chrominance method
            if mean_colors_resampled.shape[1] > window:
                print('mean_colors_resampled_shape', mean_colors_resampled.shape[1])
                print('mean_colors_shape', len(mean_colors))
                if use_POS:
                    frames_not_found += 1
                    bpm = self.cpu_POS()
                    utils_realtime.put_snr_bpm_onframe(bpm, None, frame)

                else:
                    col_c = np.zeros((3, window))

                    for col in [B, G, R]:
                        col_stride = mean_colors_resampled[col, -window:]  # select last samples
                        y_ACDC = signal.detrend(col_stride / np.mean(col_stride))
                        col_c[col] = y_ACDC * skin_vec[col]

                    X_chrom = col_c[R] - col_c[G]
                    Y_chrom = col_c[R] + col_c[G] - 2 * col_c[B]
                    Xf = utils_realtime.bandpass_filter(X_chrom)
                    Yf = utils_realtime.bandpass_filter(Y_chrom)
                    Nx = np.std(Xf)
                    Ny = np.std(Yf)
                    alpha_CHROM = Nx / Ny

                    x_stride = Xf - alpha_CHROM * Yf
                    amplitude = np.abs(np.fft.fft(x_stride, window)[:int(window / 2 + 1)])
                    normalized_amplitude = amplitude / amplitude.max()  # Normalized Amplitude

                    frequencies = np.linspace(0, fs / 2, int(window / 2) + 1) * 60
                    bpm_index = np.argmax(normalized_amplitude)
                    bpm = frequencies[bpm_index]
                    print('bpm', bpm)
                    bpm_list.append(bpm)
                    snr = utils_realtime.calculateSNR(normalized_amplitude, bpm_index)
                    utils_realtime.put_snr_bpm_onframe(bpm, snr, frame)

            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if is_test and len(skin_extraction_exection_times) > 60:
                break

        cap.reset()
        #cap.release()
        #cv2.destroyAllWindows()

        if output == 'mean_colors':
            return mean_colors
        elif output == 'skin_extraction_execution_times':
            return skin_extraction_exection_times
        elif output == 'bpm':
            print('bpm_list', len(skin_extraction_exection_times))
            print('lost frames', frames_not_found)
            return bpm_list
        else:
            return

        # plt.figure()

if __name__ == '__main__':
    pipe = Realtime_Pipeline()
    pipe.run_realtime_pipeline(is_test=False, output='', advanced_skin_extraction=False)
