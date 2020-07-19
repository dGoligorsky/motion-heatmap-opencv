import numpy as np
import cv2
import copy
from make_video import make_video
from progress.bar import Bar


def main():
    capture = cv2.VideoCapture('C1_90Min.m4v')
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    bar = Bar('Processing Frames', max=length)

    first_iteration_indicator = 1
    for i in range(0, length):

        ret, frame = capture.read()

        # If first frame
        if first_iteration_indicator == 1:

            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:

            filter = background_subtractor.apply(
                frame)  # remove the background
            cv2.imwrite('./frame.jpg', frame)
            cv2.imwrite('./diff-bkgnd-frame.jpg', filter)

            threshold = 140
            maxValue = 255

            # adaptive thresholding... you can remove this if you want. not sure it's an improvement
            th1 = cv2.adaptiveThreshold(filter, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 3, 2)

            ret, th1 = cv2.threshold(
                filter, threshold, maxValue, cv2.THRESH_TRUNC)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1)
            cv2.imwrite('./mask.jpg', accum_image)

            color_image_video = cv2.applyColorMap(
                accum_image, cv2.COLORMAP_HOT)

            video_frame = cv2.addWeighted(
                frame, 0.2, color_image_video, 0.8, 0)

            name = "./frames/frame%d.jpg" % i
            cv2.imwrite(name, video_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        bar.next()

    bar.finish()

    # make_video('./frames/', './output.avi')
    make_video('./frames/', './output.mp4')

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.2, color_image, 0.8, 0)

    # save the final heatmap
    cv2.imwrite('diff-overlay.jpg', result_overlay)

    # cleanup
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
