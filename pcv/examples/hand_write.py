import cv2
import numpy as np
from pcv.vidIO import LockedCamera
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandWriter:
    def __init__(self, **kwargs):
        self._hands = mp_hands.Hands(**kwargs)
        self._drawing = None
        self._points = []

    def __enter__(self):
        self._hands.__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        return self._hands.__exit__(*args, **kwargs)

    def __call__(self, frame):
        flipped = cv2.flip(frame, 1)
        if self._drawing is None:
            self._drawing = np.zeros(flipped.shape[:2], np.uint8)
        image = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self._hands.process(image)
        height, width = image.shape[:2]
        num_points = len(self._points)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                if handedness.classification[0].label != 'Right':
                    continue # only interested in right hands

                tips = {}
                pips = {}
                dips = {}
                bases = {}
                INDEX = 'index_finger'
                MIDDLE = 'middle_finger'
                for finger in (INDEX, MIDDLE):
                    FINGER = finger.upper()
                    for PART, d in (('TIP', tips), ('PIP', pips),
                                    ('DIP', dips), ('MCP', bases)):
                        segment = hand_landmarks.landmark[
                            getattr(mp_hands.HandLandmark,
                                    f'{FINGER}_{PART}')]
                        d[finger] = self.land2coord(segment, width, height)

                index_angle = self.finger_angle(INDEX, tips, pips, dips, bases)

                # only draw if index finger is open, and middle finger is
                #  closed (assumes ring finger and pinky also closed)
                if index_angle < 0.3:
                    middle_angle = self.finger_angle(MIDDLE, tips, pips, dips,
                                                     bases)
                    if middle_angle > 0.5:
                        self._points.append(np.int32(tips[INDEX][:2]))

                mp_drawing.draw_landmarks(flipped, hand_landmarks,
                                          mp_hands.HAND_CONNECTIONS)
                break # only allow first detected right hand to draw

        # if there are no new points, or two points
        if (latest := len(self._points)) in (num_points, 2):
            if latest == 2:
                # draw the latest line on the drawing image
                # (means only need to keep track of one line at a time)
                self._drawing = cv2.line(self._drawing, self._points[0],
                                         self._points[1], 255, 3)
            if latest != 0:
                self._points.pop(0) # remove oldest point

        # saturate red channel at drawn locations
        flipped[:,:,2] |= self._drawing

        return flipped

    @staticmethod
    def land2coord(landmark, width, height):
        return np.array([landmark.x * width, landmark.y * height,
                         landmark.z * width])

    @staticmethod
    def finger_angle(segment, tips, pips, dips, bases):
        segment_first = tips[segment] - pips[segment]
        segment_base = dips[segment] - bases[segment]
        usf = segment_first / np.linalg.norm(segment_first)
        usb = segment_base / np.linalg.norm(segment_base)
        return np.arccos(np.clip(np.dot(usf, usb), -1.0, 1.0))


def main(filename, **kwargs):
    defaults = dict(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                    max_num_hands=2)
    defaults.update(kwargs)

    with HandWriter(**defaults) as hand_writer, \
                    LockedCamera(0, process=hand_writer) as cam:
        cam.record_stream(filename)


if __name__ == '__main__':
    import sys
    filename = 'handwriting.mp4' if len(sys.argv) == 1 else sys.argv[1]
    main(filename)
