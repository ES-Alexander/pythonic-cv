#!/usr/bin/env python3

import numpy as np
import cv2
from os import mkdir
from os.path import isfile, isdir
from pcv.vidIO import LockedCamera
from pcv.interact import MouseCallback


class Fish:
    ''' A fish that floats, and can be tickled and/or caught. '''
    def __init__(self, name, position, velocity, tank_dims, depth):
        ''' Create a fish with specified attributes.

        'name' should be a string, and determines the internal properties
            of the fish, such as size and colour.
        'position' and 'velocity' should be complex numbers with the real
            axis pointing down the image, and the imaginary axis pointing
            across to the right. The origin is the top left corner.
              This helps facilitate easy flipping at the sides (complex
            conjugate), and rotation at the top/bottom.
        'tank_dims' is a tuple of the dimensions of the tank, as
            (height, width) coordinates.
        'depth' is this fish's location relative to the viewer/other fish.
            It should generally be between 1 and 50, and will likely raise
            an Exception if below -5 or above 95 (it is used to determine
            water cover).

        '''
        self.name      = name
        self.position  = position
        self.velocity  = velocity
        self.x_min     = self.y_min = 0
        self.y_max, self.x_max = tank_dims
        self.depth     = depth
        self.is_caught = 0
        self._changed  = [False, False]
        self.determine_appearance()
        self.update_angle()

    @property
    def position_tuple(self):
        ''' The x, y position coordinates. '''
        return self.position.imag, self.position.real

    @property
    def velocity_tuple(self):
        ''' The x, y velocity coordinates. '''
        return self.velocity.imag, self.velocity.real

    @property
    def bbox(self):
        ''' A rough bounding box of the fish. '''
        long = self.axes[0]
        return np.array(list(self.position_tuple)*2).reshape(2,2) \
              - np.array([[long], [-long]])

    def determine_appearance(self):
        ''' Use self.name to determine size and colour. '''
        # convert name to (0,1] numbers, use only lowercase for good range
        min_char, max_char = ord('a')-1, ord('z')
        range_ = max_char - min_char
        numbers = [(ord(c.lower())-min_char)/range_
                   for c in self.name if c not in " -'"]
        self.colours = np.array([255*np.array([numbers[i], numbers[-i%3]])
                                 for i in range(-3,0)])
        # add blue for depth (at least 5% behind water)
        alpha = self.depth / 100 + 0.05
        self.colours = self.add_water(self.colours, alpha).T

        # determine size and shape
        self.size = 3*(sum(numbers) + len(numbers)) / (sum(numbers[:2])/2)
        self.axes = (self.size / (numbers[1]+2),
                     self.size / (numbers[2]+3.2))

        # eye properties
        self._eye_offset = np.array([3*self.axes[0]/5, self.axes[1]/4])
        self._eye_size = int(self.axes[0] / 7)
        self._pupil_size = int(self.axes[1] / 8)

        # fin properties
        self._fin_points = np.array([[0,0],[-1,numbers[2]],[-1,-numbers[1]]]) \
            * self.axes[1]

        # tail properties
        self._tail_points = np.array(
            [[-self._eye_offset[0], 0],
              -self._eye_offset * [2, 3],
              self._eye_offset * [-(1.2 + 2 * numbers[0] / 3), 1]] * 2)
        self._tail_points = [self._tail_points[:3], self._tail_points[3:]]
        self._tail_points[1][1,1] *= -1
        self._tail_points[1][2,1] *= -1

    @staticmethod
    def add_water(colour, alpha):
        ''' Add (100*alpha)% water cover to the specified colour. '''
        beta = 1 - alpha
        colour *= beta
        colour[0] += 200 * alpha
        colour[1] += 40 * alpha
        return colour

    def update_angle(self):
        ''' Compute the updated angle and rotation based on the velocity. '''
        angle = np.arctan2(*self.velocity_tuple[::-1])
        self._rotate_fin = np.array([np.cos(angle), np.sin(angle)])
        self._rotate = np.array([self._rotate_fin,
                                 (self._rotate_fin * [1,-1])[::-1]])
        self.angle = np.degrees(angle)

    def draw(self, img, acceleration):
        ''' Draw self and update state according to acceleration.

        If self.is_caught, draws a fading ellipse for 50 frames before
            skipping drawing altogether.

        'img' is the image to draw onto, and should be an 8-bit colour image.
        'acceleration' should be a single-channel image with the same width
            and height as 'img', mapping position to acceleration.

        '''
        if self.is_caught < 50:
            colour = self.colours[0].copy()
            thickness = -1
            if self.is_caught:
                alpha = self.is_caught / 50
                colour = tuple(self.add_water(colour, alpha))
                pos = tuple(int(dim) for dim in self.position_tuple)
                self.is_caught += 1
                thickness = 1

            cv2.ellipse(img, (self.position_tuple,
                              tuple(2*dim for dim in self.axes),
                              self.angle), colour, thickness)

            if not self.is_caught:
                for draw_func in (self.draw_eye, self.draw_tail, self.draw_fin):
                    draw_func(img)

                self.update_state(acceleration)

    def draw_eye(self, img):
        ''' Draw eye on 'img'. '''
        eye_offset = self._eye_offset * [1, -np.sign(self.velocity.imag)]
        eye_offset = eye_offset @ self._rotate
        pupil_offset = eye_offset * 1.05
        eye_pos = tuple(int(dim) for dim in eye_offset + self.position_tuple)
        pupil_pos = tuple(int(dim) for dim in
                          pupil_offset + self.position_tuple)
        for pos, size, colour in [[eye_pos, self._eye_size, (244,212,204)],
                                  [pupil_pos, self._pupil_size, (40,8,0)]]:
            cv2.circle(img, pos, size, colour, -1)

    def draw_tail(self, img):
        ''' Draw tail on 'img'. '''
        colour = tuple(int(channel) for channel in self.colours[1])
        for half in self._tail_points:
            half_points = half @ self._rotate + self.position_tuple
            cv2.fillConvexPoly(img, np.int32(half_points), colour)

    def draw_fin(self, img):
        ''' Draw fin on 'img'. '''
        colour = tuple(int(channel) for channel in self.colours[1])
        fin_points = self._rotate_fin * self._fin_points + self.position_tuple
        cv2.fillConvexPoly(img, np.int32(fin_points), colour)

    def update_state(self, acceleration):
        ''' Update the fish position/velocity. '''
        # update position
        self.position += self.velocity

        # update velocity
        self.velocity *= 0.995 # add a touch of damping to avoid crazy speeds

        x,y = np.int32(np.round(self.position_tuple))
        x_accel, y_accel = acceleration
        try:
            self.velocity += (x_accel[y,x] + 1j * y_accel[y,x])
        except IndexError:
            pass # outside the tank 

        # update relevant velocity component if outside the tank
        for index, (min_, max_) in enumerate([[self.x_min, self.x_max],
                                              [self.y_min, self.y_max]]):
            val = self.position_tuple[index]
            left = val < min_
            if (left or val > max_):
                if not self._changed[index]:
                    if index == 0: # mirror if hitting side
                        self.velocity = self.velocity.conjugate()
                    else: # rotate 90 degrees if hitting top/bottom
                        direc = -2 * (left - 0.5)
                        self.velocity *= \
                            direc * np.sign(self.velocity.imag) * 1j
                    self._changed[index] = True
            elif self._changed[index]:
                # back in the tank
                self._changed[index] = False

        self.update_angle()

    def catch(self):
        self.is_caught = 1
        print(f'Caught {self}!')

    def __str__(self):
        return f'{self.name}: {self.size/100:.3f}kg'


class FishTank:
    ''' A tank for storing and tracking fish. '''
    def __init__(self, tank_dims, max_fish=40, name_file='names.txt'):
        ''' Create a new fish tank as specified.

        'tank_dims' should be a tuple of (height, width) pixels
        'max_fish' is the largest number of fish that can be generated on
            initialisation (random number generated between 1 and max_fish).
            Defaults to 40.
        'name_file' is the filename of a newline separated file containing
            possible names for the fish in the tank. Defaults to 'names.txt'
            which on distribution contains a few hundred popular names from
            around the world. Feel free to change it to a set of names of your
            family and friends!

        '''
        # store/initialise input parameters
        self.dims = np.int32(tank_dims)
        self.max_fish = max_fish
        with open(name_file) as names:
            self.names = [name.strip() for name in names]
        
        self._initialise_fish()
        self._initialise_stats()
        self._initialise_instructions()
        self._setup_mouse_control()

    def _initialise_fish(self):
        ''' initialise fish, including storage and trackers '''
        # create a fast random number generator
        self.rng = np.random.default_rng()

        self.caught_fish = []
        self.fish = []
        self._num_fish = self.rng.integers(2, self.max_fish)
        for i in range(self._num_fish):
            self.fish.append(self.random_fish(i))

    def random_fish(self, depth):
        ''' Create a random fish instance. '''
        return Fish(self.random_name(), self.random_position(),
                    self.random_velocity(), self.dims, depth)

    def random_name(self):
        ''' Randomly choose a name for a fish. '''
        return self.names[self.rng.integers(len(self.names))]

    def random_position(self):
        ''' Determine a valid random position for a fish. '''
        offset = 40
        return complex(self.rng.integers(offset, self.dims[0]-offset),
                       self.rng.integers(offset, self.dims[1]-offset))

    def random_velocity(self):
        ''' Create a random initial velocity for a fish. '''
        max_x = self.dims[1] // 100
        max_y = self.dims[0] // 100
        return complex(self.rng.integers(-max_y, max_y),
                       self.rng.integers(-max_x, max_x))

    def _initialise_stats(self):
        ''' Intialise stats and tracking parameters. '''
        self._prev = np.zeros(tuple(self.dims), dtype=np.int32)
        self._precision = 0
        self._gradient = False
        self._attempts = 0
        self._t = 1

    def _initialise_instructions(self):
        ''' Create some helpful instructions to display at the start. '''
        self._instructions_visible = True

        scale = 0.6
        thickness = 1
        height, width = self.dims
        font = cv2.FONT_HERSHEY_SIMPLEX
        self._instructions = np.zeros(tuple((height, width, 3)), dtype=np.uint8)
       
        instructions = (
            "Who lives in a pineapple with OpenCV?",
            '',
            "Press 'i' to toggle instructions on/off",
            "Press 'g' to toggle the image gradient used for acceleration",
            "Press 'q' to quit, and SPACE to pause/resume",
            '',
            "Catch fish by dragging your 'net' over them with the mouse",
            "(if your box is too big or small they'll escape).",
            "Caught fish will have their image, with name and size",
            "displayed in the 'gallery' folder.",
            '',
            "'Hit rate' is the percentage of attempts you've caught a fish in.",
            "'Avg size ratio' is the ratio of your box size over the fish size",
            "for each of your successful catches - smaller is more skillful.",
            '',
            "Some fish might escape the tank and not come back, and that's ok.",
        )

        # add instructions to an empty image, for merging later
        num_instructions = len(instructions)
        text_height = cv2.getTextSize(' ', font, scale, thickness)[0][1]
        spacing = 2 * text_height / 3
        tot_y = num_instructions * text_height + spacing * (num_instructions - 2)

        y_offset = (height - tot_y) // 2

        for index, line in enumerate(instructions):
            x,y = cv2.getTextSize(line, font, scale, thickness)[0]
            x_pos = int((width - x) / 2)
            y_pos = int(y_offset + (y + spacing) * index)
            cv2.putText(self._instructions, line, (x_pos, y_pos), font,
                        scale, (255,255,255), thickness)


    def _setup_mouse_control(self):
        ''' Specify mouse control functions. '''
        self._start_point = self._end_point = None
        self._catch_bindings = {
            cv2.EVENT_LBUTTONDOWN : self.start_catching,
            cv2.EVENT_MOUSEMOVE   : self.catch_to,
            cv2.EVENT_LBUTTONUP   : self.end_catch,
        }

    def mouse_handler(self, event, *args):
        self._catch_bindings.get(event, lambda *args: None)(*args)

    def start_catching(self, x, y, *args):
        ''' Start the creation of a net for catching fish. '''
        self._start_point = self._end_point = x,y

    def catch_to(self, x, y, *args):
        ''' Draw the net live as it resizes. '''
        if self._start_point:
            self._end_point = x,y

    def end_catch(self, x, y, *args):
        ''' Register a catch attempt and check for catches. '''
        self._catch_img = self._img
        self._catch_fish()
        self._start_point = self._end_point = None
        self._attempts += 1

    def _catch_fish(self):
        ''' Check if any fish were caught in the last attempt. '''
        # get current fish bounding boxes
        try:
            min_pts, max_pts = self._fish_bboxes
        except ValueError:
            return # no more fish to catch
        min_pts = min_pts.reshape(-1,2)
        max_pts = max_pts.reshape(-1,2)

        min_pt, max_pt = self._get_net_extent()
        caught         = self._find_caught_fish(min_pts, max_pts,
                                                min_pt, max_pt)
        self._register_catches(min_pt, max_pt, caught)

    def _get_net_extent(self):
        ''' Returns the min_pt, max_pt of the net extent. '''
        pts = []
        for i in range(2):
            p1 = self._start_point[i]
            p2 = self._end_point[i]
            if p1 < p2:
                pts.append([p1,p2])
            else:
                pts.append([p2,p1])
        return np.array(pts).T

    def _find_caught_fish(self, min_pts, max_pts, min_pt, max_pt):
        ''' Returns an index array of caught fish. '''
        min_diff = min_pts - min_pt
        max_diff = max_pt - max_pts
        box_size = (max_pt - min_pt).sum()
        size_ratio = box_size / (max_pts - min_pts).sum(axis=1)
        caught, = np.nonzero((size_ratio < 4) &
            (((min_diff > 0) & (max_diff > 0)).sum(axis=1) == 2))
        self._precision += size_ratio[caught].sum()
        return caught

    def _register_catches(self, min_pt, max_pt, caught):
        ''' Register catches and track which fish are free. '''
        free_fish = []
        caught_fish = []
        for index, fish in enumerate(self.fish):
            if index in caught:
                caught_fish.append(fish)
                fish.catch()
            else:
                free_fish.append(fish)

        # save image for caught fish
        if len(caught):
            # create the gallery if it doesn't already exist
            if not isdir('gallery'):
                mkdir('gallery')

            # determine relevant image filename
            fish = '-'.join(f'{fish.name}_{fish.size/100:.3f}kg'
                            for fish in caught_fish)
            pre, extension = 'gallery/caught_', '.png'
            filename = pre + fish + extension

            # put a count at the end if the fish has already been caught
            count = 0
            while isfile(filename):
                count += 1
                filename = f'{pre}{fish}({count}){extension}'

            # ensure image is within frame
            min_pt[min_pt < 0] = 0
            max_pt[0] = min(max_pt[0], self.dims[1])
            max_pt[1] = min(max_pt[1], self.dims[0])

            # write to file
            cv2.imwrite(filename, self._catch_img[min_pt[1]:max_pt[1],
                                                  min_pt[0]:max_pt[0]])

        self.caught_fish.extend(caught_fish)
        self.fish = free_fish

    @property
    def _fish_bboxes(self):
        ''' Returns an array of the min_pts and max_pts of each fish. '''
        return np.c_[tuple(fish.bbox for fish in self.fish)]

    def toggle_gradient(self, vid=None):
        ''' Toggle gradient display mode on or off. '''
        self._gradient ^= True

    def toggle_instructions(self, vid=None):
        ''' Toggle the instructions display on or off. '''
        self._instructions_visible ^= True

    def preprocess(self, img):
        ''' Light preprocessing. '''
        self._t += 1
        max_accel = 30

        blur = cv2.GaussianBlur(img, (7,7), 0)
        flipped = cv2.flip(blur, 1) # mirror webcam
        grey = np.float32(cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)) / 255

        # calculate acceleration from difference to previous image
        diff = (grey - self._prev + 1) / 2
        x_accel = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=5)
        x_accel /= (1e-10+x_accel.max()-x_accel.min()) / max_accel
        y_accel = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=5)
        y_accel /= (1e-10+y_accel.max()-y_accel.min()) / max_accel
        self._acceleration = x_accel, y_accel
        self._prev = grey

        return flipped

    def __call__(self, flipped):
        ''' Main processing, while waiting for next image. '''
        if self._gradient:
            x_accel, y_accel = self._acceleration
            max_val = np.max([x_accel.max(), y_accel.max()])
            min_val = np.min([x_accel.min(), y_accel.min()])
            range_ = max_val - min_val
            x_norm = (x_accel - min_val) / range_
            y_norm = (y_accel - min_val) / range_
            gradient = cv2.addWeighted(x_norm, 0.5, y_norm, 0.5, 0.0)
            flipped = cv2.merge([np.uint8(255 * gradient)]*3)
        else:
            self._draw_water(flipped)

        self._draw_fish(flipped, self._acceleration)
        self._img = flipped
        self._text_overlay(flipped)
        self._draw_net(flipped)

        return flipped

    def _draw_water(self, img):
        # make some blue and green that varies a bit with time
        blue = np.zeros(img.shape, dtype=img.dtype)
        mag = 30 * np.sin(self._t/100)
        blue[:,:,0] = 200 + mag * np.sin(np.arange(img.shape[1])/(mag/6+50))
        blue[:,:,1] = 40 + mag * np.sin(np.arange(img.shape[0])[:,None] \
                                        / (mag/6+50))

        # blend with the background image
        alpha = 0.45
        beta = 1 - alpha
        cv2.addWeighted(img, alpha, blue, beta, 0.0, img)

    def _draw_fish(self, img, acceleration):
        ''' Draw in all the free and caught fish. '''
        for fish in self.fish:
            fish.draw(img, acceleration)
        for fish in self.caught_fish:
            fish.draw(img, None)

    def _text_overlay(self, img):
        ''' Show instructions or how many fish have been caught + stats. '''
        if self._instructions_visible:
            cv2.addWeighted(img, 0.3, self._instructions, 0.7, 0, img)
            return

        caught_fish = len(self.caught_fish)
        texts = [f'Caught {caught_fish}/{self._num_fish}']
        if self._attempts:
            texts.append(f'Hit rate: {100 * caught_fish / self._attempts:.2f}%')
        if caught_fish:
            texts.append(f'Avg size ratio: {self._precision / caught_fish:.3f}')

        for index, text in enumerate(texts):
            cv2.putText(img, text, (10, 20*(index+1)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 1)

    def _draw_net(self, img):
        ''' Draws the 'catch' net if one is in progress. '''
        if self._end_point:
            # make thicker lines for larger net
            thickness = 1 + \
                sum(abs(self._end_point[index] - self._start_point[index])
                    for index in range(2)) // 90
            cv2.rectangle(img, self._start_point, self._end_point, (0,0,100),
                          thickness)


tank = FishTank((720,1280), name_file='names.txt') 
window = 'Fish Tank'
with LockedCamera(0, preprocess=tank.preprocess, process=tank, display=window,
                  play_commands={ord('g'):tank.toggle_gradient,
                                 ord('i'):tank.toggle_instructions}) as cam:
    cam.record_stream('lol.mp4',
                      mouse_handler=MouseCallback(window, tank.mouse_handler))
