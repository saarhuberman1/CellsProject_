import numpy as np
import pandas as pd
import copy
import cv2


def open_window():
    cv2.namedWindow('', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.startWindowThread()


def close_window():
    cv2.destroyAllWindows()


class HIC_Annotator(object):
    def __init__(self, excel_file, root_dir):
        self.excel_file = excel_file
        self.root_dir = root_dir
        self.curr_im = None
        self.curr_brown_im = None
        self.display = None
        self.curr_im_path = None
        self.space_on = False
        self.curr_sheet = None
        self.curr_label = None
        self.exit = False
        self.sheets = []
        self.curr_idx = None
        self.done = 0
        self.curr_sheet_idx = 0
        self.images = []
        self.labels = []
        self.data = {}

    def get_initial_image(self):
        i = 0
        for label in self.labels:
            if not pd.isna(label):  # skip over filed labels
                i += 1
                continue
            return i
        return i

    def run(self, debug=False):
        xls = pd.ExcelFile(self.excel_file)
        self.sheets = [s for s in xls.sheet_names if s != 'summary']
        xls.close()
        for sheet in self.sheets:
            df = pd.read_excel(self.excel_file, sheet_name=sheet)
            self.data[sheet] = df
        self.curr_sheet = -1
        first_sheet = True
        while self.curr_sheet < len(self.sheets):
            self.save_sheet()
            if self.curr_idx is None or self.curr_idx >= 0:
                self.curr_sheet += 1
                if self.curr_sheet == len(self.sheets):
                    break
                go_to_prev_sheet = False
            else:
                self.curr_sheet -= 1
                go_to_prev_sheet = True

            print(f'Analysing TMA Folder: {self.sheets[self.curr_sheet]}')
            df = self.data[self.sheets[self.curr_sheet]]
            self.images = list(df['image path'])
            self.labels = list(df['label'])
            if first_sheet:
                self.curr_idx = self.get_initial_image()
            else:
                self.curr_idx = 0 if go_to_prev_sheet is False else len(self.images) - 1
            while 0 <= self.curr_idx < len(self.images):
                first_sheet = False
                im, label = self.images[self.curr_idx], self.labels[self.curr_idx]
                if self.exit:
                    break
                try:
                    self.curr_im_path = f'{self.root_dir}{im}'
                    self.curr_im = cv2.imread(self.curr_im_path) / 255
                    self.curr_brown_im = None if not self.space_on else self.color_brown()
                    if not pd.isna(label):  # set label of current image
                        self.curr_label = label
                        self.curr_im = self.color_edge('red' if label == 1 else 'blue' if label == 0 else 'black' if label == -1 else 'orange')
                    else:
                        self.curr_label = None
                    self.display = self.curr_im.copy() if not self.space_on else self.curr_brown_im
                    open_window()
                    to_break = False
                    while not to_break:
                        cv2.imshow('', self.display)
                        rv = cv2.waitKey()
                        if debug:
                            print(rv)
                        to_break = self.onkey(rv)
                    if self.exit:
                        self.goodbye_message()
                        break
                except Exception as e:
                    print(e)
                    self.curr_im = None
                    self.curr_im_path = None
                    self.curr_label = None
                    self.curr_idx += 1
                    continue
            self.save_sheet()
            if self.exit:
                break
        print('Done!')

    def goodbye_message(self):
        print('\n\n')
        if self.done <= 20:
            print(f'Lazy today? you only annotated {self.done} new images...')
        elif self.done <= 50:
            print(f'Well Done! you annotated {self.done} new images.')
        else:
            print(f'Wow! you annotated {self.done} new images!')
        print('Saving results to file and exiting, please wait...')

    def save_sheet(self):
        if self.curr_sheet < 0:
            return
        with pd.ExcelWriter(self.excel_file, engine='openpyxl', mode='a') as writer:
            for sheet in self.sheets:
                if sheet == self.sheets[self.curr_sheet]:
                    df = pd.DataFrame({'image path': self.images, 'label': self.labels})
                    self.data[self.sheets[self.curr_sheet]] = df

                df = self.data[sheet]
                workBook = writer.book
                workBook.remove(workBook[sheet])
                df.to_excel(writer, sheet_name=sheet, index=False)
            writer.save()

    def finish(self):
        close_window()
        self.curr_label = None
        self.exit = True

    def color_brown(self):
        self.curr_im = self.curr_im[..., ::-1]
        brown1 = np.array([0.41, 0.47, 0.6])
        brown2 = np.array([0.45, 0.4, 0.44])
        blue = np.array([0.39, 0.54, 0.8])

        # dist to brown:
        dist2brown1 = np.linalg.norm(self.curr_im.copy() - brown1, axis=-1, keepdims=True)
        dist2brown2 = np.linalg.norm(self.curr_im.copy() - brown2, axis=-1, keepdims=True)
        dist2brown = np.minimum(dist2brown1, dist2brown2)

        # dist blue brown:
        dist_blue_brown_1 = np.linalg.norm(brown1 - blue, axis=-1, keepdims=True)
        dist_blue_brown_2 = np.linalg.norm(brown2 - blue, axis=-1, keepdims=True)
        dist_blue_brown = np.minimum(dist_blue_brown_1, dist_blue_brown_2)

        eps = 0.03
        enhanced_im = np.concatenate([1 / (dist2brown + eps), np.ones_like(dist2brown) * (1 / (dist_blue_brown + eps)),
                                      eps * np.ones_like(dist2brown)], axis=2)
        enhanced_im = enhanced_im / enhanced_im.max()
        self.curr_im = self.curr_im[..., ::-1]
        return enhanced_im[..., ::-1]

    def color_edge(self, color, edge_width=10):
        im = self.curr_im.copy()
        if color == 'red':
            color = [0, 0, 1]
        elif color == 'blue':
            color = [1, 0, 0]
        elif color == 'black':
            color = [0, 0, 0]
        else:
            color = [2/255, 106/255, 253/255]
        im[:edge_width, :] = color
        im[:, :edge_width, :] = color
        im[-edge_width:, :] = color
        im[:, -edge_width:, :] = color
        return im

    def onkey(self, event):
        """
        The main logic: performing the desired action according to the key pressed while analysing the current image
        :param event: The value return by cv2.waitKey() after pressing the keyboard:
        "escape" --> save results and exit
        "enter" --> save decision of current image and continue to next image
        "1" --> set label "positive" for current image
        "0" --> set label "negative" for current image
        "5" --> set label "None" for current image
        "space" --> color the image by brown color value intensity.
        ">" --> discard decision for the current image and move to next image
        "<" --> discard decision for the current image and move to previous image
        "Tab" --> move to the first image in current sheet that is not annotated yet. if all are annotated - move the next sheet.
        :return: True if moving to next image or exiting, False if continue analysing the current image
        """
        if event == 27:  # escape
            self.finish()
            return True
        elif event == 13:  # enter
            self.labels[self.curr_idx] = copy.copy(self.curr_label)
            self.curr_idx += 1
            self.done += 1
            return True
        elif event == 49:  # 1
            self.curr_label = 1
            self.curr_im = self.color_edge('red')
            self.display = self.curr_im.copy()
            cv2.imshow('', self.display)
        elif event == 48:  # 0
            self.curr_label = 0
            self.curr_im = self.color_edge('blue')
            self.display = self.curr_im.copy()
            cv2.imshow('', self.curr_im)
        elif event == 57:  # 9
            self.curr_label = 2
            self.curr_im = self.color_edge('orange')
            self.display = self.curr_im.copy()
            cv2.imshow('', self.curr_im)
        elif event == 53:  # 5
            self.curr_label = -1
            self.curr_im = self.color_edge('black')
            self.display = self.curr_im.copy()
            cv2.imshow('', self.display)
        elif event == 32:  # space
            self.space_on = not self.space_on
            if not self.space_on:
                self.display = self.curr_im.copy()
                cv2.imshow('', self.display)
            else:
                if self.curr_brown_im is None:
                    self.curr_brown_im = self.color_brown()
                self.display = self.curr_brown_im
                cv2.imshow('', self.display)
        elif event == 46:  # >
            self.curr_label = None
            self.curr_idx += 1
            return True
        elif event == 44:  # <
            self.curr_label = None
            self.curr_idx -= 1
            if self.curr_sheet == 0: # don't go back if this is the first sheet
                self.curr_idx = np.max([0, self.curr_idx])
            return True
        elif event == 9:  # Tab
            self.curr_label = None
            self.curr_idx = self.get_initial_image()
            return True
        else:
            pass
        return False


if __name__ == '__main__':
    excel_file = r"C:\Users\amirlivn\Downloads\bliss\for_annotation\ER_annotations.xlsx"
    root_dir = r"C:\Users\amirlivn\Downloads\bliss"
    annotator = HIC_Annotator(excel_file, root_dir)
    annotator.run(debug=False)

