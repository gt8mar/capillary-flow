import time
import os
import csv
import numpy as np
import cv2

def main(path):
    centerline_folder = os.path.join(path, 'centerlines', 'coords')
    output_folder = os.path.join(path, 'rbc')
    os.makedirs(output_folder, exist_ok=True)
    for centerline in os.listdir(centerline_folder):
        print(centerline)
        date = centerline.split(' ')[0]
        video = centerline.split(' ')[1].split('_')[0]
        capnum = centerline[-6:-4]

        video_folder = os.path.join(os.path.dirname(path), 'pair_vids')
        for vid in os.listdir(video_folder):
            if date in vid and video in vid:
                video_file = os.path.join(video_folder, vid)
                break
        print(video_file)

        centerline_file = os.path.join(centerline_folder, centerline)
        with open(centerline_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        caplength = len(rows)
        intervals = [round(caplength * i/100) for i in [25, 50, 75]]
        intervals_list = ['25', '50', '75']
        names = intervals_list.copy()
        for point in intervals:
            radius = int(float(rows[point][2]))
            point_x = int(float(rows[point][0]))
            point_y = int(float(rows[point][1]))
            perp_coords = []
            for i in range(point - radius + 1, point + radius):
                perp_coords.append([point_x - (int(float(rows[i][1])) - point_y), point_y + (int(float(rows[i][0])) - point_x)])

                frames = os.listdir(video_file)
                frames = [frame for frame in frames if frame.endswith('.tiff')]
                num_frames = len(frames)

            rbc_img = np.zeros((len(perp_coords), num_frames))
            for j in range(num_frames):
                frame = cv2.imread(os.path.join(video_file, frames[j]), cv2.IMREAD_GRAYSCALE)
                frame2 = np.copy(frame)
                for i in range(len(perp_coords)):
                    frame2[perp_coords[i][0]][perp_coords[i][1]] = 255
                """cv2.imshow('frame', frame2)
                resized = cv2.resize(frame2, (864,648))
                cv2.imshow('frame', resized)
                cv2.waitKey(0)"""
                for i in range(len(perp_coords)):
                    rbc_img[i][j] = frame[perp_coords[i][0]][perp_coords[i][1]]
                    rbc_img = rbc_img.astype(np.uint8)
            cv2.imwrite(os.path.join(output_folder, date + ' ' + video + '_' + capnum + '_' + names.pop(0) + '.tiff'), rbc_img)


                
        
        

if __name__ == '__main__':
    ticks = time.time()
    main(path = 'D:\\frawg\\Wake Sleep Pairs\\gabby_analysis')
    print("--------------------")
    print("Runtime: " + str(time.time() - ticks))    