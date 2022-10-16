import cv2
import pyrealsense2 as rs
import numpy as np
import math


DEPTH_FRAME_WIDTH=640
DEPTH_FRAME_HEIGHT=480

RGB_FRAME_WIDTH=1280
RGB_FRAME_HEIGHT=720

IR_FRAME_WIDTH=1280
IR_FRAME_HEIGHT=720

CORNER_SIZE = (10,6)

# align
def sort_corners(cb_corners):
    if cb_corners[0,0,1] > cb_corners[-1,0,1]:
        return cb_corners[::-1]
    return cb_corners

# SVD
def find_plane(points):
    c = np.mean(points, axis=0)
    r0 = points - c
    u, s, v = np.linalg.svd(r0)
    nv = v[-1, :]
    ds = np.dot(points, nv)
    param = np.r_[nv, -np.mean(ds)]
    return param

# render
def get_center(cb_corners):
    r = cb_corners[0,0,0] + cb_corners[-1,0,0]
    c = cb_corners[0,0,1] + cb_corners[-1,0,1]
    return tuple(map(int, [r/2, c/2]))


if __name__ == '__main__':
    pipeline =rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth ,DEPTH_FRAME_WIDTH, DEPTH_FRAME_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, RGB_FRAME_WIDTH, RGB_FRAME_HEIGHT, rs.format.bgr8, 30)
    
    profile=pipeline.start(config)
    depth_sensor=profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 5)     # set short-range

    align_to=rs.stream.color
    align = rs.align(align_to)

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        cb_found_color, cb_corners_color=cv2.findChessboardCorners(aligned_color_image, CORNER_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if cb_found_color:
            cb_corners_color = sort_corners(cb_corners_color)
            center = get_center(cb_corners_color)

            points = np.zeros([len(cb_corners_color),3])
            for i, corner in enumerate(cb_corners_color):
                x = corner[0][0]
                y = corner[0][1]
                dist = aligned_depth_frame.get_distance(x,y) * 1000 # Meter -> mm
                
                Xtemp = dist*(x-intr.ppx)/intr.fx
                Ytemp = dist*(y-intr.ppy)/intr.fy
                Ztemp = dist
                points[i,0]=Xtemp
                points[i,1]=Ytemp
                points[i,2]=Ztemp
            
            param = find_plane(points)

            alpha = math.atan(param[2]/param[0])*180/math.pi
            if alpha <0 :
                alpha = alpha +90
            else:
                alpha = alpha -90

            gamma = math.atan(param[2]/param[1])*180/math.pi
            if gamma <0 :
                gamma = gamma +90
            else:
                gamma = gamma -90

            txt_alpha = "theta1: "+str(round(alpha))
            txt_gamma = "theta2: "+str(round(gamma))

            cv2.putText(aligned_color_image, txt_alpha, (100,100), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255))
            cv2.putText(aligned_color_image, txt_gamma, (100,150), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,0,255))

            print(f"alpha : {alpha}, gamma : {gamma}")

            alpha_dst = tuple([int(center[0]+alpha*5), int(center[1])])
            gamma_dst = tuple([int(center[0]), int(center[1]+gamma*5)])
            cv2.line(aligned_color_image, tuple(map(int,center)), alpha_dst, (255,0,0), 2)
            cv2.line(aligned_color_image, tuple(map(int,center)), gamma_dst, (0,0,255), 2)

        cv2.imshow("img", aligned_color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    pipeline.stop()


