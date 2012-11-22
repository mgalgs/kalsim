#!/usr/bin/python
import sys
import cv
import numpy
import random
import matplotlib.pyplot as plt
from optparse import OptionParser

from numpy import mat,eye,zeros,array,diag,average

import drawing
import kalcommon as kc
from brownian import brownian


def main(make_video = True, estimates=None, plotting=['cam1', 'cam2', 'kalman']):
    print("Hello, world! Let's run a Kalman filter simulation...")
    if estimates:
        d = drawing.Drawing(draw_estimates=estimates)
    else:
        d = drawing.Drawing()

    # some useful constants:
    total_seconds = 18
    total_frames = total_seconds*kc.fps
    center_coords = (kc.img_size[0]/2, kc.img_size[1]/2)


    print ('Doing simulation...')
    # get the ball motion (a numpy 2xnframes matrix of coords):
    # ball_motion = kc.get_brownian_ball_motion(center_coords, total_frames)
    ball_motion = kc.get_simple_ball_motion(center_coords, total_seconds)
    # ball_motion = kc.get_still_ball_motion(center_coords, total_seconds)



    cam1_estimates = []
    cam2_estimates = []
    # cam3_estimates = []
    average_estimates = []
    closest_camera_estimates = []
    kalman_estimates = []

    # do the kalman filtering:
    A_x = A_y = 0           # velocity of ball (calculated at each time step)
    xhat_x = xhat_y = 0
    P_x = P_y = 0
    xhatminus_x = xhatminus_y = 0
    Pminus_x = Pminus_y = 0
    K_x = K_y = 0

    xhat0_x = xhat0_y = 300
    P0_x = P0_y = 1

    xhatprev_x = xhat0_x
    Pprev_x = P0_x
    xhatprev_y = xhat0_y
    Pprev_y = P0_y
    for cnt,c in enumerate(ball_motion):
        # "PREDICT" (time update)
        xhatminus_x = xhatprev_x + (A_x)
        Pminus_x = Pprev_x + kc.process_uncertainty
        xhatminus_y = xhatprev_y + (A_y)
        Pminus_y = Pprev_y + kc.process_uncertainty

        # Take "measurements":
        (cam1_estimate,cam1_sigma) = kc.get_cam_estimate(c, d.cam1center)
        (cam2_estimate,cam2_sigma) = kc.get_cam_estimate(c, d.cam2center)
        # (cam3_estimate,cam3_sigma) = kc.get_cam_estimate(c, d.cam3center)

        # z = mat([cam1_estimate, cam2_estimate, cam3_estimate])
        (z_x1,z_y1) = cam1_estimate
        (z_x2,z_y2) = cam2_estimate

        # Measurement noise covariance matrix:
        R = pow(cam1_sigma,2) + pow(cam2_sigma,2)
        z_x = (pow(cam2_sigma,2)/(R))*z_x1 + (pow(cam1_sigma,2)/(R))*z_x2
        z_y = (pow(cam2_sigma,2)/(R))*z_y1 + (pow(cam1_sigma,2)/(R))*z_y2
        # R = .01

        # "CORRECT" (measurement update)
        K_x = Pminus_x / (Pminus_x + R)
        xhat_x = xhatminus_x + K_x * (z_x - xhatminus_x)
        P_x = (1 - K_x) * Pminus_x
        K_y = Pminus_y / (Pminus_y + R)
        xhat_y = xhatminus_y + K_y * (z_y - xhatminus_y)
        P_y = (1 - K_y) * Pminus_y


        # We assume constant linear motion, so update the A values
        # accordingly for the next iteration:
        if cnt > 0:
            A_x = ball_motion[cnt][0] - ball_motion[cnt-1][0]
            A_y = ball_motion[cnt][1] - ball_motion[cnt-1][1]


        # save this result for the next iteration:
        xhatprev_x = xhat_x
        Pprev_x = P_x
        xhatprev_y = xhat_y
        Pprev_y = P_y

        # save measurements for later plotting:
        cam1_estimates.append(cam1_estimate)
        cam2_estimates.append(cam2_estimate)
        # cam3_estimates.append(cam3_estimate)
        kalman_estimates.append((int(xhat_x), int(xhat_y)))

        closest_camera_estimates.append(cam1_estimate if cam1_sigma < cam2_sigma else cam2_estimate)

        average_estimates.append((
            int(average([cam1_estimate[0], cam2_estimate[0]])),
            int(average([cam1_estimate[1], cam2_estimate[1]]))
            ))

    # eo filter loop
    print ('Done!')

    # for v1,v2 in zip(closest_camera_estimates, average_estimates):
    #     print v1,v2,kc.get_dist_between_2_points(v1,v2)


    print ('Saving images...')
    # plot the actual and estimated trajectories:
    plt.plot([c[0] for c in ball_motion], [kc.img_size[1] - c[1] for c in ball_motion], kc.plotcolors['actual'], label="Actual Trajectory")
    plt.plot(ball_motion[0][0], kc.img_size[1] - ball_motion[0][1], 'ro')
    plt.plot(ball_motion[-1][0], kc.img_size[1] - ball_motion[-1][1], 'go')
    if 'cam1' in plotting:
        plt.plot([c[0] for c in cam1_estimates], [kc.img_size[1] - c[1] for c in cam1_estimates], kc.plotcolors['cam1'], label="Camera 1 Estimate")
    if 'cam2' in plotting:
        plt.plot([c[0] for c in cam2_estimates], [kc.img_size[1] - c[1] for c in cam2_estimates], kc.plotcolors['cam2'], label="Camera 2 Estimate")
    if 'average' in plotting:
        plt.plot([c[0] for c in average_estimates], [kc.img_size[1] - c[1] for c in average_estimates], kc.plotcolors['average'], label="Average Estimate")
    if 'closest' in plotting:
        plt.plot([c[0] for c in closest_camera_estimates], [kc.img_size[1] - c[1] for c in closest_camera_estimates], kc.plotcolors['closest'], label="Closest Camera Estimate")
    if 'kalman' in plotting:
        plt.plot([c[0] for c in kalman_estimates], [kc.img_size[1] - c[1] for c in kalman_estimates], kc.plotcolors['kalman'], label="Kalman Estimate")
    plt.title('Ball Trajectory')
    plt.legend(loc='best')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig(kc.trajectories_filename, format='png')

    plt.cla()
    # plot the x values:
    plt.plot([c[0] for c in ball_motion], kc.plotcolors['actual'], label="Actual Position")
    if 'cam1' in plotting:
        plt.plot([c[0] for c in cam1_estimates], kc.plotcolors['cam1'], label="Camera 1 Estimate")
    if 'cam2' in plotting:
        plt.plot([c[0] for c in cam2_estimates], kc.plotcolors['cam2'], label="Camera 2 Estimate")
    # plt.plot([c[0] for c in cam3_estimates], label="Camera 3 Estimate")
    if 'average' in plotting:
        plt.plot([c[0] for c in average_estimates], kc.plotcolors['average'], label="Average Estimate")
    if 'closest' in plotting:
        plt.plot([c[0] for c in closest_camera_estimates], kc.plotcolors['closest'], label="Closest Camera Estimate")
    if 'kalman' in plotting:
        plt.plot([c[0] for c in kalman_estimates], kc.plotcolors['kalman'], label="Kalman Estimate")
    plt.title('X Values')
    plt.legend(loc='best')
    plt.xlabel('Frames')
    plt.ylabel('Pixels')
    plt.savefig(kc.x_estimates_filename, format='png')

    plt.cla()
    # plot the y values:
    plt.plot([c[1] for c in ball_motion], kc.plotcolors['actual'], label="Actual Position")
    if 'cam1' in plotting:
        plt.plot([c[1] for c in cam1_estimates], kc.plotcolors['cam1'], label="Camera 1 Estimate")
    if 'cam2' in plotting:
        plt.plot([c[1] for c in cam2_estimates], kc.plotcolors['cam2'], label="Camera 2 Estimate")
    # plt.plot([c[1] for c in cam3_estimates], label="Camera 3 Estimate")
    if 'average' in plotting:
        plt.plot([c[1] for c in average_estimates], kc.plotcolors['average'], label="Average Estimate")
    if 'closest' in plotting:
        plt.plot([c[1] for c in closest_camera_estimates], kc.plotcolors['closest'], label="Closest Camera Estimate")
    if 'kalman' in plotting:
        plt.plot([c[1] for c in kalman_estimates], kc.plotcolors['kalman'], label="Kalman Estimate")
    plt.title('Y Values')
    plt.legend(loc='best')
    plt.xlabel('Frames')
    plt.ylabel('Pixels')
    plt.savefig(kc.y_estimates_filename, format='png')

    plt.cla()
    # plot the errors:
    cam1errors = [kc.get_dist_between_2_points(c[0], c[1]) for c in zip(ball_motion, cam1_estimates)]
    cam2errors = [kc.get_dist_between_2_points(c[0], c[1]) for c in zip(ball_motion, cam2_estimates)]
    # cam3errors = [kc.get_dist_between_2_points(c[0], c[1]) for c in zip(ball_motion, cam3_estimates)]
    averageerrors = [kc.get_dist_between_2_points(c[0], c[1]) for c in zip(ball_motion, average_estimates)]
    closesterrors = [kc.get_dist_between_2_points(c[0], c[1]) for c in zip(ball_motion, closest_camera_estimates)]
    kalmanerrors = [kc.get_dist_between_2_points(c[0], c[1]) for c in zip(ball_motion, kalman_estimates)]


    if 'cam1' in plotting:
        plt.plot(cam1errors, kc.plotcolors['cam1'], label='Camera 1')
    if 'cam2' in plotting:
        plt.plot(cam2errors, kc.plotcolors['cam2'], label='Camera 2')
    # plt.plot(cam3errors, label='Camera 3')
    if 'average' in plotting:
        plt.plot(averageerrors, kc.plotcolors['average'], label='Average')
    if 'closest' in plotting:
        plt.plot(closesterrors, kc.plotcolors['closest'], label='Closest Camera')
    if 'kalman' in plotting:
        plt.plot(kalmanerrors, kc.plotcolors['kalman'], label='Kalman')
    plt.title('Estimation Errors')
    plt.legend(loc='best')
    plt.xlabel('Frames')
    plt.ylabel('Pixels')
    plt.savefig(kc.cam_errors_filename, format='png')

    w = numpy.bartlett(25)
    cam1errors_filtered = numpy.convolve(w/w.sum(), cam1errors, mode='same')
    cam2errors_filtered = numpy.convolve(w/w.sum(), cam2errors, mode='same')
    # cam3errors_filtered = numpy.convolve(w/w.sum(), cam3errors, mode='same')
    averageerrors_filtered = numpy.convolve(w/w.sum(), averageerrors, mode='same')
    closesterrors_filtered = numpy.convolve(w/w.sum(), closesterrors, mode='same')
    kalmanerrors_filtered = numpy.convolve(w/w.sum(), kalmanerrors, mode='same')

    plt.cla()
    if 'cam1' in plotting:
        # plt.plot(cam1errors_filtered, kc.plotcolors['cam1'], label='Camera 1')
        plt.plot(cam1errors_filtered, kc.plotcolors['cam1'], label='Camera 1', linestyle='--')
    if 'cam2' in plotting:
        # plt.plot(cam2errors_filtered, kc.plotcolors['cam2'], label='Camera 2')
        plt.plot(cam2errors_filtered, kc.plotcolors['cam2'], label='Camera 2', linestyle='--')
    # plt.plot(cam3errors_filtered, label='Camera 3')
    if 'average' in plotting:
        plt.plot(averageerrors_filtered, kc.plotcolors['average'], label='Average', linewidth=2)
    if 'closest' in plotting:
        plt.plot(closesterrors_filtered, kc.plotcolors['closest'], label='Closest', linewidth=2)
    if 'kalman' in plotting:
        plt.plot(kalmanerrors_filtered, kc.plotcolors['kalman'], label='Kalman', linewidth=2)
    plt.title('Smoothed Estimation Errors')
    plt.legend(loc='best')
    plt.xlabel('Frames')
    plt.ylabel('Pixels')
    plt.savefig(kc.cam_errors_filtered_filename, format='png')

    print ('Done!')

    if make_video:
        print ('Rendering video...')
        # write out the video:
        stiter = iter(['Motion: Straight Line', 'Motion: Straight Line', 'Motion: Sinusoid'])
        for t in xrange(len(ball_motion)):
            # ball coordinates:
            d.ball_coords = ball_motion[t]
            # set up the status text:
            d.cam1_estimate = cam1_estimates[t]
            d.cam2_estimate = cam2_estimates[t]
            # d.cam3_estimate = cam3_estimates[t]
            d.average_estimate = average_estimates[t]
            d.kalman_estimate = kalman_estimates[t]
            d.frame_num = t
            if not (t % (len(ball_motion)/3)):
                d.status_text = stiter.next()

            # do the drawing
            img = d.get_base_image()

            d.write_frame(img)

        print ("Done!")


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d","--draw", dest="draw_estimates", action="append")
    parser.add_option("-p","--plot", dest="plotting", action="append")
    parser.add_option("-n","--no-video", dest="no_video", action="store_true", default=False)
    (options, args) = parser.parse_args()
    # print options, args
    main(make_video = not options.no_video,
         estimates=options.draw_estimates,
         plotting=options.plotting if options.plotting else ['cam1', 'cam2', 'kalman'])
