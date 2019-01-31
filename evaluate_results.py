#!/usr/bin/env python
# import poselab
import cv2
import os
import argparse
import numpy as np
import csv
import logging
import pickle
import matplotlib.pyplot as plt


def rad_to_deg(rad):
    return rad*180.0/np.pi


def wrap2Pi(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def normalize(q):
    return q/np.linalg.norm(q)

def qinv(q):
    """ Inverts a Quaternion """

    assert q.size == 4

    q[1:] = -q[1:]
    q = q/np.linalg.norm(q)

    return q

def Smtrx(x):
    """ Setup a Skew-Symmetrix Matrix """
    assert x.size == 3
    return np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]])

def quaternion_multiplication(a, b):
    """ Calculates the quaternion product between two quaternions """
    a0 = a[0]
    a1 = a[1:]
    b0 = b[0]
    b1 = b[1:]
    logging.debug('a0, type: {0}, value: {1}'.format(type(a0), a0))
    logging.debug('a1, type: {0}, value: {1}'.format(type(a1), a1))
    logging.debug('b0, type: {0}, value: {1}'.format(type(b0), b0))
    logging.debug('b1, type: {0}, value: {1}'.format(type(b1), b1))
    v0 = a0*b0-np.dot(a1, b1)
    v1 = a0*b1+a1*b0-np.dot(Smtrx(a1), b1)
    return np.insert(v1, 0, v0)


def get_posenet_error(q1, q2):
    """ corresponds to: theta =  2*acos(|z_0|), where z = q1*inv(q2) """

    ang = 2 * np.arccos(abs(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return ang * 180 / np.pi




if __name__ == '__main__':
    
    with open('tracking_results.pkl', 'rb') as f:
        results = pickle.load(f)

    blender_results = results['true']
    aruco_results = results['pred']

    # Euler Angle difference:

    ea_errors = []
    tvec_errors = []
    rot_errors = []
    trans_errors = []
    index = 0
    for b_ea, a_ea in zip(blender_results['ea'], aruco_results['ea']):

        error_tvec = (blender_results['tvec'][index] - aruco_results['tvec'][index])
        error_euler = (b_ea - a_ea)*180/np.pi
        #abs_error_tvec = abs(error_tvec)
        #abs_error_euler = abs(error_euler)
        error_rot = get_posenet_error(blender_results['q'][index], aruco_results['q'][index])
        error_trans = np.linalg.norm(error_tvec)
        if any(abs(rad_to_deg(error_euler)) > 180.0):
            error_euler = np.array([wrap2Pi(x) for x in error_euler.tolist()])

        index += 1
        ea_errors.append(error_euler)
        tvec_errors.append(error_tvec)
        rot_errors.append(error_rot)
        trans_errors.append(error_trans)


    ea_errors = np.array(ea_errors)
    tvec_errors = np.array(tvec_errors)
    abs_ea_errors = np.abs(ea_errors)
    abs_tvec_errors = np.abs(tvec_errors)

    roll_error = ea_errors[:, 0]
    pitch_error = ea_errors[:, 1]
    yaw_error = ea_errors[:, 2]
    x_error = tvec_errors[:, 0]
    y_error = tvec_errors[:, 1]
    z_error = tvec_errors[:, 2]

    abs_roll_error = abs_ea_errors[:, 0]
    abs_pitch_error = abs_ea_errors[:, 1]
    abs_yaw_error = abs_ea_errors[:, 2]
    abs_x_error = abs_tvec_errors[:, 0]
    abs_y_error = abs_tvec_errors[:, 1]
    abs_z_error = abs_tvec_errors[:, 2]




    #Metrics
    undetected = results['undetected']
    size = len(blender_results['tvec'])
    avg_trans = [np.mean(res) for res in [tvec_errors[:, i] for i in range(3)]]
    avg_rot = [np.mean(res) for res in [ea_errors[:, i] for i in range(3)]]

    abs_avg_trans = [np.mean(res) for res in [abs_tvec_errors[:, i] for i in range(3)]]
    abs_avg_rot = [np.mean(res) for res in [abs_ea_errors[:, i] for i in range(3)]]



    avg_rel_rot = np.mean(rot_errors)
    euc_trans = np.mean(trans_errors)



    data = """
    Number of undetected markers: {0} out of {1} samples.
    
    Average errors: 
    y: {2[0]:.4f} m
    z: {2[1]:.4f} m
    x: {2[2]:.4f} m
    pitch: {3[0]:.4f} deg
    yaw: {3[1]:.4f} deg
    roll: {3[2]:.4f} deg
    
    
    Average absolute value errors: 
    y: {6[0]:.4f} m
    z: {6[1]:.4f} m
    x: {6[2]:.4f} m
    pitch: {7[0]:.4f} deg
    yaw: {7[1]:.4f} deg
    roll: {7[2]:.4f} deg
    
    
    Average relative angle between predicted and true rotation: {4:.4f} deg
    Average Eucledian distance between predicted and true translation: {5:.4f} m
    """.format(undetected, size, avg_trans, avg_rot, avg_rel_rot, euc_trans, abs_avg_trans, abs_avg_rot)
    print(data)

    plt.figure(1, figsize=(13, 20))

    plt.subplot(322)
    plt.hist(roll_error, bins=50, edgecolor='k')
    plt.xlabel("pitch error [deg]")
    plt.ylabel("number of samples")
    plt.subplot(323)
    plt.hist(pitch_error, bins=50, edgecolor='k')
    plt.xlabel("yaw error [deg]")
    plt.ylabel("number of samples")
    plt.subplot(321)
    plt.hist(yaw_error, bins=50, edgecolor='k')
    plt.xlabel("roll error [deg]")
    plt.ylabel("number of samples")

    plt.subplot(325)
    plt.hist(x_error, bins=50, edgecolor='k')
    plt.xlabel("sway error [m]")
    plt.ylabel("number of samples")
    plt.subplot(326)
    plt.hist(y_error, bins=50, edgecolor='k')
    plt.xlabel("heave error [m]")
    plt.ylabel("number of samples")
    plt.subplot(324)
    plt.hist(z_error, bins=50, edgecolor='k')
    plt.xlabel("surge error [m]")
    plt.ylabel("number of samples")
    plt.savefig('Pose_res_2.eps', format='eps', dpi=1000)

    plt.figure(2, figsize=(13, 20))

    plt.subplot(322)
    plt.hist(abs_roll_error, bins=50, edgecolor='k')
    plt.xlabel("absolute pitch error [deg]")
    plt.ylabel("number of samples")
    plt.subplot(323)
    plt.hist(abs_pitch_error, bins=50, edgecolor='k')
    plt.xlabel("absolute yaw error [deg]")
    plt.ylabel("number of samples")
    plt.subplot(321)
    plt.hist(abs_yaw_error, bins=50, edgecolor='k')
    plt.xlabel("absolute roll error [deg]")
    plt.ylabel("number of samples")
    plt.subplot(325)
    plt.hist(abs_x_error, bins=50, edgecolor='k')
    plt.xlabel("absolute sway error [m]")
    plt.ylabel("number of samples")
    plt.subplot(326)
    plt.hist(abs_y_error, bins=50, edgecolor='k')
    plt.xlabel("absolute heave error [m]")
    plt.ylabel("number of samples")
    plt.subplot(324)
    plt.hist(abs_z_error, bins=50, edgecolor='k')
    plt.xlabel("absolute surge error [m]")
    plt.ylabel("number of samples")
    plt.savefig('Pose_res_3.eps', format='eps', dpi=1000)




    plt.figure(3, figsize=(10, 5))
    plt.subplot(121)
    plt.hist(rot_errors, bins=50, edgecolor='k')
    plt.xlabel("angle error [deg]")
    plt.ylabel("number of samples")
    plt.subplot(122)
    plt.hist(trans_errors, bins=50, edgecolor='k')
    plt.xlabel("Euclidean translational error [m]")
    plt.ylabel("number of samples")
    plt.savefig('Pose_res_1.eps', format='eps', dpi=1000)


    #plt.show()
