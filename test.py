import math
import helper
import posenet
import numpy as np
import pickle
from keras.optimizers import Adam
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

def quat_to_euler(q):
    roll = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    pitch = np.arcsin(2*(q[0]*q[2]) - q[3]*q[1])
    yaw = np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))

    return np.array([roll, pitch, yaw])


if __name__ == "__main__":
    # Test model
    model = posenet.create_posenet()
    model.load_weights('train5.h5')
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipvalue=2.0)
    model.compile(optimizer=adam, loss={'cls1_fc_pose_xyz': posenet.euc_loss1x, 'cls1_fc_pose_wpqr': posenet.euc_loss1q,
                                        'cls2_fc_pose_xyz': posenet.euc_loss2x, 'cls2_fc_pose_wpqr': posenet.euc_loss2q,
                                        'cls3_fc_pose_xyz': posenet.euc_loss3x, 'cls3_fc_pose_wpqr': posenet.euc_loss3q})

    #dataset_train, dataset_test = helper.getKings()
    dataset_train, dataset_test = helper.getBlender()

    X_test = np.squeeze(np.array(dataset_test.images))
    y_test = np.squeeze(np.array(dataset_test.poses))


    testPredict = model.predict(X_test)


    valsx = testPredict[4]
    valsq = testPredict[5]

    # Get results... :/
    errors_x = []
    errors_q = []
    thetas = []

    roll_error = []
    pitch_error = []
    yaw_error = []
    surge_error = []
    sway_error = []
    heave_error = []

    results = {
        'true': {'ID': [], 'q': [], 'ea': [],  'tvec': []},
        'pred': {'ID': [], 'q': [], 'ea': [], 'tvec': []},
        'undetected': []}

    for i in range(len(dataset_test.images)):
        q = np.asarray(dataset_test.poses[i][3:7])
        results['true']['ID'].append("data_{}".format(i))
        results['true']['q'].append(q)
        results['true']['ea'].append(quat_to_euler(q))
        results['true']['tvec'].append(np.asarray(dataset_test.poses[i][0:3]))

        results['pred']['ID'].append("data_{}".format(i))
        results['pred']['q'].append(valsq[i])
        results['pred']['ea'].append(quat_to_euler(valsq[i]))
        results['pred']['tvec'].append(valsx[i])

    with open('tracking_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    with open("results.txt", "w") as my_output_file:
        results = np.zeros((len(dataset_test.images), 2))
        for i in range(len(dataset_test.images)):

            pose_q= np.asarray(dataset_test.poses[i][3:7])
            pose_x= np.asarray(dataset_test.poses[i][0:3])
            predicted_x = valsx[i]
            predicted_q = valsq[i]

            res = np.concatenate((predicted_x, predicted_q))

            string = np.array2string(res, separator="\t")
            my_output_file.write("{}\t {}\t {}\t {}\t {}\t {}\t {}\n".format(res[0], res[1], res[2], res[3], res[4], res[5], res[6]))

            pose_q = np.squeeze(pose_q)
            pose_x = np.squeeze(pose_x)
            predicted_q = np.squeeze(predicted_q)
            predicted_x = np.squeeze(predicted_x)

            #Compute Individual Sample Error
            q1 = pose_q / np.linalg.norm(pose_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1,q2)))
            theta = 2 * np.arccos(d) * 180/math.pi
            error_q = np.arccos(np.dot(q1, q2))/(np.linalg.norm(q1)*np.linalg.norm(q2)) * 180/math.pi
            error_x = np.linalg.norm(pose_x-predicted_x)
            results[i,:] = [error_x,theta]
            thetas.append(theta)
            errors_q.append(error_q)
            errors_x.append(error_x)

            eul_pred = quat_to_euler(q2) * 180/np.pi
            eul_true = quat_to_euler(q1) * 180/np.pi
            diff_euler = eul_pred - eul_true

            if diff_euler[0] > 180:
                diff_euler[0] -= 180
            elif diff_euler[0] < -180:
                diff_euler[0] += 180
            roll_error.append(diff_euler[0])
            pitch_error.append(diff_euler[1])
            yaw_error.append(diff_euler[2])

            diff_x = predicted_x-pose_x
            surge_error.append(diff_x[0])
            sway_error.append(diff_x[1])
            heave_error.append(diff_x[2])



            print('Iteration:  ', i, '  Error XYZ (m):  ', error_x, '  Error Q (degrees):  ', theta)
    median_result = np.median(results,axis=0)
    print('Median error ', median_result[0], 'm  and ', median_result[1], 'degrees.')
    print('Avergae translational error ', np.average(errors_x), 'm and rotation error ', np.average(errors_q), 'deg ' \
        ' and rotation 2 error ', np.average(thetas), ' deg.')

    #Calculate 6DOF errors
    roll_avg = np.average(roll_error)
    pitch_avg = np.average(pitch_error)
    yaw_avg = np.average(yaw_error)

    surge_avg = np.average(surge_error)
    sway_avg = np.average(sway_error)
    heave_avg = np.average(heave_error)





    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.hist(errors_x, bins = 50)
    plt.xlabel("Translation error [m]")
    plt.title("Translation error")


    plt.subplot(122)
    plt.hist(thetas, bins=50)
    plt.xlabel("Rotation error2 [deg]")
    plt.title("Rotation error 2")

    plt.suptitle("Performance of CNN PoseNet estimator, on Blender dataset", fontsize = 16)

    plt.figure(2, figsize=(20, 15))
    plt.subplot(231)
    plt.hist(roll_error, bins=30)
    plt.xlabel("roll error [deg]")
    plt.subplot(232)
    plt.hist(pitch_error, bins=30)
    plt.xlabel("pitch error [deg]")
    plt.subplot(233)
    plt.hist(yaw_error, bins=30)
    plt.xlabel("yaw error [deg]")
    plt.subplot(234)
    plt.hist(surge_error, bins=30)
    plt.xlabel("translational error x [m]")
    plt.subplot(235)
    plt.hist(sway_error, bins=30)
    plt.xlabel("translational error y [m]")
    plt.subplot(236)
    plt.hist(heave_error, bins=30)
    plt.xlabel("translational error z [m]")

    plt.show()