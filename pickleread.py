import pickle
import numpy as np

with open('tracking_results.pkl', 'rb') as f:
    results = pickle.load(f)

blender_results = results['true']
aruco_results = results['pred']

for b_q, a_q in zip(blender_results['ea'], aruco_results['q']):
    print(b_q)

