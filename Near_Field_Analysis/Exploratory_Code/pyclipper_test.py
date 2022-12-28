#Matthew Leung
#June 2022
#Test of pyclipper Python library, which is a Python wrapper for C++ Clipper library: http://www.angusj.com/delphi/clipper.php
#pyclipper: https://github.com/fonttools/pyclipper

import numpy as np
import matplotlib.pyplot as plt
import pyclipper

if __name__ == "__main__":
    #Simple rectangle:
    subj = ((180, 200), (260, 200), (260, 150), (180, 150))
    #More complicated polygon:
    subj = [[180,180], [200,200], [220,170], [235,210],[260,160],[220,150],[190,150],[180,160]]
    
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(subj, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    
    shape_offest_amount = -5
    solution = pco.Execute(shape_offest_amount)
    
    #subj_np = np.array(list(zip(*subj))).T #<-- No need for this!
    subj_np = np.array(subj)
    solution_np = np.array(solution[0])
    
    plt.figure()
    plt.plot(subj_np[:,0], subj_np[:,1], color='tab:blue', label='Original Shape')
    plt.plot([subj_np[0,0],subj_np[-1,0]], [subj_np[0,1],subj_np[-1,1]], color='tab:blue')
    plt.plot(solution_np[:,0], solution_np[:,1], color='tab:orange', label='Inward Offset')
    plt.plot([solution_np[0,0],solution_np[-1,0]], [solution_np[0,1],solution_np[-1,1]], color='tab:orange')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
    
    print(solution)

