#Matthew Leung
#May 2022
#This script plots some angular velocity profiles that I tried implementing
#for the stepper motor, using the Arduino AccelStepper library.

import os
import numpy as np
import matplotlib.pyplot as plt

def step_delay_fcn(x, omega_max):
    #Constant speed
    f1 = lambda t: omega_max
    f2 = lambda t: 0
    f3 = lambda t: -omega_max
    f4 = lambda t: 0
    return np.piecewise(x, [x<=0, (x>0)*(x<0.9), (x>=0.9)*(x<1.0), (x>=1.0)*(x<1.9), (x>=1.9)*(x<2.0)], [f2,f1,f2,f3,f4])

def triangle_fcn(x, omega_max):
    #Triangular shaped angular velocity profile
    half_period = 1.0
    t_delay = 0.1
    t_rise = (half_period-t_delay) / 2
    f0 = lambda t: 0
    f1 = lambda t: omega_max/t_rise * t
    f2 = lambda t: -omega_max/t_rise * (t-t_rise) + omega_max
    f3 = lambda t: 0
    f4 = lambda t: -omega_max/t_rise * (t-half_period)
    f5 = lambda t: omega_max/t_rise * (t-half_period-t_rise) - omega_max
    f6 = lambda t: 0
    return np.piecewise(x, [x<=0,
                            (x>0)*(x<t_rise),
                            (x>=t_rise)*(x<2*t_rise),
                            (x>=2*t_rise)*(x<half_period-t_delay),
                            (x>=half_period)*(x<half_period+t_rise),
                            (x>=half_period+t_rise)*(x<half_period+2*t_rise),
                            (x>=1.9)*(x<2.0)],
                            [f0,f1,f2,f3,f4,f5,f6])

def trapezoid_fcn(x, omega_max):
    #Trapezoid shaped angular velocity profile
    half_period = 1.0
    t_delay = 0.1
    t_rise = 0.2
    f0 = lambda t: 0
    f1 = lambda t: omega_max/t_rise * t
    f2 = lambda t: omega_max
    f3 = lambda t: -omega_max/t_rise * (t-(half_period-t_rise-t_delay)) + omega_max
    f4 = lambda t: 0
    f5 = lambda t: -omega_max/t_rise * (t-half_period)
    f6 = lambda t: -omega_max
    f7 = lambda t: omega_max/t_rise * (t-(2*half_period-t_rise-t_delay)) - omega_max
    f8 = lambda t: 0
    return np.piecewise(x, [x<=0,
                            (x>0)*(x<t_rise),
                            (x>=t_rise)*(x<half_period-t_rise-t_delay),
                            (x>=half_period-t_rise-t_delay)*(x<half_period-t_delay),
                            (x>=half_period-t_delay)*(x<half_period),
                            (x>=half_period)*(x<half_period+t_rise),
                            
                            (x>=half_period+t_rise)*(x<half_period+half_period-t_rise-t_delay),
                            (x>=half_period+half_period-t_rise-t_delay)*(x<half_period+half_period-t_delay),
                            (x>=half_period+half_period-t_delay)*(x<half_period+half_period)],
                            [f0,f1,f2,f3,f4,f5,f6,f7,f8])


if __name__ == "__main__":
    savedir = os.path.join(os.getcwd(), 'velocity_plot_demo')
    if os.path.isdir(savedir) == False: os.mkdir(savedir)
    
    plotpoints = np.linspace(0,2,num=500)
    plt.figure()
    plt.plot(plotpoints, step_delay_fcn(plotpoints, np.pi))
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')
    savename = os.path.join(savedir, 'velocity_plot_step.pdf')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    
    plt.figure()
    plt.plot(plotpoints, triangle_fcn(plotpoints, np.pi))
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')
    savename = os.path.join(savedir, 'velocity_plot_tri.pdf')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    
    plt.figure()
    plt.plot(plotpoints, trapezoid_fcn(plotpoints, np.pi))
    plt.xlabel('Time [s]')
    plt.ylabel('Angular Velocity [rad/s]')
    savename = os.path.join(savedir, 'velocity_plot_trap.pdf')
    plt.savefig(savename, bbox_inches='tight')
    plt.show()
    