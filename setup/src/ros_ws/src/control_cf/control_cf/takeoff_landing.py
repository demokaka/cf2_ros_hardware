import numpy as np
from control_cf.traj_gen import *

"""
    Take off action: There are 3 types 
    1. Generate a trajectory from initial (x0,y0) position on the ground to (x0,y0,z0) position
    2. Measure current position, generate a trajectory from that position to the height z0 
    In both cases, track the takeoff reference.
    3. Continually increase the thrust value until the thrust_0 for equilibrium right at the height z0, 
    set all angles at 0 degs
    ===> Need time of takeoff
    Landing action: 
    1. Measure current position (x0,y0,z0), generate trajectory to land at (x0,y0,0) and track it
    2. Just continually cut down the thrust and other angles are set at 0
    ===> Need time of landing
"""

def generate_takeoff_traj(x0, y0, z0, T_takeoff, T_hovering, Ts):
    
    h0 = z0
    t1 = T_takeoff
    t_a = np.arange(start=0,stop=t1,step=Ts)
    ref_takeoff = {}

    a = 6*h0/T_takeoff**5
    b = -15*h0/T_takeoff**4
    c = 10*h0/T_takeoff**3

    x_a = np.zeros(len(t_a))
    y_a = np.zeros(len(t_a))
    z_a = np.zeros(len(t_a))
    vx_a = np.zeros(len(t_a))
    vy_a = np.zeros(len(t_a))
    vz_a = np.zeros(len(t_a))
    az_a = np.zeros(len(t_a))

    # position
    x_a = x0 * np.ones(len(t_a))
    y_a = y0 * np.ones(len(t_a))
    z_a = a * np.multiply(np.multiply(np.multiply(np.multiply(t_a,t_a),t_a),t_a),t_a) + b * np.multiply(np.multiply(np.multiply(t_a,t_a),t_a),t_a) + c * np.multiply(np.multiply(t_a,t_a),t_a) 
    # velocity
    vz_a= 5 * a * np.multiply(np.multiply(np.multiply(t_a,t_a),t_a),t_a) + 4 * b * np.multiply(np.multiply(t_a,t_a),t_a) + 3 * c * np.multiply(t_a,t_a)
    # acceleration
    az_a = 20 * a * np.multiply(np.multiply(t_a,t_a),t_a) + 12 * b * np.multiply(t_a,t_a) + 6 * c * t_a

    ref_pos_a = from_setpoints_to_ref(x_a,y_a,z_a,Ts,T_takeoff)
    ref_vel_a = from_setpoints_to_ref(vx_a,vy_a,vz_a,Ts,T_takeoff)
    ref_a = concat_element_ref(ref_pos_a,ref_vel_a)
    vref_a = from_setpoints_to_ref(vx_a,vy_a,az_a,Ts,T_takeoff)

    # TakeOff--->Hovering
    t2 = t1 + T_hovering
    t_b = np.arange(start=t1,stop=t2,step=Ts)
    x_b = np.zeros(len(t_a))
    y_b = np.zeros(len(t_a))
    z_b = np.zeros(len(t_a)) 
    vx_b = np.zeros(len(t_a))
    vy_b = np.zeros(len(t_a))
    vz_b = np.zeros(len(t_a))


    x_b = x0 * np.ones(len(t_b))
    y_b = y0 * np.ones(len(t_b))
    z_b = z0  * np.ones(len(t_b)) 

    ref_pos_b = from_setpoints_to_ref(x_b,y_b,z_b,Ts,T_hovering)
    ref_vel_b = from_setpoints_to_ref(vx_b,vy_b,vz_b,Ts,T_hovering)
    ref_b = concat_element_ref(ref_pos_b,ref_vel_b)
    vref_b = from_setpoints_to_ref(vx_b,vy_b,vz_b,Ts,T_hovering)

    ref_full_takeoff = concat_ref(ref_a,ref_b)
    v_ref_takeoff = concat_ref(vref_a,vref_b)
    tt_takeoff = np.hstack((t_a,t_b))

    ref_full_takeoff = ref_full_takeoff.T
    v_ref_takeoff = v_ref_takeoff.T
    
    psi = 0.0
    ddx, ddy, ddz = v_ref_takeoff[0, :], v_ref_takeoff[1, :], v_ref_takeoff[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust)
    theta = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + 9.81))
    ref_takeoff = {
        "trajectory": ref_full_takeoff,
        "time_step": tt_takeoff,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt_takeoff.shape[0],
        "v_ref": v_ref_takeoff}
    
    return ref_takeoff

def generate_landing_traj(x0, y0, z0, T_hovering, T_landing, Ts):
    # Flight--->Hovering
    hf = z0
    print(hf)
    t1 = T_hovering
    t_a = np.arange(start=0,stop=t1,step=Ts)

    x_a = np.zeros(len(t_a))
    y_a = np.zeros(len(t_a))
    z_a = np.zeros(len(t_a))

    vx_a = np.zeros(len(t_a))
    vy_a = np.zeros(len(t_a))  
    vz_a = np.zeros(len(t_a))


    x_a = x0 * np.ones(len(t_a))
    y_a = y0 * np.ones(len(t_a))
    z_a = z0 * np.ones(len(t_a)) 

    ref_pos_a = from_setpoints_to_ref(x_a,y_a,z_a,Ts,T_hovering)
    ref_vel_a = from_setpoints_to_ref(vx_a,vy_a,vz_a,Ts,T_hovering)
    ref_a = concat_element_ref(ref_pos_a,ref_vel_a)
    vref_a = from_setpoints_to_ref(vx_a,vy_a,vz_a,Ts,T_hovering)

    # Hovering--->Landing
    t2 = t1 + T_landing
    t_b = np.arange(start=t1,stop=t2,step=Ts)

    al = -6*(hf)/T_landing**5
    bl = 15*(hf)/T_landing**4
    cl = -10*(hf)/T_landing**3
    fl = (hf)
    tl = t_b-t1

    x_b = np.zeros(len(t_b))
    y_b = np.zeros(len(t_b))
    z_b = np.zeros(len(t_b))

    vx_b = np.zeros(len(t_b))
    vy_b = np.zeros(len(t_b))
    vz_b = np.zeros(len(t_b))

    az_b = np.zeros(len(t_b))


    x_b = x0 * np.ones(len(t_b))
    y_b = y0 * np.ones(len(t_b))
    # z_b = al * np.multiply(np.multiply(t_e,t_e),t_e) + bl * np.multiply(t_e,t_e) + cl * t_e + dl[i] * np.ones(len(t_e)) 
    z_b = al * np.multiply(np.multiply(np.multiply(np.multiply(tl,tl),tl),tl),tl) + bl * np.multiply(np.multiply(np.multiply(tl,tl),tl),tl) + cl * np.multiply(np.multiply(tl,tl),tl)+fl 
    
    # vz_b = 3 * al * np.multiply(t_e,t_e) + 2 * bl * t_e + cl * np.ones(len(t_e)) 
    vz_b = 5 * al * np.multiply(np.multiply(np.multiply(tl,tl),tl),tl) + 4 * bl * np.multiply(np.multiply(tl,tl),tl) + 3 * cl * np.multiply(tl,tl)
    
    az_b = 20 * al * np.multiply(np.multiply(tl,tl),tl) + 12 * bl * np.multiply(tl,tl) + 6 * cl * tl
        

    ref_pos_b = from_setpoints_to_ref(x_b,y_b,z_b,Ts,T_landing)
    ref_vel_b = from_setpoints_to_ref(vx_b,vy_b,vz_b,Ts,T_landing)
    ref_b = concat_element_ref(ref_pos_b,ref_vel_b)
    vref_b = from_setpoints_to_ref(vx_b,vy_b,az_b,Ts,T_landing)
    ref_full_landing = concat_ref(ref_a,ref_b)
    v_ref_landing = concat_ref(vref_a,vref_b)

    ref_full_landing = ref_full_landing.T
    v_ref_landing = v_ref_landing.T

    tt_landing = np.hstack((t_a,t_b))

    psi = 0.0
    ddx, ddy, ddz = v_ref_landing[0, :], v_ref_landing[1, :], v_ref_landing[2, :]
    thrust = np.sqrt(ddx ** 2 + ddy ** 2 + (ddz + 9.81) ** 2)
    phi = np.arcsin((ddx * np.sin(psi) - ddy * np.cos(psi)) / thrust)
    theta = np.arctan((ddx * np.cos(psi) + ddy * np.sin(psi)) / (ddz + 9.81))

    ref_landing = {}
    ref_landing = {
        "trajectory": ref_full_landing,
        "time_step": tt_landing,
        "thrust": thrust,
        "phi": phi,
        "theta": theta,
        "Nsim": tt_landing.shape[0],
        "v_ref": v_ref_landing}
    return ref_landing


if __name__=="__main__":
    # set print options for numpy
    np.set_printoptions(precision=5, suppress=True)


    ref_takeoff = generate_takeoff_traj(3.0,4.0,0.8,5,2,0.1)
    print(ref_takeoff['time_step'],ref_takeoff['trajectory'])

    ref_landing = generate_landing_traj(3.0,4.0,0.8,2,5,0.1)
    print(ref_landing['time_step'],ref_landing['trajectory'])

    ref_full = concat_reftraj(ref_takeoff, ref_landing)
    print(ref_full['time_step'],ref_full['trajectory'])
    import matplotlib.pyplot as plt

    view_pos = 1
    view_vel = 1
    view_acc = 1
    # Position plot
    if view_pos:
        fig_pos = plt.figure("Position plot",)
        dimension_traj = 3
        pos_labels = ['x','y','z']
        pos_ref_plot = {}
        # pos_plot = {}

        for m in range(dimension_traj):
            ax = fig_pos.add_subplot(dimension_traj,1,m+1)
            ax.grid(True)

            pos_ref_plot[m], = ax.plot(ref_full['time_step'], ref_full['trajectory'][m,:])

            plt.setp(pos_ref_plot[m], color='m', linestyle='--', linewidth=3.5)

            # pos_plot[m] = {}
            # for i in range(nbr_agents):
            #     pos_plot[m][i], = ax.plot(ref['time_step'], simulator['x_sim'][m,i,:])
            #     plt.setp(pos_plot[m][i], color=colors[i], linestyle='-', linewidth=3.0)
                
            if m==0:
                ax.set_title(r"Positions on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${pos_labels[m]}(t)$", usetex=False, fontsize=12)

    # Velocity plot
    if view_vel:
        fig_vel = plt.figure("Velocity plot",)
        dimension_traj = 3
        vel_labels = ['v_{x}','v_{y}','v_{z}']
        vel_ref_plot = {}
        # vel_plot = {}

        for j in range(dimension_traj):
            ax = fig_vel.add_subplot(dimension_traj,1,j+1)
            ax.grid(True)

            vel_ref_plot[j], = ax.plot(ref_full['time_step'], ref_full['trajectory'][j+3,:])

            plt.setp(vel_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

            # vel_plot[j] = {}
            # for i in range(nbr_agents):
            #     vel_plot[j][i], = ax.plot(ref['time_step'], simulator['x_sim'][j+3,i,:])
            #     plt.setp(vel_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

            if j==0:
                ax.set_title(r"Velocity on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${vel_labels[j]}(t)$", usetex=False, fontsize=12)

    # Acceleration plot
    if view_acc:
        fig_acc = plt.figure("Acceleration plot",)
        dimension_traj = 3
        acc_labels = ['a_{x}','a_{y}','a_{z}']
        acc_ref_plot = {}
        # acc_plot = {}

        for j in range(dimension_traj):
            ax = fig_acc.add_subplot(dimension_traj,1,j+1)
            ax.grid(True)

            acc_ref_plot[j], = ax.plot(ref_full['time_step'], ref_full['v_ref'][j,:])

            plt.setp(acc_ref_plot[j], color='m', linestyle='--', linewidth=3.5)

            # acc_plot[j] = {}
            # for i in range(nbr_agents):
            #     acc_plot[j][i], = ax.plot(ref['time_step'][1:len(ref['time_step'])], simulator['u_sim'][j,i,:])
            #     plt.setp(acc_plot[j][i], color=colors[i], linestyle='-', linewidth=3.0)

            if j==0:
                ax.set_title(r"Acceleration on $x$, $y$ and $z$-axis", usetex=False, fontsize=14)
            ax.set_xlabel(r"Time $t(s)$", usetex=False, fontsize=12)
            ax.set_ylabel(f"${acc_labels[j]}(t)$", usetex=False, fontsize=12)
    plt.show()