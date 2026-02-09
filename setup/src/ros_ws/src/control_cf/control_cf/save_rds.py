import numpy as np
import csv

def save_rds(data_destination, data):
    np.save(data_destination, data)

def save_rds_csv(data_destination, data):
    with open(data_destination, "w", newline="") as fp:
        # Create a writer object
        writer = csv.DictWriter(fp, fieldnames=data.keys())

        # Write the header row
        writer.writeheader()

        # Write the data rows
        writer.writerow(data)

        print('Done writing dict to a csv file')

# def save_data(data_destination, output_data,ref):
#     data = {'result': output_data,'parameter':ref}
#     np.save(data_destination, data)

if __name__=="__main__":

    # ############## USE FOR TESTING SAVE FUNCTION ###################
    # # replicator dynamics parameters
    # rd_params = {}
    # rd_params['ts'] = 0.1           # sampling time of replicator dynamics
    # rd_params['nbr_steps'] = 20     # total rd steps 
    # rd_params['rho'] = 0.75         # maximum distance of connectivity  
    # nbr_agents = 4
    # Nb = 300

    # ### Simulator for simulation loop ###
    # simulator = {}
    # simulator['x_sim'] = np.zeros((6, nbr_agents, Nb+1))
    # simulator['u_sim'] = np.zeros((3, nbr_agents, Nb))

    # simulator['dist_ij'] = np.zeros((nbr_agents, nbr_agents, Nb))
    # simulator['A_sim'] = np.zeros((nbr_agents, nbr_agents, Nb))
    # simulator['F_sim'] = np.zeros((nbr_agents, 3, rd_params['nbr_steps'], Nb))
    # simulator['p_sim'] = np.zeros((3, nbr_agents, rd_params['nbr_steps']+1, Nb))
    # simulator['dp_sim'] = np.zeros((3, nbr_agents, rd_params['nbr_steps'], Nb))

    # simulator['xref_sim'] = np.zeros((6, nbr_agents, Nb))


    # save_rds("./Data/Test_save_rds_00.npy", simulator)
    # save_rds_csv("./Data/Test_save_rds_00.csv", simulator)

    ######### USE FOR TESTING LOAD FUNCTION ###################

    load_data = np.load('./Data/Test_save_rds_00.npy',allow_pickle=True).item()
    print(load_data)
    print(load_data.keys())