import time
import torch
import numpy as np

from PPO import PPO
import pickle as pkl
from Classes.TabAdvEnv import *
# prepare data and models
from Utils.data_utils import preprocess_credit, drop_corolated_target, split_to_datasets, preprocess_top100, preprocess_HCDR, \
                            preprocess_ICU, preprocess_HATE,rearrange_columns, rearrange_columns_edittable, over_sampling, \
                            write_edditable_file

from Utils.models_utils import load_target_models, load_surrogate_model, train_GB_model, train_LGB_model, train_RF_model, train_XGB_model,  \
                            train_REGRESSOR_model, compute_importance

from Utils.attack_utils import get_attack_set, get_balanced_attack_set

#################################### Testing ###################################
# np.array(x_adv.iloc[50:51])), y_adv[50]
def test(target_model, x_adv, y_adv, raw_data_path, version):
    # Set parameters


    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # env_name = "TabularAdv-v2"
    # # env_name = "RoboschoolWalker2d-v1"
    # has_continuous_action_space = True
    # max_ep_len = 1000           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    # render = True              # render environment on screen
    # frame_delay = 0             # if required; add delay b/w frames
    #
    # total_test_episodes = 10    # total num of testing episodes

    # K_epochs = 80               # update policy for K epochs
    # eps_clip = 0.2              # clip parameter for PPO
    # gamma = 0.99                # discount factor
    #
    # lr_actor = 0.0003           # learning rate for actor
    # lr_critic = 0.001           # learning rate for critic

    ########################### Our Env ##########################
    env_name = f'TabularAdv-v{version}'  # environment name
    has_continuous_action_space = True  # False

    max_ep_len = 400  # max timesteps in one episode
    # max_training_timesteps = int(1e5)  # break training loop if timeteps > max_training_timesteps (100000)

    # print_freq = max_ep_len * 4  # print avg reward in the interval (in num timesteps)
    # log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    # save_model_freq = int(2e4)  # save model frequency (in num timesteps)

    # TODO: define action_std
    action_std = 0.5  #
    # update_timestep = max_ep_len * 4  # update policy every n timesteps
    K_epochs = 40  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    # lr_actor = 0.0003       # learning rate for actor network
    # lr_critic = 0.001       # learning rate for critic network
    lr_actor = 0.6  # learning rate for actor network
    lr_critic = 0.6  # learning rate for critic network

    random_seed = 0
    # render = True  # render environment on screen
    render = False  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 20  # total num of testing episodes
########################################################################
    # env = gym.make(env_name)
    env = TabAdvEnv(GB,x_adv, y_adv, raw_data_path)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            # YAEL
            adv_label = target_model.predict(state.reshape(1, -1))
            #print("adv_label: {}, orig label: {}".format(adv_label, y_adv))
            success = np.equal(adv_label, np.abs(1-y_adv))
            #print("adv sample: ", state)

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break
        if done:
            break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

    return success, state

if __name__ == '__main__':

    configurations = get_config()
    data_path = configurations["data_path"]
    raw_data_path = configurations["raw_data_path"]
    perturbability_path = configurations["perturbability_path"]
    results_path = configurations["results_path"]
    seed = int(configurations["seed"])
    dataset_name = raw_data_path.split("/")[1]
    models_path = configurations['models_path']
    exp_type = configurations['exp_type']

    # Get dataset
    datasets = split_to_datasets(raw_data_path, save_path=data_path)

    # Get scalers
    # scaler = pickle.load(open(data_path+"/scaler.pkl", 'rb' ))
    # scaler_pt = pickle.load(open(data_path+"/scaler_pt.pkl", 'rb' ))
    # constraints, perturbability = get_constraints(dataset_name, perturbability_path)

    columns_names = list(datasets.get('x_test').columns)

    # process(dataset_name, raw_data_path, TorchMinMaxScaler)
    # train_models(data_path, datasets)
    # train_REG_models(dataset_name, data_path, datasets)
    x_adv = datasets.get('x_test')  # first sample
    y_adv = datasets.get('y_test')  # first sample
    # Get models
    GB, LGB, XGB, RF = load_target_models(data_path, models_path)
    target_models = [GB, XGB, LGB, RF]
    scaler = None

    # attack_x_clean, attack_y_clean = get_attack_set(datasets, target_models, None, scaler ,data_path)
    attack_x_clean = pd.read_csv(open(data_path + "/x_attack_clean", 'rb'))
    attack_y_clean = pd.read_csv(open(data_path + "/y_attack_clean", 'rb'))
    attack_size = 300
    x_adv, attack_y = get_balanced_attack_set(dataset_name, attack_x_clean, attack_y_clean, attack_size, seed)
    y_adv = attack_y.transpose().values.tolist()[0]

    advs = []
    success_rate = 0
    success_TOATL = 0
    ind = [0,1,2,3,4,8,9,10,455,456,457,458,459] #train set
    

    #===== test set of GB =====#

    for i in range (10,70): #class 0
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(GB,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL += 1
        
        advs.append(state)
    print("success rate for class 0: ", success_rate/60)
    success_rate = 0
    for i in range (460,520): #class 1
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(GB,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL   += 1
        
        advs.append(state)
    print("success rate for class 1: ", success_rate/60)

    print("success rate TOTAL: ", success_TOATL/120)
    

    #===== transfer to LGB =====#

    ### test set ###
    for i in range (10,70): #class 0
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(LGB,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL += 1
        
        advs.append(state)
    print("success rate for class 0: ", success_rate/60)
    success_rate = 0
    for i in range (460,520): #class 1
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(LGB,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL   += 1
        
        advs.append(state)
    print("success rate for class 1: ", success_rate/60)

    print("success rate TOTAL: ", success_TOATL/120)

    #===== transfer to XGB =====#

    ### test set ###
    for i in range (10,70): #class 0
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(XGB,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL += 1
        
        advs.append(state)
    print("success rate for class 0: ", success_rate/60)
    success_rate = 0
    for i in range (460,520): #class 1
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(XGB,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL   += 1
        
        advs.append(state)
    print("success rate for class 1: ", success_rate/60)

    print("success rate TOTAL: ", success_TOATL/120)

#===== transfer to RF =====#

    ### test set ###
    for i in range (10,70): #class 0
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(RF,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL += 1
        
        advs.append(state)
    print("success rate for class 0: ", success_rate/60)
    success_rate = 0
    for i in range (460,520): #class 1
        # np.array(x_adv.iloc[50:51])), y_adv[50]
        # test(target_model, x_adv, y_adv, raw_data_path, version):
        success,state = test(RF,torch.from_numpy(np.array(x_adv.iloc[i:i+1])), y_adv[i], raw_data_path, 1008)
        if success:
            success_rate += 1
            success_TOATL   += 1
        
        advs.append(state)
    print("success rate for class 1: ", success_rate/60)

    print("success rate TOTAL: ", success_TOATL/120)