import time
import torch
import numpy as np


def rollout(actor_crit, env, N_rollout=10_000): 
    #save the following (use .append)
    Start_state = [] #hold an array of (x_t)
    Actions = [] #hold an array of (u_t)
    Rewards = [] #hold an array of (r_{t+1})
    End_state = [] #hold an array of (x_{t+1})
    Terminal = [] #hold an array of (terminal_{t+1})
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        obs, info = env.reset() 
        for i in range(N_rollout): 
            # action = np.random.choice(env.action_space.n, p=pi(obs)) #sample action from policy
            action = np.random.choice(actor_crit.num_actions, p=pi(obs)) #sample action from policy

            Start_state.append(obs) 
            Actions.append(action)

            obs_next, reward, terminated, truncated, info = env.step(action)

            Terminal.append(terminated)
            Rewards.append(reward) 
            End_state.append(obs_next) 

            if terminated or truncated: 
                obs, info = env.reset() 
            else:
                obs = obs_next
                
    #error checking:
    assert len(Start_state)==len(Actions)==len(Rewards)==len(End_state)==len(Terminal), f'error in lengths: {len(Start_state)}=={len(Actions)}=={len(Rewards)}=={len(End_state)}=={len(Terminal)}'
    return np.array(Start_state), np.array(Actions), np.array(Rewards), np.array(End_state), np.array(Terminal).astype(int)


def A2C_rollout(actor_crit, optimizer, env, alpha_actor=0.5, alpha_entropy=0.5, gamma=0.98, \
                N_iterations=10, N_rollout=20000, N_epochs=10, batch_size=32, N_evals=10, best_score=-float('inf')):
    curr_best = best_score
    torch.save(actor_crit.state_dict(),'./models/actor-crit-checkpoint')
    try:
        for iteration in range(N_iterations):
            print(f'rollout iteration {iteration}')
            
            #2. rollout 
            Start_state, Actions, Rewards, End_state, Terminal = rollout(actor_crit, env, N_rollout=N_rollout)
            
            #Data conversion, no changes required
            convert = lambda x: [torch.tensor(xi,dtype=torch.float32) for xi in x]
            Start_state, Rewards, End_state, Terminal = convert([Start_state, Rewards, End_state, Terminal])
            Actions = Actions.astype(int)

            print('starting training on rollout information...')
            for epoch in range(N_epochs): 
                for i in range(batch_size,len(Start_state)+1,batch_size): 
                    Start_state_batch, Actions_batch, Rewards_batch, End_state_batch, Terminal_batch = \
                    [d[i-batch_size:i] for d in [Start_state, Actions, Rewards, End_state, Terminal]]
                    
                    #Advantage:
                    Vnow = actor_crit.critic(Start_state_batch) #c=)
                    Vnext = actor_crit.critic(End_state_batch) #c=)
                    A = Rewards_batch + gamma*Vnext*(1-Terminal_batch) - Vnow #c=)
                    
                    # # action_index = np.stack((np.arange(batch_size),Actions_batch),axis=0) #to filter actions
                    # logp = actor_crit.actor(Start_state_batch,return_logp=True) #c=)
                    # logp_cur = logp[np.arange(batch_size), Actions_batch] #c=)
                    # p = torch.exp(logp) #c=) #probability for with all actions in a list
                    # # p_cur = torch.exp(logp_cur) #c=) #probability for choosing the current chosen action
                    
                    # or
                    action_index = np.stack((np.arange(batch_size),Actions_batch),axis=0) # Filtering actions
                    logp = actor_crit.actor(Start_state_batch,return_logp=True)[action_index] 
                    p = torch.exp(logp) 
                    
                    L_value_function = torch.mean(A**2) #c=)
                    L_policy = -(A.detach()*logp).mean() #c=) #detach A, the gradient should only to through logp
                    L_entropy = -torch.mean((-p*logp),0).sum() #c=) 
                    
                    Loss = L_value_function + alpha_actor*L_policy + alpha_entropy*L_entropy #c=) 
                    
                    optimizer.zero_grad()
                    Loss.backward()
                    optimizer.step()
                
                print(f'logp{p[0]} logp{logp.shape}')

                score = np.mean([eval_actor(actor_crit, env) for i in range(N_evals)])

                print(f'iteration={iteration} epoch={epoch} Average Reward per episode:',score)
                print('\t Value loss:  ',L_value_function.item())
                print('\t Policy loss: ',L_policy.item())
                print('\t Entropy:     ',-L_entropy.item())
                
                if score > curr_best:
                    curr_best = score
                    print('################################# \n new best', curr_best, 'saving actor-crit... \n#################################')
                    torch.save(actor_crit.state_dict(),'./models/actor-crit-checkpoint')
            
            print('loading best result')
            actor_crit.load_state_dict(torch.load('./models/actor-crit-checkpoint'))
    except Exception as e:
        print(e)
    finally: #this will always run even when using the a KeyBoard Interrupt. 
        print('loading best result')
        actor_crit.load_state_dict(torch.load('./models/actor-crit-checkpoint'))

        return curr_best


def eval_actor(actor_crit, env):
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        rewards_acc = 0 
        obs, info = env.reset() 
        while True: 
            action = np.argmax(pi(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            rewards_acc += reward 
            if terminated or truncated: 
                return rewards_acc 


def show(actor_crit,env):
    pi = lambda x: actor_crit.actor(torch.tensor(x[None,:],dtype=torch.float32))[0].numpy()
    with torch.no_grad():
        try:
            obs, info = env.reset() 
            env.render() 
            time.sleep(1) 
            while True: 
                action = np.argmax(pi(obs))
                obs, reward, terminated, truncated, info = env.step(action) 
                # time.sleep(1/60) 
                env.render()
                if terminated or truncated: 
                    time.sleep(0.5) 
                    break  
        finally: #this will always run even when an error occurs
            env.close()