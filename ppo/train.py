import os
import json
import vessl
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from environment.env import PMSP
from ppo.agent import Agent
# from ppo.validation import evaluate


def train(args):
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)

    with open(args.save_log_dir + "parameters.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(args.save_log_dir + "train_log.csv", 'w') as f:
        f.write('episode, mean value, reward, mean tardiness, mean setup time, SSPT, ATCS, MDD, COVERT"\n')

    rule_weight = {100: {"ATCS": [2.730, 1.153], "COVERT": 6.8},
                   200: {"ATCS": [3.519, 1.252], "COVERT": 4.4},
                   400: {"ATCS": [3.338, 1.209], "COVERT": 3.9}}

    env = PMSP(num_job=args.num_job, num_m=args.num_machine,
               reward_weight=[args.weight_tard, 1 - args.weight_tard], rule_weight=rule_weight[args.num_job])
    agent = Agent(env.state_size, env.action_size, args)

    if not bool(args.vessl):
        writer = SummaryWriter(args.save_log_dir)

    T_decay = (args.T - args.T_min) / args.T_step
    for e in range(1, args.num_episodes + 1):
        agent.policy.train()
        state, mask = env.reset()

        ep_reward = 0
        ep_value = 0

        step = 0
        while True:
            action, action_logprob, state_value = agent.get_action(state, mask, train=True)
            next_state, reward, done, mask = env.step(action)

            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.logprobs.append(action_logprob)
            agent.buffer.masks.append(mask)
            agent.buffer.rewards.append(reward)
            agent.buffer.state_values.append(state_value)

            state = next_state
            ep_reward += reward
            ep_value += state_value
            step += 1

            if step % args.num_steps == 0 or done:
                agent.update(state, mask, done)

            if done:
                tardiness = env.monitor.tardiness / env.num_job
                setup_time = env.monitor.setup / env.num_job
                break

        if agent.T > args.T_min:
            agent.T -= T_decay

        print("episode: %d | mean value: %.4f | reward: %.4f | tardiness %.4f | setup time: %.4f | SSPT : %.2f, ATCS : %.2f, MDD : %.2f, COVERT : %.2f"
              % (e, ep_value / step, ep_reward, tardiness, setup_time,
                 env.action_history["SSPT"] / env.num_job,
                 env.action_history["ATCS"] / env.num_job,
                 env.action_history["MDD"] / env.num_job,
                 env.action_history["COVERT"] / env.num_job))
        with open(args.save_log_dir + "train_log.csv", 'a') as f:
            f.write('%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % (e, ep_value / step, ep_reward, tardiness, setup_time,
                                                                      env.action_history["SSPT"] / env.num_job,
                                                                      env.action_history["ATCS"] / env.num_job,
                                                                      env.action_history["MDD"] / env.num_job,
                                                                      env.action_history["COVERT"] / env.num_job))

        if bool(args.vessl):
            vessl.log(payload={"Train/MeanValue": ep_value / step,
                               "Train/Reward": ep_reward,
                               "Train/MeanTardiness": tardiness,
                               "Train/MeanSetupTime": setup_time}, step=e)
        else:
            writer.add_scalar("Train/MeanValue", ep_value / step, e)
            writer.add_scalar("Train/Reward", ep_reward, e)
            writer.add_scalar("Train/MeanTardiness", tardiness, e)
            writer.add_scalar("Train/MeanSetupTime", setup_time, e)
            writer.add_scalar("Train/SSPT", env.action_history["SSPT"] / env.num_job, e)
            writer.add_scalar("Train/ATCS", env.action_history["ATCS"] / env.num_job, e)
            writer.add_scalar("Train/MDD", env.action_history["MDD"] / env.num_job, e)
            writer.add_scalar("Train/COVERT", env.action_history["COVERT"] / env.num_job, e)

        # if e == 1 or e % args.eval_every == 0:
        #     tardiness, setup_time = evaluate(agent, args)
        #
        #     with open(args.save_log_dir + "validation_log.csv", 'a') as f:
        #         f.write('%d,%1.2f\n' % (e, average_move))
        #
        #     if bool(args.vessl):
        #         vessl.log(payload={"Validation/Move": average_move}, step=e)
        #     else:
        #         writer.add_scalar("Validation/Move", average_move, e)

        if e % args.save_every == 0:
            agent.save(e, args.save_model_dir)
