import os
import json
import vessl
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from new_version.environment.env import PMSP
from new_version.ppo.agent import Agent
# from ppo.validation import evaluate


def train(args):
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)

    with open(args.save_log_dir + "parameters.json", 'w') as f:
        json.dump(vars(args), f, indent=4)

    with open(args.save_log_dir + "train_log.csv", 'w') as f:
        f.write('episode, mean value, reward, mean tardiness, mean setup time\n')

    env = PMSP(num_job=args.num_job, num_m=args.num_machine,
               reward_weight=[args.weight_tard, 1 - args.weight_tard])
    agent = Agent(env.state_size, env.action_size, args)

    if not bool(args.vessl):
        writer = SummaryWriter(args.save_log_dir)

    for e in range(1, args.num_episodes + 1):
        agent.policy.train()
        state, mask = env.reset()

        ep_reward = 0
        ep_value = 0

        step = 0
        while True:
            action, action_logprob, state_value = agent.get_action(state, mask, train=True)
            next_state, reward, done, next_mask = env.step(action)

            agent.buffer.states.append(state)
            agent.buffer.actions.append(action)
            agent.buffer.logprobs.append(action_logprob)
            agent.buffer.masks.append(mask)
            agent.buffer.rewards.append(reward)
            agent.buffer.state_values.append(state_value)

            state = next_state
            mask = next_mask
            ep_reward += reward
            ep_value += state_value
            step += 1

            if step % args.num_steps == 0 or done:
                agent.update(state, mask, done)

            if done:
                tardiness = env.monitor.tardiness / env.num_job
                setup_time = env.monitor.setup / env.num_job
                break

        print("episode: %d | mean value: %.4f | reward: %.4f | tardiness %.4f | setup time: %.4f"
              % (e, ep_value / step, ep_reward, tardiness, setup_time))
        with open(args.save_log_dir + "train_log.csv", 'a') as f:
            f.write('%d,%.2f,%.2f,%.2f,%.2f\n' % (e, ep_value / step, ep_reward, tardiness, setup_time))

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
