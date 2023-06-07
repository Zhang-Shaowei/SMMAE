import datetime
import os
import pprint
import time
import threading

import numpy as np
import torch
import torch as th
from types import SimpleNamespace as SN
from os.path import dirname, abspath
import torch.nn.functional as F

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from components.density import VAEDensity


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def get_adaptive_high_alpha(t_env, args):
    t_max = args.t_max
    alpha_high_value = args.alpha_high_value
    alpha_low_value = args.alpha_low_value
    return alpha_high_value - (alpha_high_value - alpha_low_value) * (1.0 * t_env / t_max)

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }


    # Default/Base scheme
    if args.use_adaptive_alpha:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "q_vals": {"vshape": (env_info["n_actions"],), "group": "agents"},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        exp_scheme = {
            # "agent_state": {"vshape": env_info["agent_state_shape"], "group": "agents"},
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "last_density_update_t_env": {"vshape": (1,), "dtype": th.int},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "actions": {"vshape": (1,), "dtype": th.long, "group": "agents"},
            "agent_id": {"vshape": (1,), "dtype": th.int, "group": "agents"},
            "reward": {"vshape": (1,), "group": "agents"},
            "original_reward": {"vshape": (1,), "group": "agents"},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        buffer = ReplayBuffer(
            scheme,
            groups,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )

        exp_buffer = ReplayBuffer(
            exp_scheme,
            groups,
            args.buffer_size,
            env_info["episode_limit"] + 1,
            preprocess=preprocess,
            device="cpu" if args.buffer_cpu_only else args.device,
        )

    else:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                              preprocess=preprocess,
                              device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    if args.mac == 'smmae_mac':
        mac = mac_REGISTRY[args.mac](buffer.scheme, exp_buffer.scheme, groups, args)
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)


    if args.use_adaptive_alpha:
        density_model = VAEDensity(env_info['n_agents'], env_info['obs_shape'], args)

        runner.setup(scheme=scheme, exp_scheme=exp_scheme, groups=groups, preprocess=preprocess, mac=mac,
                     density_model=density_model)
    else:
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    #
    if args.use_adaptive_alpha:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, exp_buffer.scheme, logger, args)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_adaptive_alpha:
        assert runner.density_model is not None
        learner.set_up_density(runner.density_model)


    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    episode_for_alpha_cnt = 0
    ce_errors_list = []

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))


    if args.use_adaptive_alpha:
        alphas = th.Tensor([0.99 for _ in range(args.n_agents)]).to(args.device)

    while runner.t_env <= args.t_max:
        if args.use_adaptive_alpha:
            episode_batch, exp_batch = runner.run(alphas=alphas, test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            exp_buffer.insert_episode_batch(exp_batch)

            episode_for_alpha_cnt += 1    # use mean of multi episodes to calculate alpha

            q_vals = episode_batch["q_vals"].float()
            terminated = episode_batch["terminated"][:, :-1].float()
            avail_actions = episode_batch["avail_actions"]
            mask = episode_batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            mac_out_detach = q_vals.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]

            enc_output, enc_slf_attn = learner.intrinsic.forward(episode_batch)
            ce_errors = []
            for agent_id in range(args.n_agents):
                ce_error_agent = F.cross_entropy(enc_output[:, agent_id, :].detach(),
                                                 cur_max_actions.reshape(-1, args.n_agents)[:, agent_id],
                                                 reduction='none').view(*mask.shape)
                masked_ce_error_agent = ce_error_agent * mask

                ce_error = masked_ce_error_agent.sum() / mask.sum()
                ce_errors.append(ce_error.item())
            ce_errors_list.append(ce_errors)

            # log ce errors data
            if (runner.t_env - last_log_T) >= args.log_interval:
                smmae_errors_cpu = np.array(ce_errors)
                logger.log_stat("pred_loss/preprocessed_pred_loss_mean", smmae_errors_cpu.mean(), runner.t_env)
                logger.log_stat("pred_loss/preprocessed_pred_loss_max", smmae_errors_cpu.max(), runner.t_env)
                logger.log_stat("pred_loss/preprocessed_pred_loss_min", smmae_errors_cpu.min(), runner.t_env)
                for i in range(args.n_agents):
                    logger.log_stat(f"pred_loss/preprocessed_pred_loss_{i}", smmae_errors_cpu[i], runner.t_env)
                alpha_cpu = alphas.to("cpu")
                for i in range(args.n_agents):
                    logger.log_stat(f"alphas_data/alpha_{i}", alpha_cpu[i], runner.t_env)


            if episode_for_alpha_cnt % args.episode_num_for_alpha == 0:        # update alpha every {} step

                ce_errors = torch.Tensor(ce_errors_list).mean(dim=0).to(args.device) # use mean for stability
                ce_errors_list = []           # clean up previous errors


                # 2 levels alpha , each agent has one
                ce_errors_cpu = ce_errors.cpu()
                adaptive_alpha_high_value = get_adaptive_high_alpha(runner.t_env, args)
                small_delta = 1.0 * (adaptive_alpha_high_value - args.alpha_low_value) / 10
                alphas_cpu = alphas.cpu()
                for i in range(len(ce_errors_cpu)):
                    if ce_errors_cpu[i] >=  args.alpha_threshold:
                        alphas[i] = args.alpha_low_value
                    else:
                        alphas[i] = min(alphas_cpu[i] + small_delta, adaptive_alpha_high_value)

        else:
            # Run for a whole episode at a time
            episode_batch = runner.run(test_mode=False)
            buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        if args.use_adaptive_alpha:
            if exp_buffer.can_sample(args.batch_size):
                replay_buffer_idx, exp_episode_sample = exp_buffer.sample(args.batch_size, need_index=True)
                # Truncate batch to only filled timesteps
                max_ep_t = exp_episode_sample.max_t_filled()
                exp_episode_sample = exp_episode_sample[:, :max_ep_t]
                if exp_episode_sample.device != args.device:
                    exp_episode_sample.to(args.device)

                batch_to_update_flag = (runner.t_env - exp_episode_sample['last_density_update_t_env']) >= args.episode_update_vae_density_interval
                exp_buffer.update_last_density_update_flag(replay_buffer_idx, runner.t_env, args.episode_update_vae_density_interval)

                learner.train_exp(episode_sample, exp_episode_sample, runner.t_env, episode, batch_to_update_flag)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T,
                              runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
