# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import argparse
import ray
import os
import time

from pathlib import Path
import gym

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.scenario import Scenario

from benchmark.agents import load_config
from benchmark.agents.sac.sac_agent import SacdAgent


RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{n_agent}"


def main(
    scenario,
    config_file,
    log_dir,
    restore_path=None,
    num_workers=1,
    horizon=1000,
    paradigm="decentralized",
    headless=False,
    cluster=False,
):
    if cluster:
        ray.init(address="auto", redis_password="5241590000000000")
        print(
            "--------------- Ray startup ------------\n{}".format(
                ray.state.cluster_resources()
            )
        )
    scenario_path = Path(scenario).absolute()
    agent_missions_count = Scenario.discover_agent_missions_count(scenario_path)
    if agent_missions_count == 0:
        agent_ids = ["default_policy"]
    else:
        agent_ids = [f"AGENT-{i}" for i in range(agent_missions_count)]

    config = load_config(config_file)
    agents = {
        agent_id: AgentSpec(
            **config["agent"], interface=AgentInterface(**config["interface"])
        )
        for agent_id in agent_ids
    }

    env = gym.make(
        "smarts.env:hiway-v0",
        seed=42,
        scenarios=[str(scenario_path)],
        headless=headless,
        agent_specs=agents,)

    obs_space, act_space = config["policy"][1:3]
    SacConfig = config["run"]["config"]
    log_dir = os.path.join(log_dir, 'sac', time.strftime("%d-%m-%Y_%H-%M-%S"))

    SacAgent = SacdAgent(env=env, test_env=env, obs_space=obs_space, act_space=act_space, log_dir=log_dir,
                         agent_ids=agent_ids, config=config, cuda=args.cuda, seed=args.seed, **SacConfig)

    SacAgent.run()


def parse_args():
    parser = argparse.ArgumentParser("Benchmark learning")
    parser.add_argument("scenario", type=str, help="Scenario name",)
    parser.add_argument("--paradigm", type=str, default="decentralized",
                        help="Algorithm paradigm, decentralized (default) or centralized",)
    parser.add_argument("--headless", default=False, action="store_true", help="Turn on headless mode")
    parser.add_argument("--log_dir",  default="./log/results", type=str,
                        help="Path to store RLlib log and checkpoints, default is ./log/results",)
    parser.add_argument("--config_file", "-f", type=str, required=True)
    parser.add_argument("--restore_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1, help="RLlib num workers")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--horizon", type=int, default=1000, help="Horizon for a episode")
    parser.add_argument("--cuda", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_file=args.config_file,
        log_dir=args.log_dir,
        restore_path=args.restore_path,
        num_workers=args.num_workers,
        horizon=args.horizon,
        paradigm=args.paradigm,
        headless=args.headless,
        cluster=args.cluster,
    )
