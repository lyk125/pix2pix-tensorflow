from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
import os
import sys
import time


def run(cmd, image="affinelayer/pix2pix-tensorflow"):
    docker = "docker"
    if sys.platform.startswith("linux"):
        docker = "nvidia-docker"

    datapath = os.path.abspath("../data")
    prefix = [docker, "run", "--rm", "--volume", os.getcwd() + ":/prj", "--volume", datapath + ":/data", "--workdir", "/prj", "--env", "PYTHONUNBUFFERED=x", "--volume", "/tmp/cuda-cache:/cuda-cache", "--env", "CUDA_CACHE_PATH=/cuda-cache", image]
    args = prefix + cmd.split(" ")
    print(" ".join(args))
    subprocess.check_call(args)


def main():
    start = time.time()

    # train + test
    for direction in ["AtoB", "BtoA"]:
        for dataset in ["facades", "maps"]:
            name = dataset + "_" + direction
            run("python pix2pix.py --mode train --output_dir test/%s_train --max_steps 1 --input_dir /data/official/%s/train --which_direction %s --seed 0" % (name, dataset, direction))
            run("python pix2pix.py --mode test --output_dir test/%s_test --max_steps 1 --input_dir /data/official/%s/val --which_direction %s --seed 0 --checkpoint test/%s_train" % (name, dataset, direction, name))

        # test lab colorization
        dataset = "color-lab"
        name = dataset + "_" + direction
        run("python pix2pix.py --mode train --output_dir test/%s_train --max_steps 1 --input_dir /data/%s/train --which_direction %s --seed 0 --lab_colorization" % (name, dataset, direction))
        run("python pix2pix.py --mode test --output_dir test/%s_test --max_steps 1 --input_dir /data/%s/val --seed 0 --checkpoint test/%s_train" % (name, dataset, name))

    for dataset, direction in [("facades", "BtoA"), ("edges2shoes", "AtoB"), ("maps", "AtoB"), ("maps", "BtoA"), ("cityscapes", "AtoB"), ("cityscapes", "BtoA"), ("edges2handbags", "AtoB")]:
        name = dataset + "_" + direction
        run("python pix2pix.py --mode test --output_dir test/%s_pretrained_test --input_dir /data/official/%s/val --max_steps 100 --which_direction %s --seed 0 --checkpoint /data/pretrained/%s" % (name, dataset, direction, name))
        run("python pix2pix.py --mode export --output_dir test/%s_pretrained_export --checkpoint /data/pretrained/%s" % (name, name))

    # test python3
    run("python pix2pix.py --mode train --output_dir test/py3_facades_AtoB_train --max_steps 1 --input_dir /data/official/facades/train --which_direction AtoB --seed 0", image="tensorflow/tensorflow:0.12.1-gpu-py3")
    run("python pix2pix.py --mode test --output_dir test/py3_facades_AtoB_test --max_steps 1 --input_dir /data/official/facades/val --which_direction AtoB --seed 0 --checkpoint test/py3_facades_AtoB_train", image="tensorflow/tensorflow:0.12.1-gpu-py3")

    print("elapsed", int(time.time() - start))
    # currently 1817


main()
