from __future__ import print_function
import tensorflow as tf
import numpy as np
from psutil import virtual_memory
from tensorflow.python.client import device_lib
from pkg_resources import parse_version
import multiprocessing
from PIL import Image
from os import path
import subprocess
import platform
import cpuinfo
import time
import os

from ai_benchmark.utils import *

from ai_benchmark.update_utils import update_info
from ai_benchmark.config import TestConstructor
from ai_benchmark.models import *
import itertools

MAX_TEST_DURATION = 100


def run_tests(training, inference, micro, verbose, use_CPU, precision, _type, start_dir):
    print("run_tests modified !")
    testInfo = TestInfo(_type, precision, use_CPU, verbose)

    if verbose > 0:
        printTestInfo(testInfo)
        printTestStart()

    benchmark_tests = TestConstructor().getTests()
    benchmark_results = BenchmarkResults()
    public_results = PublicResults()
    # os.chdir(path.dirname(__file__))

    iter_multiplier = 1
    if precision == "high":
        iter_multiplier = 10

    if use_CPU:
        if testInfo.tf_ver_2:
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
    else:
        config = None

    for test in benchmark_tests[1:2]:

        if verbose > 0 and not (micro and len(test.micro) == 0):
            print("\n" + str(test.id) + "/" + str(len(benchmark_tests)) + ". " + test.model + "\n")
        sub_id = 1

        tf.compat.v1.reset_default_graph() if testInfo.tf_ver_2 else tf.reset_default_graph()
        session = tf.compat.v1.Session(config=config) if testInfo.tf_ver_2 else tf.Session(config=config)

        with tf.Graph().as_default(), session as sess:

            input_, output_, train_vars_ = getModelSrc(test, testInfo, sess)

            if testInfo.tf_ver_2:
                tf.compat.v1.global_variables_initializer().run()
                if test.type == "nlp-text":
                    sess.run(tf.compat.v1.tables_initializer())
            else:
                tf.global_variables_initializer().run()
                if test.type == "nlp-text":
                    sess.run(tf.tables_initializer())

            if inference or micro:

                for subTest in (test.inference if inference else test.micro):

                    time_test_started = getTimeSeconds()
                    inference_times = []

                    # for i in range(subTest.iterations * iter_multiplier):
                    for i in itertools.count(start=0):

                        if getTimeSeconds() - time_test_started < subTest.max_duration \
                                or (i < subTest.min_passes and getTimeSeconds() - time_test_started < MAX_TEST_DURATION) \
                                or precision == "high":

                            data = loadData(test.type, subTest.getInputDims())
                            time_iter_started = getTimeMillis()
                            sess.run(output_, feed_dict={input_: data})
                            inference_time = getTimeMillis() - time_iter_started
                            inference_times.append(inference_time)

                            if verbose > 1:
                                print("Inference Time: " + str(inference_time) + " ms")

                    time_mean, time_std = computeStats(inference_times)

                    public_id = "%d.%d" % (test.id, sub_id)
                    public_results.test_results[public_id] = Result(time_mean, time_std)

                    benchmark_results.results_inference.append(time_mean)
                    benchmark_results.results_inference_norm.append(float(subTest.ref_time) / time_mean)

                    if verbose > 0:
                        prefix = "%d.%d - inference" % (test.id, sub_id)
                        printTestResults(prefix, subTest.batch_size, subTest.getInputDims(), time_mean, time_std, verbose)
                        sub_id += 1

        sess.close()

    testInfo.results = benchmark_results
    public_results = printScores(testInfo, public_results)

    os.chdir(start_dir)
    return public_results


class AIBenchmark:

    def __init__(self, use_CPU=None, verbose_level=1):

        self.tf_ver_2 = parse_version(tf.__version__) > parse_version('1.99')
        self.verbose = verbose_level

        if verbose_level > 0:
            printIntro()

        np.warnings.filterwarnings('ignore')

        try:

            if verbose_level < 3:

                os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

                if self.tf_ver_2:
                    import logging
                    logger = tf.get_logger()
                    logger.disabled = True
                    logger.setLevel(logging.ERROR)

                elif parse_version(tf.__version__) > parse_version('1.13'):
                    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

                else:
                    tf.logging.set_verbosity(tf.logging.ERROR)

            else:

                if self.tf_ver_2:
                    import logging
                    logger = tf.get_logger()
                    logger.disabled = True
                    logger.setLevel(logging.INFO)

                elif parse_version(tf.__version__) > parse_version('1.13'):
                    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

                else:
                    tf.logging.set_verbosity(tf.logging.INFO)

        except:
            pass

        np.random.seed(42)
        self.cwd = path.dirname(__file__)

        self.use_CPU = False
        if use_CPU:
            self.use_CPU = True

    def run(self, precision="normal"):
        return run_tests(training=True, inference=True, micro=False, verbose=self.verbose,
                         use_CPU=self.use_CPU, precision=precision, _type="full", start_dir=self.cwd)

    def run_inference(self, precision="normal"):
        return run_tests(training=False, inference=True, micro=False, verbose=self.verbose,
                         use_CPU=self.use_CPU, precision=precision, _type="inference", start_dir=self.cwd)

    def run_training(self, precision="normal"):
        return run_tests(training=True, inference=False, micro=False, verbose=self.verbose,
                         use_CPU=self.use_CPU, precision=precision, _type="training", start_dir=self.cwd)

    def run_micro(self, precision="normal"):
        return run_tests(training=False, inference=False, micro=True, verbose=self.verbose,
                         use_CPU=self.use_CPU, precision=precision, _type="micro", start_dir=self.cwd)


if __name__ == "__main__":
    benchmark = AIBenchmark(use_CPU=False, verbose_level=2)
    results = benchmark.run_inference(precision="normal")
