#!/media/data/.env_37_gpu/bin/python3.7

import ai_benchmark
benchmark = ai_benchmark.AIBenchmark(use_CPU=False, verbose_level=2)
results = benchmark.run_inference()


