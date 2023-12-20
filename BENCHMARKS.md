# Benchmarks

All benchmarks were done on the following hardware:

- CPU: Intel Core i7-12700k
- RAM: 32 GB DDR4 3600 MHz

And on the following software:

- Ubuntu 22.04.3 LTS on WSL
- rustc 1.74.1 (a28077b28 2023-12-04) 

### AlexNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model bvlcalexnet-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model bvlcalexnet-12
  Time (mean ± σ):     314.1 ms ±  13.1 ms    [User: 283.1 ms, System: 219.8 ms]
  Range (min … max):   299.1 ms … 345.5 ms    10 runs
```

### CaffeNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model caffenet-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model caffenet-12
  Time (mean ± σ):     325.7 ms ±   3.7 ms    [User: 273.8 ms, System: 235.4 ms]
  Range (min … max):   320.1 ms … 331.4 ms    10 runs
```

### Emotion Ferplus

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model emotion-ferplus-8'
Benchmark 1: ./target/release/stonnx --verbose 0 --model emotion-ferplus-8
  Time (mean ± σ):     291.9 ms ±   7.1 ms    [User: 291.7 ms, System: 178.2 ms]
  Range (min … max):   282.3 ms … 306.8 ms    10 runs
```

### GoogleNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model googlenet-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model googlenet-12
  Time (mean ± σ):      1.361 s ±  0.071 s    [User: 11.179 s, System: 1.323 s]
  Range (min … max):    1.252 s …  1.445 s    10 runs
```

### GPT2

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model GPT2'
Benchmark 1: ./target/release/stonnx --verbose 0 --model GPT2
  Time (mean ± σ):     490.1 ms ±  21.3 ms    [User: 352.5 ms, System: 530.9 ms]
  Range (min … max):   466.6 ms … 534.4 ms    10 runs
```

### Inception

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model inception-v2-9'
Benchmark 1: ./target/release/stonnx --verbose 0 --model inception-v2-9
  Time (mean ± σ):     866.6 ms ±  17.4 ms    [User: 3299.5 ms, System: 1260.5 ms]
  Range (min … max):   834.3 ms … 886.2 ms    10 runs
```

### MobileNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model mobilenetv2-7'
Benchmark 1: ./target/release/stonnx --verbose 0 --model mobilenetv2-7
  Time (mean ± σ):     786.6 ms ±   7.1 ms    [User: 797.9 ms, System: 191.0 ms]
  Range (min … max):   780.1 ms … 800.7 ms    10 runs
```

### Mnist

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model mnist-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model mnist-12
  Time (mean ± σ):       7.3 ms ±   0.9 ms    [User: 12.1 ms, System: 6.2 ms]
  Range (min … max):     5.5 ms …  12.5 ms    229 runs
```

### ResNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model resnet50-caffe2-v1-9'
Benchmark 1: ./target/release/stonnx --verbose 0 --model resnet50-caffe2-v1-9
  Time (mean ± σ):      1.089 s ±  0.013 s    [User: 1.382 s, System: 0.651 s]
  Range (min … max):    1.056 s …  1.101 s    10 runs
```

### SqueezeNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model squeezenet1.0-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model squeezenet1.0-12
  Time (mean ± σ):     205.8 ms ±   3.6 ms    [User: 322.6 ms, System: 436.8 ms]
  Range (min … max):   200.0 ms … 212.9 ms    14 runs
```

### ShuffleNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model shufflenet-9'
Benchmark 1: ./target/release/stonnx --verbose 0 --model shufflenet-9
  Time (mean ± σ):     475.0 ms ±   8.5 ms    [User: 647.9 ms, System: 318.3 ms]
  Range (min … max):   463.4 ms … 487.7 ms    10 runs
```

### SuperResolution

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model super_resolution'
Benchmark 1: ./target/release/stonnx --verbose 0 --model super_resolution
  Time (mean ± σ):      2.397 s ±  0.030 s    [User: 2.132 s, System: 0.395 s]
  Range (min … max):    2.374 s …  2.474 s    10 runs
```

### Vgg19

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model vgg19-7'
Benchmark 1: ./target/release/stonnx --verbose 0 --model vgg19-7
  Time (mean ± σ):      4.057 s ±  0.061 s    [User: 3.878 s, System: 2.102 s]
  Range (min … max):    3.964 s …  4.141 s    10 runs
```

### ZFNet512

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model zfnet512-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model zfnet512-12
  Time (mean ± σ):     451.9 ms ±  14.3 ms    [User: 451.3 ms, System: 472.5 ms]
  Range (min … max):   441.4 ms … 489.8 ms    10 runs
```

