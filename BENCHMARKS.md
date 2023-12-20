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
  Time (mean ± σ):     343.7 ms ±   8.0 ms    [User: 241.6 ms, System: 104.2 ms]
  Range (min … max):   336.3 ms … 358.4 ms    10 runs
```

### CaffeNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model caffenet-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model caffenet-12
  Time (mean ± σ):     345.9 ms ±   2.0 ms    [User: 250.2 ms, System: 98.1 ms]
  Range (min … max):   343.0 ms … 349.2 ms    10 runs
```

### Emotion Ferplus

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model emotion-ferplus-8'
Benchmark 1: ./target/release/stonnx --verbose 0 --model emotion-ferplus-8
  Time (mean ± σ):     238.7 ms ±   3.4 ms    [User: 195.5 ms, System: 48.4 ms]
  Range (min … max):   232.7 ms … 244.5 ms    12 runs
```

### GoogleNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model googlenet-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model googlenet-12
  Time (mean ± σ):      1.226 s ±  0.017 s    [User: 1.504 s, System: 0.072 s]
  Range (min … max):    1.206 s …  1.248 s    10 runs
```

### GPT2

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model GPT2'
Benchmark 1: ./target/release/stonnx --verbose 0 --model GPT2
  Time (mean ± σ):      2.020 s ±  0.085 s    [User: 1.788 s, System: 0.368 s]
  Range (min … max):    1.864 s …  2.122 s    10 runs
```

### Inception

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model inception-v2-9'
Benchmark 1: ./target/release/stonnx --verbose 0 --model inception-v2-9
  Time (mean ± σ):      1.324 s ±  0.012 s    [User: 1.686 s, System: 0.109 s]
  Range (min … max):    1.307 s …  1.336 s    10 runs
```

### MobileNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model mobilenetv2-7'
Benchmark 1: ./target/release/stonnx --verbose 0 --model mobilenetv2-7
  Time (mean ± σ):     718.9 ms ±  13.1 ms    [User: 694.8 ms, System: 29.0 ms]
  Range (min … max):   705.0 ms … 742.2 ms    10 runs
```

### Mnist

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model mnist-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model mnist-12
  Time (mean ± σ):       4.3 ms ±   0.3 ms    [User: 4.5 ms, System: 1.3 ms]
  Range (min … max):     3.5 ms …   5.9 ms    485 runs
```

### ResNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model resnet50-caffe2-v1-9'
Benchmark 1: ./target/release/stonnx --verbose 0 --model resnet50-caffe2-v1-9
  Time (mean ± σ):     837.6 ms ±   8.6 ms    [User: 761.6 ms, System: 118.1 ms]
  Range (min … max):   824.5 ms … 854.3 ms    10 runs
```

### SqueezeNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model squeezenet1.0-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model squeezenet1.0-12
  Time (mean ± σ):     131.0 ms ±   3.2 ms    [User: 117.0 ms, System: 26.1 ms]
  Range (min … max):   126.3 ms … 139.3 ms    22 runs
```

### ShuffleNet

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model shufflenet-9'
Benchmark 1: ./target/release/stonnx --verbose 0 --model shufflenet-9
  Time (mean ± σ):     374.3 ms ±   4.7 ms    [User: 378.6 ms, System: 49.2 ms]
  Range (min … max):   368.2 ms … 383.2 ms    10 runs
```

### SuperResolution

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model super_resolution'
Benchmark 1: ./target/release/stonnx --verbose 0 --model super_resolution
  Time (mean ± σ):      2.384 s ±  0.030 s    [User: 2.080 s, System: 0.303 s]
  Range (min … max):    2.342 s …  2.439 s    10 runs
```

### Vgg19

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model vgg19-7'
Benchmark 1: ./target/release/stonnx --verbose 0 --model vgg19-7
  Time (mean ± σ):      3.832 s ±  0.038 s    [User: 3.352 s, System: 0.477 s]
  Range (min … max):    3.788 s …  3.908 s    10 runs
```

### ZFNet512

```bash
hyperfine --warmup 2 './target/release/stonnx --verbose 0 --model zfnet512-12'
Benchmark 1: ./target/release/stonnx --verbose 0 --model zfnet512-12
  Time (mean ± σ):     474.5 ms ±   6.2 ms    [User: 356.1 ms, System: 120.5 ms]
  Range (min … max):   469.7 ms … 490.6 ms    10 runs
```

