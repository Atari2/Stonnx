# Stonnx

### Main Contributors:
- [s317659 Rosiello Alessio](https://github.com/Atari2)
- [s317661 Tcaciuc Claudiu Constantin](https://github.com/ClaudiuTcaciuc)

Il nome deriva dalla fusione di [Steelix](https://wiki.pokemoncentral.it/Steelix) (pokemon di metallo, scelto perchè il progetto è scritto in Rust) e ONNX. Inoltre è anche un gioco di parole con [stonks](https://www.icbz4.it/alunnifermi/wp-content/uploads/2019/12/significato-stonks-1024x766.jpg).

### Descrizione progetto:

Il progetto consiste nella realizzazione di un interprete ONNX utilizzando il linguaggio Rust. Le richieste da rispettare sono le seguenti:
- Creazione di un parser per estrarre dal file ONNX l'informazione necessaria per la creazione della rete;
- Implementazione di un sotto-set di operatori ONNX;
- Utilizzo di parallelismo per l'esecuzione della rete;
- (Opzionale) binding con altri linguaggi.

### Modalità di utilizzo:
- Eseguire il comando `cargo build --release` per compilare il progetto;
  - di default il progetto viene compilato utilizzando il parallelismo con `rayon` (per compilare con il parallelismo da noi implementato utilizzare il comando `cargo build --features custom-threadpool --release`);

- Eseguire il comando `cargo run --release -- --model <modelname>`

  - `modelname` è il percorso alla cartella contenente il modello. Nel caso il percorso sia relativo, verrà assunto relativo a `$pwd/models` dove `$pwd` rappresenta la cartella in cui risiede l'eseguibile. Nella cartello contenente il modello ci dovrà essere un file `inputs.json` con il seguente schema: 

    ```json
    {
        "inputs": [
            "<percorso verso input del modello>",
            ...
        ],
        "outputs": [
            "<percorso verso output attesi del modello>",
            ...
        ],
        "modelpath": "<percorso al file .onnx contenente il modello>"
    }
    ```

    tutti i percorsi all'interno di questo file possono essere relativi o assoluti, se relativi, saranno assunti relativi a `$pwd/models/$modelname`.

- Nel caso in cui si volesse visualizzare l'esecuzione degli operatori è possibile aggiungere l'opzione `--verbose` al comando precedente (di default verbose è 0).
  - Esempio: `cargo run --release -- --model <modelname> --verbose 1`
  - `verbose = 0`: non visualizza nulla
  - `verbose = 1`: visualizza informazioni riguardanti l'esecuzione degli operatori
  - `verbose = 2`: inserisce gli output di ogni operatore in un file `.npy`
  - `verbose = 4`: inserisce anche gli output intermedi di ogni operatore in un file `.npy`

- Inoltre può essere aggiunto il comando `--gengraph` per generare file che poi possono essere usati per disegnare grafi dei modelli. Il programma può generare il grafo in un formato proprietario `json` (che può essere letto da questo [tool](https://github.com/Atari2/ONNXGraphLayout)) oppure in generico formato `dot` (da usare con [graphviz](https://graphviz.org/)). Il formato del grafo può essere controllato dall'opzione `--graphtype` che può essere `json` o `dot` (default: `json`). Il file generato sarà posto nella stessa cartella in cui risiede il modello eseguito.

- Eseguire il comando `cargo doc --open` per visualizzare la documentazione del progetto.

### Modelli supportati:
I modelli testati fanno riferimento a quelli presenti nella sezione [archive](https://github.com/onnx/models/tree/main/archive) del [repository ufficiale di ONNX](https://github.com/onnx/models), in quanto a inizio Dicembre 2023 sono stati aggiornati e aggiunti nuovi modelli, ma lo sviluppo di questo programma è cominciato molto prima dell'aggiornamento del Model Zoo. I modelli testati sono i seguenti:
- [AlexNet](https://github.com/onnx/models/tree/main/archive/vision/classification/alexnet)
- [MobileNet](https://github.com/onnx/models/tree/main/archive/vision/classification/mobilenet)
- [GoogleNet](https://github.com/onnx/models/tree/main/archive/vision/classification/inception_and_googlenet/googlenet)
- [ResNet](https://github.com/onnx/models/tree/main/archive/vision/classification/resnet)
- [GPT2](https://github.com/onnx/models/tree/main/archive/text/machine_comprehension/gpt-2)
- [Emotion](https://github.com/onnx/models/tree/main/archive/vision/body_analysis/emotion_ferplus)
- [CaffeNet](https://github.com/onnx/models/tree/main/archive/vision/classification/caffenet)
- [Inception](https://github.com/onnx/models/tree/main/archive/vision/classification/inception_and_googlenet/inception_v2)
- [Mnist](https://github.com/onnx/models/tree/main/archive/vision/classification/mnist)
- [SqueezeNet](https://github.com/onnx/models/tree/main/archive/vision/classification/squeezenet)
- [Shufflenet](https://github.com/onnx/models/tree/main/archive/vision/classification/shufflenet)
- [Super Resolution](https://github.com/onnx/models/tree/main/archive/vision/super_resolution/sub_pixel_cnn_2016)
- [VGG](https://github.com/onnx/models/tree/main/archive/vision/classification/vgg)
- [ZFNet](https://github.com/onnx/models/tree/main/archive/vision/classification/zfnet-512)

Nota: durante la scelta dei modelli è stato selezionata la versione più recente presente nella sezione [archive](https://github.com/onnx/models/tree/main/archive) del repository di ONNX.

### Crate più importanti utilizzati:
Sono state utilizzate le seguenti crate:
- [ndarray](https://crates.io/crates/ndarray): utilizzato per la gestione degli array multidimensionali;
- [anyhow](https://crates.io/crates/anyhow): utilizzato per la gestione degli errori;
- [clap](https://crates.io/crates/clap): utilizzato per la gestione degli argomenti da linea di comando;
- [bytemuck](https://crates.io/crates/bytemuck): utilizzato per la conversione di tipi;
- [petgraph](https://crates.io/crates/petgraph): utilizzato per la creazione del grafo della rete;
- [serde](https://crates.io/crates/serde): utilizzato per la serializzazione e deserializzazione di strutture dati;
- [protobuf](https://crates.io/crates/protobuf): utilizzato per la gestione dei file protobuf;

### Descrizione varie parti del progetto:
- `src/main.rs`: file principale del progetto, contiene la funzione main e la gestione degli argomenti da linea di comando;
- `src/operators`: contiene i file con le implementazioni degli operatori ONNX;
  - tra gli operatori implementati si possono trovare (ma non solo): `Add`, `AveragePool`, `BatchNormalization`, `Concat`, `Conv`, `Dropout`, `Flatten`, `Gemm`, `GlobalAveragePool`, `MaxPool`, `MatMul`, `Mul`, `Relu`, `Reshape`, `Softmax`, `Sum`, `Transpose`;
- `src/onnxparser`: contiene i file con l'implementazione del parser per estrarre le informazioni dal file ONNX;
  - I file in questa cartella vengono generati a tempo di build (vedere `build.rs`) dal compilatore di protobuf utilizzando la libreria `protobuf_codegen`. 

- `src/executor`: contiene l'implementazione per l'esecuzione della rete, ovvero la creazione del grafo e l'esecuzione degli operatori;
- `src/parallel`: contiene i file per l'implementazione del parallelismo;
- `src/protograph`: contiene i file per l'implementazione della creazione di un file `.json` contenente il grafo della rete;
- `src/protos`: contiene il file `onnx.proto` utilizzato per la creazione del file `.rs` contenente le strutture dati per la gestione dei file protobuf;
- `src/common/mod.rs`: contiene le strutture dati utilizzate per la gestione dei file ONNX;
  - gestione del `verbose`;
  - gestione delle path per i file input e output dei vari modelli;
  - gestione dei file `.json` contenenti il grafo della rete;
  - gestione del `opset_version` degli operatori;
  - gestione dei risultati ottenuti dall'esecuzione di un operatore;
  - gestione dei tipi di dato;
  - gestione di un tratto per la rappresentazione dei tensori (`ArrayElement`);
  - gestione della lettura dei dati da formato binario, per la creazione dei tensori;
- `src/utils`: gestione di operazioni utili per la creazione dei tensori e la gestione di questi ultimi;

### Architettura del programma:

Il programma ha quattro step principali:

- Parsing e lettura del file .onnx, dei suoi input provenienti dall'esterno e inizializzazione dei tensori iniziali con questi ultimi.
- Creazione di 2 grafi, rappresentati da due HashMap, uno che connette ogni operatore ai suoi input e uno che connette ogni input (tensore) agli operatori in cui viene usato, praticamente l'inverso del primo. Avere queste due rappresentazioni ci permette quindi di avere un grafo delle dipendenze di ciascun operatore, che verrà poi usato nell'esecuzione del modello stesso. Infatti, quando un operatore avrà soddisfatto tutte le sue dipendenze (e.g. i suoi input verranno prodotti da un operatore precedente), esso potrà essere messo in coda per l'esecuzione nel thread pool. 
- Esecuzione dell'inferenza: il modello viene eseguito in parallelo, partendo dagli operatori che non hanno dipendenze o che hanno dipendenze già completamente soddisfatte, questi verranno messi in coda nel thread pool, ogni volta che un operatore viene portato a termine, il risultato di quest'ultimo verrà comunicato al thread principale che aggiornerà il grafo delle dipendenze e farà partire gli operatori che grazie a questo risultato hanno soddisfatto ora le loro dipendenze e così via, finchè il grafo delle dipendenze non sarà vuoto, e avremo quindi ottenuto il risultato finale.
- **TODO**: Spiegare meglio come funziona la parallelizzazione
- **TODO**: inserire qualche immagine del grafo 
- Comparazione degli output: il programma legge inoltre anche gli output "di reference" che dovrebbero essere ottenuti dall'esecuzione del modello e li confronta con quelli effettivamente ottenuti, controllando che la differenza tra i singoli valori dei due risultati sia massimo 0.0001 e stampando qualche statistica del confronto.

### Utilizzare Stonnx come libreria

Il programma viene compilato come libreria dinamica (.dll / .so / .dylib) con il nome `stonnx_api` e può essere utilizzato normalmente attraverso i binding esposti.

Si è reso disponibile un binding verso i seguenti linguaggi:

- Python
- C
- C++
- C#

I bindings, molto limitati per il momento, sono stati fatti mettendo a disposizione qualche funzione per la creazione della rete e l'esecuzione della stessa.

### Benchmark
Qui di seguito sono riportati i risultati dei benchmark effettuati su alcuni modelli. I benchmark sono stati effettuati utilizzando hyperfine, su un computer con le seguenti caratteristiche:
- CPU: AMD Ryzen 7 5800X
- RAM: 32GB DDR4 3200MHz


#### AlexNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model bvlcalexnet-12 "            
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model bvlcalexnet-12
  Time (mean ± σ):     714.2 ms ±  10.5 ms    [User: 279.4 ms, System: 110.6 ms]
  Range (min … max):   694.9 ms … 731.1 ms    10 runs
```
#### CaffeNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model caffenet-12 "            
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model caffenet-12
  Time (mean ± σ):     730.8 ms ±   5.4 ms    [User: 308.8 ms, System: 86.9 ms]
  Range (min … max):   725.4 ms … 740.3 ms    10 runs
```
#### Emotion
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model emotion-ferplus-8 "
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model emotion-ferplus-8
  Time (mean ± σ):     483.9 ms ±   7.1 ms    [User: 225.6 ms, System: 27.5 ms]
  Range (min … max):   477.8 ms … 495.2 ms    10 runs
```
#### GoogleNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model googlenet-12 "     
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model googlenet-12
  Time (mean ± σ):      2.387 s ±  0.046 s    [User: 1.324 s, System: 0.034 s]
  Range (min … max):    2.328 s …  2.475 s    10 runs
```
#### GPT2
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model GPT2 "        
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model GPT2
  Time (mean ± σ):      2.563 s ±  0.025 s    [User: 1.286 s, System: 0.160 s]
  Range (min … max):    2.526 s …  2.601 s    10 runs
```
#### Inception
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model inception-v2-9 "
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model inception-v2-9
  Time (mean ± σ):      2.516 s ±  0.038 s    [User: 1.343 s, System: 0.055 s]
  Range (min … max):    2.482 s …  2.605 s    10 runs
```
#### MobileNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model mobilenetv2-7 "
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model mobilenetv2-7
  Time (mean ± σ):      1.566 s ±  0.027 s    [User: 0.769 s, System: 0.015 s]
  Range (min … max):    1.539 s …  1.614 s    10 runs
```
#### Mnist
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model mnist-12 "     
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model mnist-12
  Time (mean ± σ):      44.5 ms ±   8.8 ms    [User: 1.8 ms, System: 3.1 ms]
  Range (min … max):    28.8 ms …  75.8 ms    24 runs
```
#### ResNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model resnet50-caffe2-v1-9 "
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model resnet50-caffe2-v1-9
  Time (mean ± σ):      1.730 s ±  0.041 s    [User: 0.656 s, System: 0.079 s]
  Range (min … max):    1.682 s …  1.826 s    10 runs
```
#### Shufflenet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model squeezenet1.0-12 "    
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model squeezenet1.0-12
  Time (mean ± σ):     279.3 ms ±   3.8 ms    [User: 88.8 ms, System: 6.5 ms]
  Range (min … max):   275.0 ms … 289.1 ms    10 runs
```
#### SqueezeNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model squeezenet1.0-12 "    
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model squeezenet1.0-12
  Time (mean ± σ):     280.6 ms ±   8.1 ms    [User: 75.9 ms, System: 1.2 ms]
  Range (min … max):   263.8 ms … 294.5 ms    10 runs
```
#### Super Resolution
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model super_resolution "
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model super_resolution
  Time (mean ± σ):      4.646 s ±  0.043 s    [User: 1.669 s, System: 0.049 s]
  Range (min … max):    4.585 s …  4.724 s    10 runs
```
#### VGG
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model vgg19-7 "         
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model vgg19-7
  Time (mean ± σ):      7.452 s ±  0.094 s    [User: 3.000 s, System: 0.279 s]
  Range (min … max):    7.351 s …  7.626 s    10 runs
```
#### ZFNet
```
hyperfine ".\target\release\stonnx.exe --verbose 0 --model zfnet512-12 "
Benchmark 1: .\target\release\stonnx.exe --verbose 0 --model zfnet512-12
  Time (mean ± σ):     952.5 ms ±  15.6 ms    [User: 334.4 ms, System: 136.9 ms]
  Range (min … max):   929.6 ms … 981.2 ms    10 runs
```