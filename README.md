# ONNXRustProto

### Main Contributors:
- [s317659 Rosiello Alessio](https://github.com/Atari2)
- [s317661 Tcaciuc Claudiu Constantin](https://github.com/ClaudiuTcaciuc)

### Descrizione progetto:

Il progetto consiste nella realizzazione di un interprete ONNX utilizzando il linguaggio Rust. Le richieste da rispettare sono le seguenti:
- Creazione di un parser per estrarre dal file ONNX l'informazione necessaria per la creazione della rete;
- Implementazione di un sotto-set di operatori ONNX;
- Utilizzo di parallelismo per l'esecuzione della rete;
- (Opzionale) binding con altri linguaggi.

### Modalità di utilizzo:
- Eseguire il comando `cargo build --release` per compilare il progetto;
- Eseguire il comando `cargo run --release -- --model <modelname>`
- Nel caso in cui si volesse visualizzare l'esecuzione degli operatori è possibile aggiungere l'opzione `--verbose` al comando precedente (di default verbose è 0).
  - Esempio: `cargo run --release -- --model <modelname> --verbose 1`
  - `verbose = 0`: non visualizza nulla
  - `verbose = 1`: visualizza informazioni riguardanti l'esecuzione degli operatori
  - `verbose = 2`: inserisce gli output di ogni operatore in un file `.npy`
  - `verbose = 4`: inserisce anche gli output intermedi di ogni operatore in un file `.npy`
- Eseguire il comando `cargo doc --open` per visualizzare la documentazione del progetto.

### Modelli supportati:
I modelli testati fanno riferimento a quelli presenti nella sezione [archive](https://github.com/onnx/models/tree/main/archive) del [repository ufficiale di ONNX](https://github.com/onnx/models), in quanto a inizio Dicembre 2023 sono stati aggiornati e aggiunti nuovi modelli. I modelli testati sono i seguenti:
- [AlexNet](https://github.com/onnx/models/tree/main/archive/vision/classification/alexnet)
- [MobileNet](https://github.com/onnx/models/tree/main/archive/vision/classification/mobilenet)
- [GoogleNet](https://github.com/onnx/models/tree/main/archive/vision/classification/inception_and_googlenet/googlenet)
- [ResNet](https://github.com/onnx/models/tree/main/archive/vision/classification/resnet)
- [GPT2](https://github.com/onnx/models/tree/main/archive/text/machine_comprehension/gpt-2)
- [Emotion](https://github.com/onnx/models/tree/main/archive/vision/body_analysis/emotion_ferplus)
- [UltraFace (RFB)](https://github.com/onnx/models/tree/main/archive/vision/body_analysis/ultraface)

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

### Binding con altri linguaggi:
Si è reso disponibile un binding verso i seguenti linguaggi:
- Python
- C
- C++
- C#

Il binding, molto limitato per il momento, è stato fatto mettendo a disposizione qualche funzione per la creazione della rete e l'esecuzione della stessa.