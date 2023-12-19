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

### Utilizzare ONNXRustProto come libreria

Il programma viene compilato come libreria dinamica (.dll / .so / .dylib) con il nome `stonnx_api` e può essere utilizzato normalmente attraverso i binding esposti.

Si è reso disponibile un binding verso i seguenti linguaggi:

- Python
- C
- C++
- C#

I bindings, molto limitati per il momento, sono stati fatti mettendo a disposizione qualche funzione per la creazione della rete e l'esecuzione della stessa.

