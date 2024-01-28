const { invoke } = window.__TAURI__.tauri;
const { open } = window.__TAURI__.dialog;
const { dirname} = window.__TAURI__.path;

let modelInputEl;
let modelErrorMsgEl;
let modelMsgEl;

async function runModel() {
    if (modelInputEl.value === "") {
        modelErrorMsgEl.textContent = "Please select a model file.";
        return;
    } else {
        modelErrorMsgEl.textContent = "Running model...";
    }
    const result = await invoke("run_model", { path: modelInputEl.value });
    if (typeof result === "string") {
        modelErrorMsgEl.textContent = result;
        return;
    } else {
        modelErrorMsgEl.textContent = "";
    }
    modelMsgEl.replaceChildren();
    for (const key in result) {
        const tableRow = modelMsgEl.appendChild(document.createElement("tr"));
        result[key]['data'] = JSON.parse(result[key]['data'])['data'];
        const tableNameCol = tableRow.appendChild(document.createElement("td"));
        tableNameCol.textContent = key;
        const tableTypeCol = tableRow.appendChild(document.createElement("td"));
        tableTypeCol.textContent = result[key]['data_type'];
        const tableShapeCol = tableRow.appendChild(document.createElement("td"));
        tableShapeCol.textContent = result[key]['shape'];
        const tableDataCol = tableRow.appendChild(document.createElement("td"));
        const data = result[key]['data'];
        tableDataCol.textContent = `[ ${data.slice(0, 5).map(x => x.toFixed(2)).join(', ')} ... ${data.slice(-5).map(x => x.toFixed(2)).join(', ')} ]`;
    }
}

async function pickModel() {
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    const result = await open({
        multiple: false,
        filter: "Model Files|*.onnx",
    });
    if (result === null) {
        return;
    }
    const path = await dirname(result);
    modelInputEl.value = path;
}

window.addEventListener("DOMContentLoaded", () => {
    modelInputEl = document.querySelector("#model-input");
    modelMsgEl = document.querySelector("#model-msg-body");
    modelErrorMsgEl = document.querySelector("#model-error-msg");
    document.querySelector("#model-form").addEventListener("submit", (e) => {
        e.preventDefault();
        pickModel();
    });
    document.querySelector("#run-button").addEventListener("click", (e) => {
        e.preventDefault();
        runModel();
    })
});
