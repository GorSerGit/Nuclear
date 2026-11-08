"use strict";

const $ = (id) => document.getElementById(id);
const tabs = [...document.querySelectorAll("[data-tab]")];
const editorButtons = [...document.querySelectorAll("[data-editor]")];
const cameraList = $("camera-list");
const viewer = $("viewer");
const generation = $("generation");
const historyFallback = $("history-fallback");
const historyControls = $("history-controls");
const clearHistory = $("clear-history");
const resultsMessage = $("results-message");
let activeTab = "results";
let editor = "plan";
let inventory = [];
let inventoryKey = null;
let inventoryRequest = 0;
let loadingInventory = false;
let runs = [];
let selectedRun = null;
let submitting = false;
let clearing = false;
let historyRequest = 0;
let shownLog = null;
let dependencies = null;
let dependencyRequest = 0;
let dependencyAction = false;
const deviceLabels = { auto: "Авто", cpu: "CPU", gpu: "GPU" };

function workspace() {
  return { videos: $("videos-dir").value.trim(), configs: $("configs-dir").value.trim() };
}

function workspaceQuery() {
  return new URLSearchParams(workspace()).toString();
}

async function request(url, body) {
  const response = await fetch(url, body === undefined ? undefined : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || `Ошибка запроса (${response.status})`);
  return result;
}

function cameras() {
  return [...cameraList.querySelectorAll("input")];
}

function updateButtons() {
  const busy = runs.some((run) => run.status === "running");
  $("generate").disabled = busy || submitting || clearing || loadingInventory
    || dependencyAction || dependencies?.busy || !(dependencies?.cpu || dependencies?.gpu)
    || inventoryKey !== workspaceQuery() || !cameras().some((box) => box.checked);
  clearHistory.disabled = busy || submitting || clearing;
  $("install-dependencies").disabled = !dependencies || dependencies.busy || dependencies.phase === "reboot"
    || dependencyAction || busy || submitting;
  $("install-driver").disabled = $("skip-driver").disabled = dependencyAction;
  $("runtime-message").hidden = !!(dependencies?.cpu || dependencies?.gpu || dependencies?.busy);
}

function renderVideos() {
  const cards = inventory.map((row) => {
    const name = encodeURIComponent(row.name);
    const url = `/source/${name}?${inventoryKey}`;
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "none";
    if (dependencies?.base_ready) video.poster = `/poster/${name}?${inventoryKey}`;
    video.src = url;
    const info = document.createElement("div");
    info.className = "video-info";
    const title = document.createElement("strong");
    title.textContent = `Камера ${row.camera} · ${row.name}`;
    const size = document.createElement("small");
    size.textContent = `${(row.size / 1024 ** 2).toFixed(1)} МБ`;
    const download = document.createElement("a");
    download.href = url;
    download.download = row.name;
    download.textContent = "Скачать";
    download.title = "Скачать исходный файл";
    const error = document.createElement("small");
    error.hidden = true;
    error.textContent = "Запись не воспроизводится в браузере.";
    video.addEventListener("error", () => { error.hidden = false; });
    info.append(title, size, download, error);
    const card = document.createElement("article");
    card.append(video, info);
    return card;
  });
  $("video-list").replaceChildren(...cards);
}

async function refreshInventory() {
  const id = ++inventoryRequest;
  const key = workspaceQuery();
  loadingInventory = true;
  updateButtons();
  try {
    const items = (await request(`/api/videos?${key}`)).sort((a, b) => a.camera - b.camera);
    if (id !== inventoryRequest) return;
    const old = cameras();
    const selected = inventoryKey === key && old.some((box) => !box.checked)
      ? new Set(old.filter((box) => box.checked).map((box) => box.dataset.file)) : null;
    const changed = inventoryKey !== key || JSON.stringify(inventory) !== JSON.stringify(items);
    inventory = items;
    inventoryKey = key;
    const rows = inventory.map((row) => {
      const label = document.createElement("label");
      label.className = "check";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.value = row.camera;
      box.dataset.file = row.name;
      box.checked = selected === null || selected.has(row.name);
      label.append(box, `Камера ${row.camera} · ${row.name}`);
      return label;
    });
    cameraList.replaceChildren(...rows);
    if (!rows.length) cameraList.textContent = "В выбранной папке нет видео.";
    $("videos-message").textContent = rows.length ? `Записей: ${rows.length}` : "В выбранной папке нет видео.";
    if (activeTab === "videos" && (changed || !$("video-list").children.length)) renderVideos();
    if (changed && activeTab !== "videos") $("video-list").replaceChildren();
  } catch (error) {
    if (id !== inventoryRequest) return;
    inventoryKey = null;
    inventory = [];
    cameraList.textContent = error.message;
    $("videos-message").textContent = error.message;
    $("video-list").replaceChildren();
  } finally {
    if (id === inventoryRequest) {
      loadingInventory = false;
      updateButtons();
    }
  }
}

function renderEditor() {
  $("editor").hidden = !dependencies?.base_ready;
  $("editor-dependencies").hidden = !!dependencies?.base_ready;
  for (const button of editorButtons) button.setAttribute("aria-pressed", String(button.dataset.editor === editor));
  if (!dependencies?.base_ready) return;
  const src = `/${editor}/?${workspaceQuery()}`;
  if ($("editor").getAttribute("src") !== src) $("editor").src = src;
}

function showTab() {
  const next = tabs.find((button) => button.dataset.tab === location.hash.slice(1));
  if (activeTab === "videos") {
    $("videos").classList.remove("dragging");
    for (const video of $("video-list").querySelectorAll("video")) video.pause();
  }
  activeTab = next?.dataset.tab || "results";
  for (const button of tabs) {
    const selected = button.dataset.tab === activeTab;
    button.setAttribute("aria-pressed", String(selected));
    $(button.dataset.tab).hidden = !selected;
  }
  if (activeTab === "plan") renderEditor();
  if (activeTab === "dependencies") refreshDependencies();
  if (activeTab === "videos" || activeTab === "process") refreshInventory();
}

function resultLabel(run) {
  const date = new Date(run.created).toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" });
  if (!run.number) return `${run.label} · ${date}`;
  const { start, end, cameras: ids } = run.options;
  const time = start === null && end === null ? "вся запись" : `${start ?? 0}–${end ?? "конец"} с`;
  return `№${run.number} · ${time} · камеры ${ids.join(", ")} · ${date}`;
}

function renderResults() {
  if (clearing) return;
  const ready = runs.filter((run) => run.viewer);
  const key = ready.map((run) => run.id).join(",");
  if (generation.dataset.key !== key) {
    generation.replaceChildren(...(ready.length
      ? ready.map((run) => new Option(resultLabel(run), run.id))
      : [new Option("Нет готовых результатов", "")]));
    generation.dataset.key = key;
  }
  const run = ready.find((item) => item.id === selectedRun) || ready[0];
  generation.disabled = !run;
  generation.value = run?.id || "";
  viewer.hidden = !run;
  $("empty-results").hidden = !!run;
  if (run && viewer.getAttribute("src") !== run.viewer) {
    restoreHistoryControls();
    viewer.src = run.viewer;
  } else if (!run && viewer.hasAttribute("src")) {
    restoreHistoryControls();
    viewer.removeAttribute("src");
  }
}

function restoreHistoryControls() {
  historyFallback.append(historyControls);
  historyFallback.hidden = false;
}

viewer.addEventListener("load", () => {
  const slot = viewer.contentDocument?.getElementById("generation-controls");
  if (!slot) {
    restoreHistoryControls();
    return;
  }
  slot.append(historyControls);
  historyFallback.hidden = true;
  slot.closest("details").hidden = false;
});

async function renderStatus() {
  const run = runs.find((item) => item.number);
  const status = $("status-text");
  const open = $("open-result");
  const log = $("job-log");
  const descriptions = {
    running: "в работе",
    ready: "готова",
    failed: run?.error || "ошибка обработки",
    interrupted: "прервана",
  };
  const text = run ? `Обработка №${run.number} · ${descriptions[run.status]}` : "Ожидание запуска";
  if (status.textContent !== text) status.textContent = text;
  open.hidden = !run?.viewer;
  open.onclick = () => {
    selectedRun = run.id;
    renderResults();
    location.hash = "results";
  };
  if (shownLog !== run?.id) {
    shownLog = run?.id;
    log.textContent = "";
    log.hidden = true;
  }
  if (!run || activeTab !== "process") return;
  try {
    const { text: output } = await request(`/api/generations/${run.id}/log`);
    if (shownLog !== run.id || output === log.textContent) return;
    const wasAtEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 12;
    log.textContent = output;
    log.hidden = !output;
    if (wasAtEnd) log.scrollTop = log.scrollHeight;
  } catch (error) {
    if (shownLog !== run.id || clearing) return;
    log.textContent = error.message;
    log.hidden = false;
  }
}

async function refreshHistory() {
  const id = ++historyRequest;
  try {
    const result = await request("/api/generations");
    if (id !== historyRequest) return;
    runs = result;
    renderResults();
    updateButtons();
    await renderStatus();
  } catch (error) {
    if (id === historyRequest) resultsMessage.textContent = error.message;
  }
}

async function refreshDependencies() {
  const id = ++dependencyRequest;
  try {
    const state = await request("/api/dependencies");
    if (id !== dependencyRequest) return;
    const baseChanged = dependencies?.base_ready !== state.base_ready;
    dependencies = state;
    if (baseChanged && activeTab === "plan") renderEditor();
    if (baseChanged && activeTab === "videos" && inventoryKey !== null) renderVideos();
    const nvidia = state.nvidia || {};
    const summary = [`Python ${state.python}`, nvidia.name || state.gpu_name, nvidia.driver && `драйвер ${nvidia.driver}`,
      nvidia.cuda && `CUDA драйвера ${nvidia.cuda}`, state.runtime_cuda && `CUDA PyTorch ${state.runtime_cuda}`,
      state.base_ready ? "разметка готова" : "разметка недоступна",
      `CPU ${state.cpu ? "готов" : "недоступен"}`, `GPU ${state.gpu ? "готов" : "недоступен"}`].filter(Boolean);
    $("dependency-summary").textContent = summary.join(" · ");
    $("dependency-stage").textContent = state.stage;
    if (!dependencyAction) $("dependency-error").textContent = state.error || state.cuda_error || "";
    $("driver-offer").hidden = !state.offer;
    $("driver-title").textContent = state.offer ? `${state.offer.title} · ${state.offer.hardware_id}` : "";
    $("driver-help").href = state.driver_help;
    $("driver-help").hidden = !state.needs_driver;
    const log = $("dependency-log");
    log.hidden = !state.log;
    if (log.textContent !== state.log) {
      const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 12;
      log.textContent = state.log;
      if (atEnd) log.scrollTop = log.scrollHeight;
    }
    const items = [...Object.entries(state.packages).map(([name, value]) => `${name} · ${value || "не установлен"}`),
      ...Object.entries(state.weights).map(([name, exists]) => `${name} · ${exists ? "готов" : "отсутствует"}`)];
    $("dependency-versions").replaceChildren(...items.map((text) => {
      const item = document.createElement("li");
      item.textContent = text;
      return item;
    }));
    updateButtons();
  } catch (error) {
    if (id === dependencyRequest) $("dependency-error").textContent = error.message;
  }
}

async function updateDependencies(endpoint, body) {
  dependencyAction = true;
  updateButtons();
  $("dependency-error").textContent = "";
  try {
    await request(`/api/dependencies/${endpoint}`, body);
    await refreshDependencies();
  } catch (error) {
    $("dependency-error").textContent = error.message;
  } finally {
    dependencyAction = false;
    updateButtons();
  }
}

function upload(file, query, progress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const params = new URLSearchParams(query);
    params.set("name", file.name);
    xhr.open("POST", `/api/videos/upload?${params}`);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => { progress.value = event.loaded; };
    xhr.onload = () => {
      if (xhr.status === 201) resolve();
      else reject(new Error(xhr.response?.error || `Ошибка загрузки (${xhr.status})`));
    };
    xhr.onerror = xhr.onabort = () => reject(new Error("Загрузка прервана."));
    xhr.send(file);
  });
}

async function uploadFiles(files) {
  if (!files.length) return;
  const query = workspaceQuery();
  const list = $("uploads");
  list.hidden = false;
  const items = files.map((file) => {
    const item = document.createElement("li");
    const progress = document.createElement("progress");
    progress.max = Math.max(file.size, 1);
    progress.value = 0;
    progress.setAttribute("aria-label", file.name);
    const status = document.createElement("span");
    status.textContent = "В очереди";
    item.append(file.name, progress, status);
    list.append(item);
    return { file, progress, status };
  });
  for (const { file, progress, status } of items) {
    status.textContent = "Загрузка…";
    try {
      await upload(file, query, progress);
      status.textContent = "Готово";
    } catch (error) {
      status.textContent = error.message;
      status.className = "error";
    }
  }
  await refreshInventory();
}

for (const button of tabs) {
  button.onclick = () => {
    location.hash = button.dataset.tab;
  };
}
for (const button of editorButtons) {
  button.onclick = () => {
    editor = button.dataset.editor;
    renderEditor();
  };
}
window.addEventListener("hashchange", () => {
  showTab();
  renderStatus();
});
generation.onchange = () => {
  selectedRun = generation.value === runs.find((run) => run.viewer)?.id ? null : generation.value;
  renderResults();
};
$("select-all").onclick = () => { cameras().forEach((box) => { box.checked = true; }); updateButtons(); };
$("select-none").onclick = () => { cameras().forEach((box) => { box.checked = false; }); updateButtons(); };
cameraList.addEventListener("change", updateButtons);
$("refresh-videos").onclick = refreshInventory;
$("pick-videos").onclick = () => $("video-files").click();
$("video-files").onchange = () => {
  uploadFiles([...$("video-files").files]);
  $("video-files").value = "";
};
for (const type of ["dragover", "drop"]) {
  window.addEventListener(type, (event) => {
    if (!event.dataTransfer?.types.includes("Files")) return;
    event.preventDefault();
    if (activeTab !== "videos") return;
    $("videos").classList.toggle("dragging", type === "dragover");
    if (type === "drop") uploadFiles([...event.dataTransfer.files]);
  });
}
$("videos").addEventListener("dragleave", (event) => {
  if (!$("videos").contains(event.relatedTarget)) $("videos").classList.remove("dragging");
});
$("quality-info").onclick = (event) => {
  const button = event.currentTarget;
  button.setAttribute("aria-expanded", String(button.getAttribute("aria-expanded") !== "true"));
};
$("quality-info").onkeydown = (event) => {
  if (event.key === "Escape") event.currentTarget.setAttribute("aria-expanded", "false");
};
$("open-config").onclick = async () => {
  const message = $("config-message");
  try {
    const result = await request("/api/config/open", { ...workspace(), kind: $("config-kind").value });
    message.textContent = `Открыт ${result.file}`;
  } catch (error) {
    message.textContent = error.message;
  }
};
function updateInstallHint() {
  $("install-dependencies").title = `Устройство — ${deviceLabels[$("device").value]}. Выбор в системных настройках обработки.`;
}
$("device").onchange = updateInstallHint;
updateInstallHint();
$("install-dependencies").onclick = () => updateDependencies("install", { device: $("device").value });
$("install-driver").onclick = () => updateDependencies("driver", { install: true });
$("skip-driver").onclick = () => updateDependencies("driver", { install: false });
for (const id of ["videos-dir", "configs-dir"]) {
  $(id).addEventListener("input", updateButtons);
  $(id).addEventListener("change", refreshInventory);
}

$("process-form").onsubmit = async (event) => {
  event.preventDefault();
  if ($("generate").disabled) return;
  const error = $("process-error");
  error.hidden = true;
  const seconds = (id) => $(id).value === "" ? null : $(id).valueAsNumber;
  submitting = true;
  updateButtons();
  try {
    await request("/api/generations", {
      ...workspace(), cameras: cameras().filter((box) => box.checked).map((box) => +box.value),
      start: seconds("start"), end: seconds("end"), quality: $("quality").value,
      device: $("device").value,
      silhouettes: $("silhouettes").checked, fresh: $("fresh").checked,
      data: $("data-dir").value.trim(), ui: $("ui-dir").value.trim(),
    });
    selectedRun = null;
    await refreshHistory();
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
  } finally {
    submitting = false;
    updateButtons();
  }
};

clearHistory.onclick = async () => {
  const confirmed = window.confirm(
    "Удалить все сохранённые обработки и прежние результаты офисной обработки? Исходные записи и разметка сохранятся.",
  );
  if (!confirmed) return;
  clearing = true;
  restoreHistoryControls();
  viewer.removeAttribute("src");
  updateButtons();
  resultsMessage.textContent = "Удаляю результаты…";
  try {
    await request("/api/generations/clear", { confirm: true });
    selectedRun = null;
    resultsMessage.textContent = "";
  } catch (error) {
    resultsMessage.textContent = error.message;
  } finally {
    clearing = false;
    await refreshHistory();
  }
};

showTab();
async function poll() {
  await Promise.all([refreshHistory(), refreshDependencies()]);
  const busy = dependencies?.busy || runs.some((run) => run.status === "running");
  window.setTimeout(poll, busy ? 1500 : 6000);
}
poll();
