(() => {
  "use strict";

  const REFERENCE_PACKET = "REFERENCE_ANNOTATION";
  const BLIND_PACKET = "BLIND_FIRST_STOP_REVIEW";
  const BLINDING_CONDITION = "FROZEN_FIRST_STOP_VISIBLE_EVIDENCE_ONLY_V1";
  const DECISIONS = new Set([
    "DELIVERABLE",
    "NEEDS_FURTHER_INSPECTION",
    "UNABLE_TO_JUDGE",
  ]);

  const byId = (id) => document.getElementById(id);
  const elements = {
    packetFile: byId("packet-file"),
    sessionFile: byId("session-file"),
    exportSession: byId("export-session"),
    modeName: byId("mode-name"),
    saveState: byId("save-state"),
    progressText: byId("progress-text"),
    progressBar: byId("progress-bar"),
    packetNumber: byId("packet-number"),
    itemList: byId("item-list"),
    itemKicker: byId("item-kicker"),
    itemTitle: byId("item-title"),
    itemState: byId("item-state"),
    referenceToolbar: byId("reference-toolbar"),
    viewerStage: byId("viewer-stage"),
    svg: byId("review-svg"),
    emptyBackground: byId("empty-background"),
    baseImage: byId("base-image"),
    reportLayer: byId("report-layer"),
    supportLayer: byId("support-layer"),
    polygonLayer: byId("polygon-layer"),
    draftPolygon: byId("draft-polygon"),
    handleLayer: byId("handle-layer"),
    showCertain: byId("show-certain"),
    showUncertain: byId("show-uncertain"),
    showReport: byId("show-report"),
    showSupport: byId("show-support"),
    reportToggleWrap: byId("report-toggle-wrap"),
    supportToggleWrap: byId("support-toggle-wrap"),
    layerOpacity: byId("layer-opacity"),
    referenceForm: byId("reference-form"),
    blindForm: byId("blind-form"),
    packetId: byId("packet-id"),
    packetKind: byId("packet-kind"),
    reviewerAlias: byId("reviewer-alias"),
    referenceType: byId("reference-type"),
    annotationOrigin: byId("annotation-origin"),
    methodParticipant: byId("method-participant"),
    sawModel: byId("saw-model"),
    referenceNotes: byId("reference-notes"),
    blindReviewer: byId("blind-reviewer"),
    blindCondition: byId("blind-condition"),
    problemType: byId("problem-type"),
    reviewBasis: byId("review-basis"),
    blindNotes: byId("blind-notes"),
    message: byId("message"),
    previousItem: byId("previous-item"),
    nextItem: byId("next-item"),
  };

  const state = {
    packet: null,
    mode: null,
    currentIndex: 0,
    items: {},
    history: {},
    future: {},
    tool: "select",
    selectedRegion: null,
    view: null,
    drag: null,
    changeLog: [],
    dirty: false,
  };

  const clone = (value) => JSON.parse(JSON.stringify(value));

  function showMessage(text, kind = "") {
    elements.message.textContent = text;
    elements.message.className = `message ${kind}`.trim();
  }

  function readJsonFile(file) {
    return file.text().then((text) => JSON.parse(text));
  }

  function currentPacketItem() {
    return state.packet ? state.packet.items[state.currentIndex] : null;
  }

  function currentItemState() {
    const item = currentPacketItem();
    return item ? state.items[item.item_id] : null;
  }

  function defaultReferenceState() {
    return {
      state: "UNANNOTATED",
      regions: [],
      uncertain_regions: [],
      open_polygon: null,
      notes: "",
    };
  }

  function defaultBlindState() {
    return {
      state: "UNREVIEWED",
      decision: "",
      problem_type: "",
      review_basis: "",
      notes: "",
    };
  }

  function validatePacket(packet) {
    if (!packet || packet.schema_version !== 1 || !Array.isArray(packet.items) || !packet.packet_id) {
      throw new Error("任务包结构无效");
    }
    if (packet.packet_kind !== REFERENCE_PACKET && packet.packet_kind !== BLIND_PACKET) {
      throw new Error("不支持的任务包类型");
    }
    if (packet.packet_kind === BLIND_PACKET && packet.blinding_condition !== BLINDING_CONDITION) {
      throw new Error("盲评材料条件不匹配");
    }
    const ids = new Set();
    packet.items.forEach((item) => {
      if (!item.item_id || ids.has(item.item_id) || !Number.isInteger(item.native_width) || !Number.isInteger(item.native_height)) {
        throw new Error("任务项目身份无效");
      }
      ids.add(item.item_id);
      if (packet.packet_kind === REFERENCE_PACKET && (!item.image_data || !item.specimen_key || item.frame !== "registered_cscan")) {
        throw new Error("参考标注项目不完整");
      }
      if (packet.packet_kind === BLIND_PACKET && (!item.measured_image_data || !item.report_overlay_data || !item.support_overlay_data || !item.report_id)) {
        throw new Error("盲评项目不完整");
      }
    });
  }

  function loadPacket(packet) {
    validatePacket(packet);
    state.packet = packet;
    state.mode = packet.packet_kind === REFERENCE_PACKET ? "reference" : "blind";
    state.currentIndex = 0;
    state.items = {};
    state.history = {};
    state.future = {};
    state.selectedRegion = null;
    state.changeLog = [];
    packet.items.forEach((item) => {
      state.items[item.item_id] = state.mode === "reference" ? defaultReferenceState() : defaultBlindState();
      state.history[item.item_id] = [];
      state.future[item.item_id] = [];
    });
    state.dirty = false;
    elements.referenceForm.hidden = state.mode !== "reference";
    elements.referenceToolbar.hidden = state.mode !== "reference";
    elements.blindForm.hidden = state.mode !== "blind";
    elements.reportToggleWrap.hidden = state.mode !== "blind";
    elements.supportToggleWrap.hidden = state.mode !== "blind";
    elements.exportSession.disabled = false;
    elements.previousItem.disabled = packet.items.length < 2;
    elements.nextItem.disabled = packet.items.length < 2;
    elements.modeName.textContent = state.mode === "reference" ? "参考区域标注" : "首次 STOP 报告盲评";
    elements.packetId.textContent = packet.packet_id;
    elements.packetKind.textContent = packet.packet_kind;
    elements.packetNumber.textContent = packet.packet_number === 0 ? "练习包" : `第 ${packet.packet_number || 1} 包`;
    if (state.mode === "blind") {
      elements.blindCondition.textContent = packet.blinding_condition;
    }
    elements.saveState.textContent = "尚未导出";
    renderAll(true);
    showMessage(packet.test_only ? "已载入 TEST_ONLY 练习包" : `已载入 ${packet.items.length} 个项目`, "success");
  }

  function itemStatus(itemState) {
    if (!itemState) return {label: "未开始", className: "neutral", dot: ""};
    if (itemState.state === "DRAFT") return {label: "草稿", className: "draft", dot: "draft"};
    if (itemState.state === "UNABLE_TO_JUDGE") return {label: "无法判读", className: "unable", dot: "unable"};
    if (itemState.state === "CONFIRMED_NO_CERTAIN_REGION") return {label: "已确认无确定区", className: "done", dot: "done"};
    if (itemState.state === "CONFIRMED") return {label: "已确认", className: "done", dot: "done"};
    return {label: "未开始", className: "neutral", dot: ""};
  }

  function completedCount() {
    return Object.values(state.items).filter((item) =>
      state.mode === "reference"
        ? ["CONFIRMED", "CONFIRMED_NO_CERTAIN_REGION", "UNABLE_TO_JUDGE"].includes(item.state)
        : item.state === "CONFIRMED"
    ).length;
  }

  function renderNavigator() {
    elements.itemList.replaceChildren();
    state.packet.items.forEach((item, index) => {
      const appState = state.items[item.item_id];
      const status = itemStatus(appState);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `item-button${index === state.currentIndex ? " active" : ""}`;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", String(index === state.currentIndex));
      const title = state.mode === "reference" ? item.specimen_key : `匿名报告 ${String(item.display_number).padStart(3, "0")}`;
      const subtitle = state.mode === "reference" ? `${item.native_width} × ${item.native_height}` : item.task;
      button.innerHTML = `<span class="index">${String(index + 1).padStart(2, "0")}</span><span class="labels"><strong></strong><span></span></span><i class="status-dot ${status.dot}"></i>`;
      button.querySelector("strong").textContent = title;
      button.querySelector(".labels span").textContent = subtitle;
      button.addEventListener("click", () => navigateTo(index));
      elements.itemList.append(button);
    });
    const done = completedCount();
    elements.progressText.textContent = `${done} / ${state.packet.items.length}`;
    elements.progressBar.style.width = `${state.packet.items.length ? (100 * done) / state.packet.items.length : 0}%`;
  }

  function resetView() {
    const item = currentPacketItem();
    if (!item) return;
    state.view = {x: -0.5, y: -0.5, width: item.native_width, height: item.native_height};
    applyView();
  }

  function applyView() {
    if (!state.view) return;
    const {x, y, width, height} = state.view;
    elements.svg.setAttribute("viewBox", `${x} ${y} ${width} ${height}`);
  }

  function setImageAttributes(element, href, item) {
    element.setAttribute("href", href || "");
    element.setAttribute("x", "-0.5");
    element.setAttribute("y", "-0.5");
    element.setAttribute("width", String(item.native_width));
    element.setAttribute("height", String(item.native_height));
  }

  function polygonPoints(region, item) {
    return region.polygon
      .map(([u, v]) => `${u * (item.native_width - 1)},${v * (item.native_height - 1)}`)
      .join(" ");
  }

  function allReferenceRegions(itemState) {
    return [
      ...itemState.regions.map((region) => ({...region, bucket: "regions"})),
      ...itemState.uncertain_regions.map((region) => ({...region, bucket: "uncertain_regions"})),
    ];
  }

  function renderReferenceLayers(item) {
    const itemState = currentItemState();
    elements.polygonLayer.replaceChildren();
    elements.handleLayer.replaceChildren();
    const opacity = Number(elements.layerOpacity.value) / 100;
    allReferenceRegions(itemState).forEach((region) => {
      if ((region.certainty === "certain" && !elements.showCertain.checked) ||
          (region.certainty === "uncertain" && !elements.showUncertain.checked)) return;
      const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      polygon.setAttribute("points", polygonPoints(region, item));
      polygon.setAttribute("class", `region ${region.certainty}${state.selectedRegion === region.id ? " selected" : ""}`);
      polygon.style.opacity = String(opacity);
      polygon.dataset.regionId = region.id;
      polygon.addEventListener("pointerdown", (event) => {
        if (state.tool !== "select") return;
        event.stopPropagation();
        state.selectedRegion = region.id;
        renderReferenceLayers(item);
      });
      elements.polygonLayer.append(polygon);
    });
    const selected = allReferenceRegions(itemState).find((region) => region.id === state.selectedRegion);
    if (selected) {
      const radius = Math.max(state.view.width, state.view.height) * 0.007;
      selected.polygon.forEach(([u, v], index) => {
        const handle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        handle.setAttribute("cx", String(u * (item.native_width - 1)));
        handle.setAttribute("cy", String(v * (item.native_height - 1)));
        handle.setAttribute("r", String(radius));
        handle.setAttribute("class", "vertex-handle");
        handle.dataset.handleIndex = String(index);
        handle.dataset.regionId = selected.id;
        elements.handleLayer.append(handle);
      });
    }
    const open = itemState.open_polygon;
    elements.draftPolygon.setAttribute("points", open ? polygonPoints({polygon: open.polygon}, item) : "");
  }

  function renderCurrent(resetViewport = false) {
    const item = currentPacketItem();
    const itemState = currentItemState();
    if (!item) return;
    elements.viewerStage.classList.remove("empty");
    elements.emptyBackground.style.display = "none";
    elements.itemKicker.textContent = state.mode === "reference" ? "试样" : "匿名首次 STOP 报告";
    elements.itemTitle.textContent = state.mode === "reference" ? item.specimen_key : `报告 ${String(item.display_number).padStart(3, "0")} · ${item.task}`;
    const status = itemStatus(itemState);
    elements.itemState.textContent = status.label;
    elements.itemState.className = `state-chip ${status.className}`;
    if (state.mode === "reference") {
      setImageAttributes(elements.baseImage, item.image_data, item);
      setImageAttributes(elements.reportLayer, "", item);
      setImageAttributes(elements.supportLayer, "", item);
      elements.referenceNotes.value = itemState.notes || "";
      renderReferenceLayers(item);
    } else {
      setImageAttributes(elements.baseImage, item.measured_image_data, item);
      setImageAttributes(elements.reportLayer, item.report_overlay_data, item);
      setImageAttributes(elements.supportLayer, item.support_overlay_data, item);
      elements.reportLayer.style.display = elements.showReport.checked ? "" : "none";
      elements.supportLayer.style.display = elements.showSupport.checked ? "" : "none";
      elements.reportLayer.style.opacity = String(Number(elements.layerOpacity.value) / 100);
      elements.problemType.value = itemState.problem_type || "";
      elements.reviewBasis.value = itemState.review_basis || "";
      elements.blindNotes.value = itemState.notes || "";
      document.querySelectorAll("[data-decision]").forEach((button) => {
        button.classList.toggle("active", button.dataset.decision === itemState.decision);
      });
      elements.polygonLayer.replaceChildren();
      elements.handleLayer.replaceChildren();
      elements.draftPolygon.setAttribute("points", "");
    }
    if (resetViewport || !state.view) resetView();
    elements.previousItem.disabled = state.currentIndex === 0;
    elements.nextItem.disabled = state.currentIndex === state.packet.items.length - 1;
  }

  function renderAll(resetViewport = false) {
    if (!state.packet) return;
    renderNavigator();
    renderCurrent(resetViewport);
  }

  function syncCurrentInputs() {
    const itemState = currentItemState();
    if (!itemState) return;
    if (state.mode === "reference") {
      itemState.notes = elements.referenceNotes.value;
    } else {
      itemState.problem_type = elements.problemType.value;
      itemState.review_basis = elements.reviewBasis.value;
      itemState.notes = elements.blindNotes.value;
    }
  }

  function navigateTo(index) {
    if (!state.packet || index < 0 || index >= state.packet.items.length || index === state.currentIndex) return;
    syncCurrentInputs();
    state.currentIndex = index;
    state.selectedRegion = null;
    renderAll(true);
  }

  function pushHistory() {
    const item = currentPacketItem();
    if (!item) return;
    state.history[item.item_id].push(clone(state.items[item.item_id]));
    if (state.history[item.item_id].length > 80) state.history[item.item_id].shift();
    state.future[item.item_id] = [];
  }

  function markEdited() {
    const itemState = currentItemState();
    if (itemState && ["CONFIRMED", "CONFIRMED_NO_CERTAIN_REGION", "UNABLE_TO_JUDGE"].includes(itemState.state)) {
      recordStateChange(itemState.state, "DRAFT");
      itemState.state = "DRAFT";
    }
    markDirty();
  }

  function recordStateChange(from, to) {
    const item = currentPacketItem();
    if (!item || from === to) return;
    state.changeLog.push({item_id: item.item_id, from, to, sequence: state.changeLog.length + 1});
  }

  function undo() {
    const item = currentPacketItem();
    if (!item || !state.history[item.item_id].length) return;
    state.future[item.item_id].push(clone(state.items[item.item_id]));
    state.items[item.item_id] = state.history[item.item_id].pop();
    state.selectedRegion = null;
    markDirty();
    renderAll();
  }

  function redo() {
    const item = currentPacketItem();
    if (!item || !state.future[item.item_id].length) return;
    state.history[item.item_id].push(clone(state.items[item.item_id]));
    state.items[item.item_id] = state.future[item.item_id].pop();
    state.selectedRegion = null;
    markDirty();
    renderAll();
  }

  function screenToImage(event) {
    const item = currentPacketItem();
    const matrix = elements.svg.getScreenCTM();
    if (!item || !matrix) throw new Error("无法读取图像坐标");
    const point = elements.svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const local = point.matrixTransform(matrix.inverse());
    return {
      x: Math.max(0, Math.min(item.native_width - 1, local.x)),
      y: Math.max(0, Math.min(item.native_height - 1, local.y)),
    };
  }

  function normalizedPoint(event) {
    const item = currentPacketItem();
    const point = screenToImage(event);
    return [point.x / (item.native_width - 1), point.y / (item.native_height - 1)];
  }

  function setTool(tool) {
    if (!state.packet || state.mode !== "reference") return;
    state.tool = tool;
    document.querySelectorAll("[data-tool]").forEach((button) => {
      button.classList.toggle("active", button.dataset.tool === tool);
    });
    elements.svg.style.cursor = tool === "pan" ? "grab" : tool === "select" ? "default" : "crosshair";
  }

  function addDraftPoint(event) {
    if (!state.packet || state.mode !== "reference" || !["certain", "uncertain"].includes(state.tool)) return;
    const itemState = currentItemState();
    if (!itemState.open_polygon) {
      pushHistory();
      itemState.open_polygon = {certainty: state.tool, polygon: []};
    } else if (itemState.open_polygon.certainty !== state.tool) {
      showMessage("请先闭合或取消当前多边形", "error");
      return;
    }
    itemState.open_polygon.polygon.push(normalizedPoint(event));
    itemState.state = "DRAFT";
    markDirty();
    renderReferenceLayers(currentPacketItem());
  }

  function finishDraftPolygon() {
    if (!state.packet || state.mode !== "reference") return;
    const itemState = currentItemState();
    const open = itemState.open_polygon;
    if (!open) return;
    const points = open.polygon.filter((point, index, values) =>
      index === 0 || Math.hypot(point[0] - values[index - 1][0], point[1] - values[index - 1][1]) > 1e-8
    );
    if (points.length > 1 && Math.hypot(points[0][0] - points.at(-1)[0], points[0][1] - points.at(-1)[1]) <= 1e-8) points.pop();
    const distinct = new Set(points.map(([u, v]) => `${u.toFixed(12)},${v.toFixed(12)}`));
    const area = Math.abs(points.reduce((total, point, index) => {
      const next = points[(index + 1) % points.length];
      return total + point[0] * next[1] - next[0] * point[1];
    }, 0));
    if (distinct.size < 3 || area <= 1e-12) {
      showMessage("多边形至少需要三个不同且非共线的顶点", "error");
      return;
    }
    const certainty = open.certainty;
    const bucket = certainty === "certain" ? "regions" : "uncertain_regions";
    const id = `${certainty}-${Date.now()}-${itemState[bucket].length + 1}`;
    itemState[bucket].push({id, certainty, polygon: points});
    itemState.open_polygon = null;
    itemState.state = "DRAFT";
    state.selectedRegion = id;
    markDirty();
    renderAll();
    showMessage(certainty === "certain" ? "已添加确定区" : "已添加不确定区", "success");
  }

  function cancelDraftPolygon() {
    const itemState = currentItemState();
    if (!itemState || !itemState.open_polygon) return;
    itemState.open_polygon = null;
    markDirty();
    renderReferenceLayers(currentPacketItem());
  }

  function deleteSelectedRegion() {
    const itemState = currentItemState();
    if (!itemState || !state.selectedRegion) return;
    pushHistory();
    itemState.regions = itemState.regions.filter((region) => region.id !== state.selectedRegion);
    itemState.uncertain_regions = itemState.uncertain_regions.filter((region) => region.id !== state.selectedRegion);
    state.selectedRegion = null;
    itemState.state = "DRAFT";
    markEdited();
    renderAll();
  }

  function selectedRegion() {
    const itemState = currentItemState();
    if (!itemState || !state.selectedRegion) return null;
    return allReferenceRegions(itemState).find((region) => region.id === state.selectedRegion) || null;
  }

  function beginPointer(event) {
    if (!state.packet) return;
    if (state.mode === "reference" && event.target.dataset.handleIndex !== undefined) {
      const region = selectedRegion();
      if (!region) return;
      pushHistory();
      state.drag = {kind: "vertex", index: Number(event.target.dataset.handleIndex), pointerId: event.pointerId};
      elements.svg.setPointerCapture(event.pointerId);
      event.preventDefault();
      return;
    }
    if (state.tool === "pan" || event.button === 1 || state.mode === "blind") {
      state.drag = {
        kind: "pan",
        pointerId: event.pointerId,
        clientX: event.clientX,
        clientY: event.clientY,
        view: {...state.view},
      };
      elements.svg.setPointerCapture(event.pointerId);
      elements.svg.style.cursor = "grabbing";
      event.preventDefault();
      return;
    }
    if (state.mode === "reference" && state.tool === "select" && event.target === elements.svg) {
      state.selectedRegion = null;
      renderReferenceLayers(currentPacketItem());
    }
  }

  function movePointer(event) {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    if (state.drag.kind === "vertex") {
      const region = selectedRegion();
      if (!region) return;
      region.polygon[state.drag.index] = normalizedPoint(event);
      markEdited();
      renderReferenceLayers(currentPacketItem());
      return;
    }
    const rect = elements.svg.getBoundingClientRect();
    state.view.x = state.drag.view.x - (event.clientX - state.drag.clientX) * state.drag.view.width / rect.width;
    state.view.y = state.drag.view.y - (event.clientY - state.drag.clientY) * state.drag.view.height / rect.height;
    applyView();
  }

  function endPointer(event) {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    elements.svg.releasePointerCapture(event.pointerId);
    state.drag = null;
    elements.svg.style.cursor = state.tool === "pan" || state.mode === "blind" ? "grab" : "default";
    if (state.mode === "reference") renderAll();
  }

  function zoom(factor, clientPoint = null) {
    if (!state.packet || !state.view) return;
    const item = currentPacketItem();
    const minWidth = item.native_width * 0.08;
    const maxWidth = item.native_width * 4;
    const nextWidth = Math.max(minWidth, Math.min(maxWidth, state.view.width * factor));
    const applied = nextWidth / state.view.width;
    let anchorX = state.view.x + state.view.width / 2;
    let anchorY = state.view.y + state.view.height / 2;
    if (clientPoint) {
      const local = screenToImage(clientPoint);
      anchorX = local.x;
      anchorY = local.y;
    }
    state.view.x = anchorX - (anchorX - state.view.x) * applied;
    state.view.y = anchorY - (anchorY - state.view.y) * applied;
    state.view.width *= applied;
    state.view.height *= applied;
    applyView();
    if (state.mode === "reference") renderReferenceLayers(item);
  }

  function setApplicationState(next) {
    const itemState = currentItemState();
    if (!itemState) return;
    syncCurrentInputs();
    if (!elements.reviewerAlias.value.trim()) {
      showMessage("请填写评阅者别名", "error");
      return;
    }
    if (next !== "UNABLE_TO_JUDGE" && !elements.referenceType.value) {
      showMessage("请选择真实参考来源", "error");
      return;
    }
    if (next === "CONFIRMED" && itemState.regions.length === 0) {
      showMessage("确认完成前至少需要一个确定区", "error");
      return;
    }
    if (next === "CONFIRMED_NO_CERTAIN_REGION" && itemState.regions.length > 0) {
      showMessage("当前仍有确定区，请删除后再确认无确定区", "error");
      return;
    }
    if (itemState.open_polygon) {
      showMessage("请先闭合或取消未完成的多边形", "error");
      return;
    }
    recordStateChange(itemState.state, next);
    itemState.state = next;
    markDirty();
    renderAll();
    showMessage("当前试样状态已更新", "success");
  }

  function saveReferenceDraft() {
    const itemState = currentItemState();
    if (!itemState) return;
    syncCurrentInputs();
    recordStateChange(itemState.state, "DRAFT");
    itemState.state = "DRAFT";
    markDirty();
    renderAll();
    showMessage("草稿已保留在当前会话，请导出会话形成可移动备份", "success");
  }

  function chooseDecision(decision) {
    if (!DECISIONS.has(decision)) return;
    const itemState = currentItemState();
    if (!itemState) return;
    itemState.decision = decision;
    if (itemState.state !== "CONFIRMED") itemState.state = "DRAFT";
    markDirty();
    renderAll();
  }

  function saveBlindDraft() {
    const itemState = currentItemState();
    if (!itemState) return;
    syncCurrentInputs();
    recordStateChange(itemState.state, "DRAFT");
    itemState.state = "DRAFT";
    markDirty();
    renderAll();
    showMessage("盲评草稿已保留，请导出会话形成可移动备份", "success");
  }

  function confirmBlind() {
    const itemState = currentItemState();
    if (!itemState) return;
    syncCurrentInputs();
    if (!elements.blindReviewer.value.trim()) {
      showMessage("请填写评阅者编号", "error");
      return;
    }
    if (!DECISIONS.has(itemState.decision)) {
      showMessage("请选择一个判断", "error");
      return;
    }
    recordStateChange(itemState.state, "CONFIRMED");
    itemState.state = "CONFIRMED";
    markDirty();
    renderAll();
    showMessage("当前报告判断已确认", "success");
  }

  function sessionSnapshot() {
    syncCurrentInputs();
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    if (state.mode === "reference") {
      return {
        schema_version: 1,
        session_kind: "CSCAN_REFERENCE_SESSION",
        packet_id: state.packet.packet_id,
        export_id: `reference-${stamp}`,
        reviewer: {
          reviewer_alias: elements.reviewerAlias.value.trim(),
          reference_type: elements.referenceType.value,
          participated_in_method_development: elements.methodParticipant.checked,
          saw_model_outputs: elements.sawModel.checked,
          annotation_origin: elements.annotationOrigin.value,
        },
        items: clone(state.items),
        change_log: clone(state.changeLog),
      };
    }
    return {
      schema_version: 1,
      session_kind: "CSCAN_BLIND_REVIEW_SESSION",
      packet_id: state.packet.packet_id,
      export_id: `blind-${stamp}`,
      reviewer_id: elements.blindReviewer.value.trim(),
      blinding_condition: BLINDING_CONDITION,
      items: clone(state.items),
      change_log: clone(state.changeLog),
    };
  }

  function downloadJson(payload, filename) {
    const blob = new Blob([JSON.stringify(payload, null, 2) + "\n"], {type: "application/json;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportSession() {
    if (!state.packet) return;
    const session = sessionSnapshot();
    downloadJson(session, `${session.export_id}.json`);
    state.dirty = false;
    elements.saveState.textContent = "已导出";
    showMessage("会话文件已导出", "success");
  }

  function importSession(session) {
    if (!state.packet) throw new Error("请先导入对应任务包");
    const expectedKind = state.mode === "reference" ? "CSCAN_REFERENCE_SESSION" : "CSCAN_BLIND_REVIEW_SESSION";
    if (!session || session.schema_version !== 1 || session.session_kind !== expectedKind || session.packet_id !== state.packet.packet_id || !session.items) {
      throw new Error("会话与当前任务包不匹配");
    }
    const known = new Set(state.packet.items.map((item) => item.item_id));
    if (Object.keys(session.items).some((id) => !known.has(id))) throw new Error("会话包含未知项目");
    state.packet.items.forEach((item) => {
      state.items[item.item_id] = clone(session.items[item.item_id] || (state.mode === "reference" ? defaultReferenceState() : defaultBlindState()));
      state.history[item.item_id] = [];
      state.future[item.item_id] = [];
    });
    state.changeLog = clone(session.change_log || []);
    if (state.mode === "reference") {
      const reviewer = session.reviewer || {};
      elements.reviewerAlias.value = reviewer.reviewer_alias || "";
      elements.referenceType.value = reviewer.reference_type || "";
      elements.annotationOrigin.value = reviewer.annotation_origin || "FROM_SCRATCH";
      elements.methodParticipant.checked = reviewer.participated_in_method_development === true;
      elements.sawModel.checked = reviewer.saw_model_outputs === true;
    } else {
      elements.blindReviewer.value = session.reviewer_id || "";
    }
    state.currentIndex = 0;
    state.selectedRegion = null;
    state.dirty = false;
    elements.saveState.textContent = "会话已恢复";
    renderAll(true);
    showMessage("会话已恢复", "success");
  }

  function markDirty() {
    state.dirty = true;
    elements.saveState.textContent = "有未导出更改";
    try {
      if (state.packet) localStorage.setItem(`cscan-review-${state.packet.packet_id}`, JSON.stringify(sessionSnapshot()));
    } catch (_) {
      // Explicit file export remains the authoritative backup path.
    }
  }

  elements.packetFile.addEventListener("change", async () => {
    const file = elements.packetFile.files[0];
    if (!file) return;
    try {
      loadPacket(await readJsonFile(file));
    } catch (error) {
      showMessage(error.message || String(error), "error");
    } finally {
      elements.packetFile.value = "";
    }
  });

  elements.sessionFile.addEventListener("change", async () => {
    const file = elements.sessionFile.files[0];
    if (!file) return;
    try {
      importSession(await readJsonFile(file));
    } catch (error) {
      showMessage(error.message || String(error), "error");
    } finally {
      elements.sessionFile.value = "";
    }
  });

  elements.exportSession.addEventListener("click", exportSession);
  elements.previousItem.addEventListener("click", () => navigateTo(state.currentIndex - 1));
  elements.nextItem.addEventListener("click", () => navigateTo(state.currentIndex + 1));
  document.querySelectorAll("[data-tool]").forEach((button) => button.addEventListener("click", () => setTool(button.dataset.tool)));
  document.querySelectorAll("[data-decision]").forEach((button) => button.addEventListener("click", () => chooseDecision(button.dataset.decision)));
  byId("undo").addEventListener("click", undo);
  byId("redo").addEventListener("click", redo);
  byId("cancel-polygon").addEventListener("click", cancelDraftPolygon);
  byId("delete-region").addEventListener("click", deleteSelectedRegion);
  byId("zoom-in").addEventListener("click", () => zoom(0.8));
  byId("zoom-out").addEventListener("click", () => zoom(1.25));
  byId("reset-view").addEventListener("click", resetView);
  byId("save-draft").addEventListener("click", saveReferenceDraft);
  byId("confirm-reference").addEventListener("click", () => setApplicationState("CONFIRMED"));
  byId("confirm-no-certain").addEventListener("click", () => setApplicationState("CONFIRMED_NO_CERTAIN_REGION"));
  byId("unable-reference").addEventListener("click", () => setApplicationState("UNABLE_TO_JUDGE"));
  byId("save-blind-draft").addEventListener("click", saveBlindDraft);
  byId("confirm-blind").addEventListener("click", confirmBlind);

  [elements.showCertain, elements.showUncertain, elements.layerOpacity].forEach((control) => control.addEventListener("input", () => state.mode === "reference" && renderReferenceLayers(currentPacketItem())));
  elements.showReport.addEventListener("change", () => { elements.reportLayer.style.display = elements.showReport.checked ? "" : "none"; });
  elements.showSupport.addEventListener("change", () => { elements.supportLayer.style.display = elements.showSupport.checked ? "" : "none"; });
  elements.layerOpacity.addEventListener("input", () => {
    elements.reportLayer.style.opacity = String(Number(elements.layerOpacity.value) / 100);
  });

  elements.svg.addEventListener("click", (event) => {
    if (event.detail === 1) addDraftPoint(event);
  });
  elements.svg.addEventListener("dblclick", (event) => {
    if (["certain", "uncertain"].includes(state.tool)) {
      event.preventDefault();
      finishDraftPolygon();
    }
  });
  elements.svg.addEventListener("pointerdown", beginPointer);
  elements.svg.addEventListener("pointermove", movePointer);
  elements.svg.addEventListener("pointerup", endPointer);
  elements.svg.addEventListener("pointercancel", endPointer);
  elements.svg.addEventListener("wheel", (event) => {
    if (!state.packet) return;
    event.preventDefault();
    zoom(event.deltaY < 0 ? 0.88 : 1.14, event);
  }, {passive: false});

  document.addEventListener("keydown", (event) => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
    if (event.key === "Escape") cancelDraftPolygon();
    if (event.key === "Enter" && state.mode === "reference") finishDraftPolygon();
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      event.shiftKey ? redo() : undo();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      redo();
    }
  });

  window.addEventListener("beforeunload", (event) => {
    if (!state.dirty) return;
    event.preventDefault();
    event.returnValue = "";
  });

  window.CScanReviewApp = {loadPacket, importSession, sessionSnapshot, screenToImage};
})();
