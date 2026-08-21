import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_COLORS = {
    DiffusionModelLoader: ["#1a2332", "#4a90e2"],
    BawkLoraStack: ["#32301a", "#f39c12"],
    BawkLoraSwitch: ["#32301a", "#f1c40f"],
    FluxWildcardEncode: ["#2a1a32", "#9b59b6"],
    BawkSampler: ["#32261a", "#e67e22"],
    FluxImageSaver: ["#1a3226", "#27ae60"],
    BawkImageLoader: ["#1a3226", "#2ecc71"],
    BawkBatchProcessor: ["#1a2a32", "#16a085"],
};

const PREVIEW_WIDGET = "final_prompt_preview";

function ensurePreviewWidget(node) {
    let widget = node.widgets?.find((w) => w.name === PREVIEW_WIDGET);
    if (widget) return widget;
    widget = ComfyWidgets.STRING(node, PREVIEW_WIDGET, ["STRING", { multiline: true }], app).widget;
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = 0.7;
    widget.inputEl.placeholder = "Final prompt appears here after the node runs";
    widget.serializeValue = async () => "";
    return widget;
}

app.registerExtension({
    name: "bawk.nodes",
    beforeRegisterNodeDef(nodeType, nodeData) {
        const colors = NODE_COLORS[nodeData.name];
        if (!colors) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            [this.bgcolor, this.color] = colors;
            for (const widget of this.widgets ?? []) {
                if (widget.name === "prompt" && widget.inputEl) {
                    widget.inputEl.style.minHeight = "120px";
                    widget.inputEl.placeholder = "Prompt... use {option1|option2} for wildcards";
                }
            }
            if (nodeData.name === "FluxWildcardEncode") {
                ensurePreviewWidget(this);
            }
            return result;
        };

        if (nodeData.name === "FluxWildcardEncode") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const text = message?.text;
                if (!text) return;
                const widget = ensurePreviewWidget(this);
                widget.value = Array.isArray(text) ? text.join("\n") : String(text);
                this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
                this.setDirtyCanvas(true, true);
            };
        }
    },
});
