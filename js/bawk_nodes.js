import { app } from "../../scripts/app.js";

const NODE_COLORS = {
    DiffusionModelLoader: ["#1a2332", "#4a90e2"],
    BawkLoraStack: ["#32301a", "#f39c12"],
    FluxWildcardEncode: ["#2a1a32", "#9b59b6"],
    BawkSampler: ["#32261a", "#e67e22"],
    FluxImageSaver: ["#1a3226", "#27ae60"],
    BawkImageLoader: ["#1a3226", "#2ecc71"],
    BawkBatchProcessor: ["#1a2a32", "#16a085"],
};

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
            return result;
        };
    },
});
