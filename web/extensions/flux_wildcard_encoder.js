/**
 * FLUX Wildcard Encoder Extension - rgthree-style Power LoRA Loader UI
 * File: web/extensions/flux_wildcard_encoder.js
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_TYPE = "FluxWildcardEncode";
const MAX_LORAS = 10; // Maximum number of LoRAs

class FluxWildcardEncoderNode {
    constructor() {
        this.loras = [];
        this.loraCounter = 0;
    }

    setup(node) {
        this.node = node;
        this.setupLoraUI();
    }

    setupLoraUI() {
        // Remove default widgets that we'll replace
        this.node.widgets = this.node.widgets?.filter(w => 
            !w.name.startsWith("lora_") && 
            w.name !== "toggle_all_loras"
        ) || [];

        // Add "Toggle All" button at the top
        const toggleAllWidget = ComfyWidgets.BOOLEAN(this.node, "toggle_all", ["BOOLEAN", { default: false }], app);
        toggleAllWidget.name = "toggle_all";
        toggleAllWidget.callback = () => {
            const enabled = toggleAllWidget.value;
            this.loras.forEach(lora => {
                if (lora.toggle) {
                    lora.toggle.value = enabled;
                }
            });
            this.node.setDirtyCanvas(true, true);
        };

        // Add initial LoRA row
        this.addLoraRow();

        // Add "Add LoRA" button
        this.addLoraButton();

        // Update node size
        this.updateNodeSize();
    }

    addLoraRow() {
        if (this.loras.length >= MAX_LORAS) {
            console.warn(`Maximum ${MAX_LORAS} LoRAs reached`);
            return;
        }

        const loraIndex = this.loraCounter++;
        const lora = {
            index: loraIndex,
            enabled: false,
            name: "None",
            strength: 1.0,
            strengthTwo: 1.0
        };

        // Create toggle widget
        const toggleWidget = ComfyWidgets.BOOLEAN(
            this.node, 
            `lora_${loraIndex}_toggle`, 
            ["BOOLEAN", { default: false }], 
            app
        );
        toggleWidget.name = `lora_${loraIndex}_on`;
        toggleWidget.callback = () => {
            lora.enabled = toggleWidget.value;
            this.updateNodeInputs();
        };

        // Create LoRA selection widget
        const loraOptions = this.getLoraOptions();
        const loraWidget = ComfyWidgets.COMBO(
            this.node,
            `lora_${loraIndex}_name`,
            [loraOptions, { default: "None" }],
            app
        );
        loraWidget.name = `lora_${loraIndex}_lora`;
        loraWidget.callback = () => {
            lora.name = loraWidget.value;
            this.updateNodeInputs();
        };

        // Create strength widget
        const strengthWidget = ComfyWidgets.NUMBER(
            this.node,
            `lora_${loraIndex}_strength`,
            ["FLOAT", { default: 1.0, min: -10.0, max: 10.0, step: 0.01 }],
            app
        );
        strengthWidget.name = `lora_${loraIndex}_strength`;
        strengthWidget.callback = () => {
            lora.strength = strengthWidget.value;
            lora.strengthTwo = strengthWidget.value; // Use same for both by default
            this.updateNodeInputs();
        };

        // Create remove button for this LoRA
        const removeWidget = ComfyWidgets.BUTTON(
            this.node,
            `remove_lora_${loraIndex}`,
            "🗑️",
            () => this.removeLoraRow(loraIndex)
        );
        removeWidget.name = `remove_lora_${loraIndex}`;

        // Store references
        lora.toggle = toggleWidget;
        lora.loraWidget = loraWidget;
        lora.strengthWidget = strengthWidget;
        lora.removeWidget = removeWidget;

        this.loras.push(lora);
        this.updateNodeSize();
        this.updateNodeInputs();
    }

    removeLoraRow(index) {
        // Find and remove the LoRA
        const loraIndex = this.loras.findIndex(l => l.index === index);
        if (loraIndex === -1) return;

        const lora = this.loras[loraIndex];

        // Remove widgets
        this.node.widgets = this.node.widgets.filter(w => 
            w !== lora.toggle && 
            w !== lora.loraWidget && 
            w !== lora.strengthWidget && 
            w !== lora.removeWidget
        );

        // Remove from array
        this.loras.splice(loraIndex, 1);

        this.updateNodeSize();
        this.updateNodeInputs();
        this.node.setDirtyCanvas(true, true);
    }

    addLoraButton() {
        const addButton = ComfyWidgets.BUTTON(
            this.node,
            "add_lora",
            "+ Add LoRA",
            () => this.addLoraRow()
        );
        addButton.name = "add_lora_button";
    }

    getLoraOptions() {
        // Get LoRA files from ComfyUI
        try {
            const loraList = app.ui.settings.getSettingValue("Comfy.LoraList") || [];
            return ["None", ...loraList];
        } catch (e) {
            // Fallback - we'll populate this from the backend
            return ["None"];
        }
    }

    updateNodeInputs() {
        // Clear existing LoRA inputs
        if (this.node.inputs) {
            this.node.inputs = this.node.inputs.filter(input => 
                !input.name.startsWith("lora_")
            );
        }

        // Add current LoRA inputs to the node's input data
        this.loras.forEach(lora => {
            if (lora.enabled && lora.name !== "None") {
                const loraData = {
                    on: lora.enabled,
                    lora: lora.name,
                    strength: lora.strength,
                    strengthTwo: lora.strengthTwo
                };

                // Store in node's widget values for serialization
                if (!this.node.widgets_values) {
                    this.node.widgets_values = {};
                }
                this.node.widgets_values[`lora_${lora.index}`] = loraData;
            }
        });
    }

    updateNodeSize() {
        // Calculate required height based on number of widgets
        const baseHeight = 100; // Base height for prompt and wildcard_seed
        const loraRowHeight = 25; // Height per LoRA row
        const buttonHeight = 25; // Height for buttons

        const totalHeight = baseHeight + 
                          (this.loras.length * loraRowHeight) + 
                          buttonHeight + 
                          25; // Toggle all button

        this.node.size = [
            Math.max(300, this.node.size[0]), // Minimum width
            Math.max(totalHeight, this.node.size[1])
        ];

        this.node.setDirtyCanvas(true, true);
    }

    onDrawForeground(ctx) {
        // Custom drawing for LoRA UI (similar to rgthree style)
        if (!this.loras.length) return;

        const margin = 10;
        let y = 80; // Start below the main inputs

        // Draw LoRA section header
        ctx.fillStyle = "#555";
        ctx.fillRect(margin, y, this.node.size[0] - margin * 2, 1);
        y += 10;

        // Draw each LoRA row
        this.loras.forEach((lora, index) => {
            this.drawLoraRow(ctx, lora, margin, y + index * 25);
        });
    }

    drawLoraRow(ctx, lora, x, y) {
        const rowHeight = 20;
        const toggleWidth = 20;
        const nameWidth = 150;
        const strengthWidth = 60;

        // Draw toggle
        this.drawToggle(ctx, x, y, toggleWidth, rowHeight, lora.enabled);

        // Draw LoRA name
        ctx.fillStyle = lora.name !== "None" ? "#fff" : "#888";
        ctx.font = "12px Arial";
        ctx.fillText(
            lora.name.length > 20 ? lora.name.substring(0, 17) + "..." : lora.name,
            x + toggleWidth + 5,
            y + 14
        );

        // Draw strength value
        ctx.fillStyle = "#fff";
        ctx.fillText(
            lora.strength.toFixed(2),
            x + toggleWidth + nameWidth + 10,
            y + 14
        );
    }

    drawToggle(ctx, x, y, width, height, enabled) {
        // Draw toggle button background
        ctx.fillStyle = enabled ? "#4a9eff" : "#333";
        ctx.fillRect(x + 2, y + 2, width - 4, height - 4);

        // Draw toggle border
        ctx.strokeStyle = "#666";
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 2, y + 2, width - 4, height - 4);

        // Draw checkmark if enabled
        if (enabled) {
            ctx.strokeStyle = "#fff";
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(x + 6, y + height / 2);
            ctx.lineTo(x + width / 2, y + height - 6);
            ctx.lineTo(x + width - 4, y + 6);
            ctx.stroke();
        }
    }

    serialize() {
        const data = {};
        this.loras.forEach(lora => {
            if (lora.enabled && lora.name !== "None") {
                data[`lora_${lora.index}`] = {
                    on: lora.enabled,
                    lora: lora.name,
                    strength: lora.strength,
                    strengthTwo: lora.strengthTwo
                };
            }
        });
        return data;
    }

    deserialize(data) {
        // Clear existing LoRAs
        this.loras = [];
        this.loraCounter = 0;

        // Recreate LoRAs from data
        Object.keys(data).forEach(key => {
            if (key.startsWith("lora_")) {
                const loraData = data[key];
                this.addLoraRow();
                const lora = this.loras[this.loras.length - 1];
                
                lora.enabled = loraData.on || false;
                lora.name = loraData.lora || "None";
                lora.strength = loraData.strength || 1.0;
                lora.strengthTwo = loraData.strengthTwo || lora.strength;

                // Update widgets
                if (lora.toggle) lora.toggle.value = lora.enabled;
                if (lora.loraWidget) lora.loraWidget.value = lora.name;
                if (lora.strengthWidget) lora.strengthWidget.value = lora.strength;
            }
        });

        this.updateNodeSize();
        this.updateNodeInputs();
    }
}

// Extension registration
app.registerExtension({
    name: "BawkNodes.FluxWildcardEncoder",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === NODE_TYPE) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // Initialize our custom UI
                this.fluxWildcardEncoder = new FluxWildcardEncoderNode();
                this.fluxWildcardEncoder.setup(this);
                
                // Override drawing
                const originalOnDrawForeground = this.onDrawForeground;
                this.onDrawForeground = function(ctx) {
                    originalOnDrawForeground?.apply(this, arguments);
                    this.fluxWildcardEncoder?.onDrawForeground(ctx);
                };
                
                return result;
            };

            // Override serialization
            const originalSerialize = nodeType.prototype.serialize;
            nodeType.prototype.serialize = function() {
                const data = originalSerialize?.apply(this, arguments) || {};
                if (this.fluxWildcardEncoder) {
                    data.fluxWildcardEncoder = this.fluxWildcardEncoder.serialize();
                }
                return data;
            };

            // Override deserialization
            const originalConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function(data) {
                originalConfigure?.apply(this, arguments);
                if (this.fluxWildcardEncoder && data.fluxWildcardEncoder) {
                    this.fluxWildcardEncoder.deserialize(data.fluxWildcardEncoder);
                }
            };
        }
    }
});

console.log("🎲 FLUX Wildcard Encoder extension loaded");