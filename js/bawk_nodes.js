/**
 * Bawk Nodes - UI Enhancements for ComfyUI
 * Improves textarea height, adds visual feedback, and enhances usability
 */

import { app } from "/scripts/app.js";

// Configuration
const CONFIG = {
    // Default textarea rows for different nodes
    promptTextareaRows: 8,  // Taller default for better visibility
    pathTextareaRows: 3,

    // Minimum heights for resizable textareas (in pixels)
    minHeight: 100,

    // Enable/disable features
    enableTallerTextareas: true,
    enableTooltipEnhancements: true,
    enableColorCoding: true,
};

/**
 * Main extension registration
 */
app.registerExtension({
    name: "bawk.nodes.ui.enhancements",

    async setup() {
        console.log("[Bawk Nodes] UI enhancements loaded");
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Only apply to Bawk nodes
        if (!nodeData.name || !isBawkNode(nodeData.name)) {
            return;
        }

        console.log(`[Bawk Nodes] Enhancing UI for: ${nodeData.name}`);

        // Hook into node creation
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);

            // Apply enhancements
            if (CONFIG.enableTallerTextareas) {
                enhanceTextareas(this, nodeData.name);
            }

            if (CONFIG.enableColorCoding) {
                applyNodeColorCoding(this, nodeData.name);
            }

            return result;
        };

        // Hook into widget value changes for dynamic behavior
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(config) {
            const result = onConfigure?.apply(this, arguments);

            // Reapply enhancements after configuration
            if (CONFIG.enableTallerTextareas) {
                enhanceTextareas(this, nodeData.name);
            }

            return result;
        };
    }
});

/**
 * Check if a node is a Bawk node
 */
function isBawkNode(nodeName) {
    const bawkNodes = [
        "FluxWildcardEncode",
        "BawkSampler",
        "BawkBatchProcessor",
        "FluxImageSaver",
        "BawkImageLoader",
        "BawkControlNet",
        "DiffusionModelLoader",
        "FluxDynamicLoraStack",
        "FluxLoraStackApply",
        "ZImageModelLoader"
    ];
    return bawkNodes.includes(nodeName);
}

/**
 * Enhance textarea widgets to be taller and more usable
 */
function enhanceTextareas(node, nodeName) {
    if (!node.widgets) return;

    for (const widget of node.widgets) {
        // Only enhance STRING widgets with multiline
        if (widget.type !== "customtext" && widget.type !== "text") {
            continue;
        }

        // Determine appropriate height based on widget name and node type
        let rows = getAppropriateRows(widget.name, nodeName);

        // Apply to input element if it exists
        if (widget.inputEl) {
            widget.inputEl.rows = rows;
            widget.inputEl.style.minHeight = `${CONFIG.minHeight}px`;

            // Make textarea resizable
            widget.inputEl.style.resize = "vertical";
            widget.inputEl.style.overflow = "auto";

            // Add placeholder for prompt fields
            if (isPromptField(widget.name)) {
                widget.inputEl.placeholder = "Enter your prompt here... Use {option1|option2} for wildcards";
            }

            console.log(`[Bawk Nodes] Enhanced textarea: ${widget.name} with ${rows} rows`);
        }

        // Update widget's computed height
        if (widget.computeSize) {
            const originalComputeSize = widget.computeSize;
            widget.computeSize = function(width) {
                const size = originalComputeSize?.call(this, width) || [width, 20];
                // Ensure minimum height
                size[1] = Math.max(size[1], CONFIG.minHeight);
                return size;
            };
        }
    }

    // Request canvas redraw
    if (node.graph && node.graph.canvas) {
        node.setDirtyCanvas(true, true);
    }
}

/**
 * Determine appropriate number of rows for a textarea
 */
function getAppropriateRows(widgetName, nodeName) {
    // Large textareas for prompts
    if (isPromptField(widgetName)) {
        return CONFIG.promptTextareaRows;
    }

    // Medium textareas for paths and URLs
    if (isPathField(widgetName)) {
        return CONFIG.pathTextareaRows;
    }

    // Default
    return 4;
}

/**
 * Check if a field is a prompt field
 */
function isPromptField(fieldName) {
    const promptFields = ["prompt", "text", "positive", "negative"];
    return promptFields.some(f => fieldName.toLowerCase().includes(f));
}

/**
 * Check if a field is a path field
 */
function isPathField(fieldName) {
    const pathFields = ["path", "file", "url", "webhook", "batch_file"];
    return pathFields.some(f => fieldName.toLowerCase().includes(f));
}

/**
 * Apply color coding to nodes for better visual organization
 */
function applyNodeColorCoding(node, nodeName) {
    // Define color schemes for different node types
    const colorSchemes = {
        // Model loading nodes - Blue theme
        "DiffusionModelLoader": {
            bgcolor: "#1a2332",
            color: "#4a90e2"
        },

        // Text encoding nodes - Purple theme
        "FluxWildcardEncode": {
            bgcolor: "#2a1a32",
            color: "#9b59b6"
        },

        // Sampling nodes - Orange theme
        "BawkSampler": {
            bgcolor: "#32261a",
            color: "#e67e22"
        },

        // Image I/O nodes - Green theme
        "FluxImageSaver": {
            bgcolor: "#1a3226",
            color: "#27ae60"
        },
        "BawkImageLoader": {
            bgcolor: "#1a3226",
            color: "#2ecc71"
        },

        // Utility nodes - Teal theme
        "BawkBatchProcessor": {
            bgcolor: "#1a2a32",
            color: "#16a085"
        },

        // Control nodes - Red theme
        "BawkControlNet": {
            bgcolor: "#32201a",
            color: "#e74c3c"
        },

        // LoRA nodes - Gold theme
        "FluxDynamicLoraStack": {
            bgcolor: "#32301a",
            color: "#f39c12"
        },
        "FluxLoraStackApply": {
            bgcolor: "#32301a",
            color: "#f1c40f"
        },

        // Z-Image nodes - Cyan theme
        "ZImageModelLoader": {
            bgcolor: "#1a3232",
            color: "#00bcd4"
        }
    };

    const scheme = colorSchemes[nodeName];
    if (scheme) {
        node.bgcolor = scheme.bgcolor;
        node.color = scheme.color;
    }
}

/**
 * Additional utility: Auto-expand textarea on focus
 * This makes editing large prompts more comfortable
 */
app.registerExtension({
    name: "bawk.nodes.textarea.autofocus",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!isBawkNode(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);

            if (!this.widgets) return result;

            for (const widget of this.widgets) {
                if (widget.inputEl && isPromptField(widget.name)) {
                    // Store original rows
                    const normalRows = widget.inputEl.rows || 8;
                    const expandedRows = normalRows * 2;

                    // Expand on focus
                    widget.inputEl.addEventListener('focus', function() {
                        this.rows = expandedRows;
                        this.style.zIndex = '1000'; // Bring to front
                    });

                    // Collapse on blur
                    widget.inputEl.addEventListener('blur', function() {
                        this.rows = normalRows;
                        this.style.zIndex = '';
                    });
                }
            }

            return result;
        };
    }
});

/**
 * Enhancement: Add helpful keyboard shortcuts
 */
app.registerExtension({
    name: "bawk.nodes.keyboard.shortcuts",

    async setup() {
        // Add global keyboard shortcuts for Bawk nodes
        document.addEventListener('keydown', function(e) {
            // Ctrl+Shift+B: Find and focus on Bawk Wildcard Encoder prompt
            if (e.ctrlKey && e.shiftKey && e.key === 'B') {
                const nodes = app.graph?._nodes || [];
                const wildcardNode = nodes.find(n => n.type === "FluxWildcardEncode");

                if (wildcardNode) {
                    const promptWidget = wildcardNode.widgets?.find(w => w.name === "prompt");
                    if (promptWidget && promptWidget.inputEl) {
                        promptWidget.inputEl.focus();
                        e.preventDefault();
                    }
                }
            }
        });

        console.log("[Bawk Nodes] Keyboard shortcuts enabled (Ctrl+Shift+B for prompt focus)");
    }
});

console.log("[Bawk Nodes] All UI extensions loaded successfully");
