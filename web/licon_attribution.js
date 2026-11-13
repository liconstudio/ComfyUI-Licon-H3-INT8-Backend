import { app } from "../../scripts/app.js";

let warned = false;

function warnUnavailable() {
    if (warned) return;
    warned = true;
    console.warn("[Licon] This frontend cannot customize the H3 source badge; inference is unaffected.");
}

app.registerExtension({
    name: "Licon.H3Int8Backend.Attribution",

    beforeRegisterVueAppNodeDefs(defs) {
        try {
            if (!Array.isArray(defs) || typeof window.LiteGraph?.getNodeType !== "function") {
                warnUnavailable();
                return;
            }

            const index = defs.findIndex((def) => def?.name === "H3Int8Backend");
            const nodeDef = window.LiteGraph.getNodeType("H3Int8Backend")?.nodeData;
            if (index === -1 || !nodeDef?.nodeSource) {
                warnUnavailable();
                return;
            }

            // Keep the real module and source identity; customize only badge text.
            nodeDef.nodeSource.badgeText = "Licon";
            defs[index] = nodeDef;
        } catch {
            warnUnavailable();
        }
    },
});
