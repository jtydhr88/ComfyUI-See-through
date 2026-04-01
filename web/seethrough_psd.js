/**
 * SeeThrough PSD Generator
 * Uses ag-psd to generate PSD files in the browser from decomposed layers.
 */

// import { app } from "../../scripts/app.js";

// const { api } = window.comfyAPI.api;
const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

let agPsdLoaded = false;
let agPsdLoadPromise = null;

async function ensureAgPsdLoaded() {
    if (agPsdLoaded) return;
    if (agPsdLoadPromise) return agPsdLoadPromise;

    agPsdLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script");
        script.src = api.fileURL("/extensions/ComfyUI-See-through/ag-psd.bundle.js");
        script.onload = () => {
            agPsdLoaded = true;
            console.log("[SeeThrough] ag-psd bundle loaded");
            resolve();
        };
        script.onerror = (e) => {
            console.error("[SeeThrough] Failed to load ag-psd bundle:", e);
            reject(new Error("Failed to load ag-psd bundle"));
        };
        document.head.appendChild(script);
    });

    return agPsdLoadPromise;
}

function loadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error(`Failed to load: ${url}`));
        img.src = url;
    });
}

async function createPSD(layerInfo, psdType) {
    await ensureAgPsdLoaded();

    const { layers, width, height, prefix, timestamp } = layerInfo;
    const isDepth = psdType === "depth";
    const suffix = isDepth ? "_depth" : "";

    console.log(`[SeeThrough] Creating ${isDepth ? "depth " : ""}PSD: ${width}x${height}, ${layers.length} layers`);

    // Create composite canvas
    const compositeCanvas = document.createElement("canvas");
    compositeCanvas.width = width;
    compositeCanvas.height = height;
    const compositeCtx = compositeCanvas.getContext("2d");

    const psdLayers = [];

    for (const layer of layers) {
        const filenameKey = isDepth ? "depth_filename" : "filename";
        const filename = layer[filenameKey];
        if (!filename) continue;

        const url = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=output&t=${Date.now()}`);
        const img = await loadImage(url);

        const lw = layer.right - layer.left;
        const lh = layer.bottom - layer.top;

        // Create layer canvas at the layer's natural size
        const layerCanvas = document.createElement("canvas");
        layerCanvas.width = lw;
        layerCanvas.height = lh;
        const layerCtx = layerCanvas.getContext("2d");
        layerCtx.drawImage(img, 0, 0, lw, lh);

        // Draw to composite at the correct position
        compositeCtx.drawImage(img, layer.left, layer.top, lw, lh);

        psdLayers.push({
            name: layer.name,
            canvas: layerCanvas,
            left: layer.left,
            top: layer.top,
            right: layer.right,
            bottom: layer.bottom,
            blendMode: "normal",
            opacity: 1,
        });
    }

    const psd = {
        width,
        height,
        canvas: compositeCanvas,
        children: psdLayers,
    };

    const psdBuffer = window.AgPsd.writePsd(psd);
    const blob = new Blob([psdBuffer], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${prefix}_${timestamp}${suffix}.psd`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`[SeeThrough] PSD downloaded: ${prefix}_${timestamp}${suffix}.psd`);
}

app.registerExtension({
    name: "ComfyUI-See-through.SavePSD",

    async nodeCreated(node) {
        if (node.comfyClass !== "SeeThrough_SavePSD") return;

        console.log("[SeeThrough] Setting up frontend PSD generator");

        // Download RGBA PSD button
        const dlBtn = node.addWidget(
            "button",
            "Download PSD",
            "Download PSD",
            async () => {
                try {
                    dlBtn.name = "Generating PSD...";
                    const logResp = await fetch(
                        api.apiURL("/view?filename=seethrough_psd_info.log&type=output&t=" + Date.now())
                    );
                    if (!logResp.ok) {
                        alert("Please run the workflow first to generate layers.");
                        return;
                    }
                    const infoFilename = (await logResp.text()).trim();
                    const infoResp = await fetch(
                        api.apiURL(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`)
                    );
                    if (!infoResp.ok) {
                        alert("Failed to load layer information.");
                        return;
                    }
                    const layerInfo = await infoResp.json();
                    await createPSD(layerInfo, "rgba");
                } catch (error) {
                    console.error("[SeeThrough] Error:", error);
                    alert(`Failed to generate PSD: ${error.message}`);
                } finally {
                    dlBtn.name = "Download PSD";
                }
            }
        );
        dlBtn.color = "#10B981";
        dlBtn.bgcolor = "#059669";

        // Download Depth PSD button
        const dlDepthBtn = node.addWidget(
            "button",
            "Download Depth PSD",
            "Download Depth PSD",
            async () => {
                try {
                    dlDepthBtn.name = "Generating Depth PSD...";
                    const logResp = await fetch(
                        api.apiURL("/view?filename=seethrough_psd_info.log&type=output&t=" + Date.now())
                    );
                    if (!logResp.ok) {
                        alert("Please run the workflow first to generate layers.");
                        return;
                    }
                    const infoFilename = (await logResp.text()).trim();
                    const infoResp = await fetch(
                        api.apiURL(`/view?filename=${encodeURIComponent(infoFilename)}&type=output&t=${Date.now()}`)
                    );
                    if (!infoResp.ok) {
                        alert("Failed to load layer information.");
                        return;
                    }
                    const layerInfo = await infoResp.json();
                    await createPSD(layerInfo, "depth");
                } catch (error) {
                    console.error("[SeeThrough] Error:", error);
                    alert(`Failed to generate depth PSD: ${error.message}`);
                } finally {
                    dlDepthBtn.name = "Download Depth PSD";
                }
            }
        );
        dlDepthBtn.color = "#6366F1";
        dlDepthBtn.bgcolor = "#4F46E5";
    },
});

console.log("[SeeThrough] PSD extension loaded");
