// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[YOLO11s_det] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape after batch strip: [84, 8400]
    // layer0.inferDims: numDims=2, d[0]=84, d[1]=8400
    // 84 = 4 (cx, cy, w, h) + 80 class scores
    // Layout: data[row * numProposals + col] where row in [0..83], col in [0..8399]
    // We need to transpose conceptually: for each proposal j, gather row 0..83.

    const unsigned int featureDim = layer0.inferDims.d[0];  // 84
    const unsigned int numProposals = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        static bool warnOnce = false;
        if (!warnOnce) {
            warnOnce = true;
            fprintf(stderr, "[YOLO11s_det] WARNING: featureDim=%u != 4+numClasses=%u\n",
                    featureDim, 4u + numClasses);
        }
    }

    for (unsigned int j = 0; j < numProposals; ++j) {
        // Find best class score (scores are post-sigmoid probabilities in [0,1])
        float maxScore = 0.0f;
        unsigned int bestClassId = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score = data[(4u + c) * numProposals + j];
            if (score > maxScore) {
                maxScore = score;
                bestClassId = c;
            }
        }

        // Per-class threshold
        float thr = (bestClassId < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClassId]
                     : 0.3f;
        if (maxScore < thr) continue;

        // Decode cx, cy, w, h (in network-input pixel coords)
        float cx = data[0u * numProposals + j];
        float cy = data[1u * numProposals + j];
        float bw = data[2u * numProposals + j];
        float bh = data[3u * numProposals + j];

        // Convert centre+size to top-left corner
        float left = cx - bw * 0.5f;
        float top  = cy - bh * 0.5f;

        // Clamp to network input
        left  = std::max<float>(0.0f, std::min<float>(left, netW));
        top   = std::max<float>(0.0f, std::min<float>(top, netH));
        float right  = std::max<float>(0.0f, std::min<float>(cx + bw * 0.5f, netW));
        float bottom = std::max<float>(0.0f, std::min<float>(cy + bh * 0.5f, netH));

        float finalW = right - left;
        float finalH = bottom - top;

        // Drop degenerate boxes
        if (finalW <= 1.0f || finalH <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClassId;
        obj.left = left;
        obj.top = top;
        obj.width = finalW;
        obj.height = finalH;
        obj.detectionConfidence = maxScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}