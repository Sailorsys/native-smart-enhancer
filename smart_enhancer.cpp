 // ===============================================

#ifndef ENHANCER_NO_JNI
#include <jni.h>
#include <android/bitmap.h>
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cstdint>
#include <limits>
#include <new>
#include <exception>

// --- knobs (all tunable in one place) ---
static const float CURVE_DARK_LUM = 30.0f;
static const float CURVE_FULL_LUM = 90.0f;
static const float CURVE_MIN_STRENGTH = 0.30f;
static const float WB_BRIGHT_LUM = 150.0f;
static const float WB_BRIGHT_FRAC = 0.01f;
static const float WB_BRIGHT_DEV = 8.5f;
static const float WB_GRAY_DEV = 8.0f;
static const float WB_FALLBACK_DEV = 12.0f;
static const float NLM_MIN_SIGMA = 1.5f;        // below: photo is clean, no denoise
static const float NLM_H_K = 0.80f;             // v4.5 (S6-2): was 0.55
static const float NLM_H_MAX = 16.0f;
static const float NLM_TEXTURE_KEEP = 0.35f;
static const float NLM_ORIG_KEEP = 0.12f;        // v4.5 (S6-2): was 0.22
static const float ALIGN_FLOOR_K = 1.0f;         // v4.5 (S6-1): gb floor = K*sigma
static const float FINE_CHROMA_EPS = 140.0f;
static const float FINE_CHROMA_BLEND = 0.85f;
static const int   NLM_MAX_PIXELS = 4000000;
static const int   FAST_LUMA_RADIUS = 2;
static const int   GF_SCALE_EDGE = 320;
static const int   GF_MID_RADIUS = 5;
static const int   GF_RADIUS = 56;
static const float GF_EPS = 420.0f;
static const float CH_DEV_LO = 38.0f;
static const float CH_DEV_HI = 95.0f;
static const float CH_STRENGTH = 0.92f;
static const float CHROMA_GATE_LO = 1.5f;       // v4.1: sigma where chroma cleaning starts
static const float CHROMA_GATE_HI = 4.0f;       // v4.1: sigma where it is fully on
static const float RESTORE_MIN = 0.90f;         // v4.1: chromaRestore bounds
static const float RESTORE_MAX = 1.12f;
static const float RESTORE_TARGET = 1.03f;
static const float LIFT_STRENGTH = 0.30f;
static const float LIFT_GAMMA = 0.78f;
static const float STRUCT_LO = 1.3f;
static const float STRUCT_HI = 3.2f;
static const float CONTRAST_DARK_REF = 35.0f;
static const float CONTRAST_DARK_FLOOR = 0.35f;
static const float CONTRAST_BRIGHT_CAP = 0.80f;
#ifndef PIVOT_CAP
#define PIVOT_CAP 120.0f
#endif  // Pivot cap (2026-11-07): S5 contrast pivot never exceeds this; 120 measured best on the day photo, 140 is the conservative middle.
static const float DEEPBLACK_REF = 35.0f;
static const float DEEPBLACK_FLOOR = 0.35f;
static const float SCORE_ALIGN_NORM = 0.5f;   // v4.2: align normalization
static const float SCORE_COH_NORM = 0.8f;     // v4.2: coherence normalization
static const float SCORE_FULL = 0.15f;        // v4.2: score at full strength
static const int   SCORE_BLOCK = 16;          // v4.2: block size in pixels
static const float ADAPT_H4_K = 1.3f;          // v4.3: correlated-noise calibration
static const float ADAPT_FLOOR = 1.0f;         // v4.3: clean-shading floor
static const float ADAPT_EF_GATE = 2.0f;       // v4.3: fine-smooth block gate
static const float ADAPT_MIN_FRAC = 0.25f;     // v4.3: min smooth-block fraction
static const float CONTRAST_NOISE_CUT = 1.0f;  // v4.4: global contrast cut at gate 1

// Forward declarations (defined later in this file).
float estimateSigma(const cv::Mat &img);
cv::Mat guidedFast(const cv::Mat &I, const cv::Mat &p, int r, float eps, int scaleEdge);
cv::Mat pilContrastMod(const cv::Mat &img, float factor, float darkScale, float brightCap, float pivotCap = 255.0f);

// --- Helper Functions ---

inline float clampf(float v, float minVal, float maxVal) {
    return std::max(minVal, std::min(maxVal, v));
}

inline int clampi(int v, int minVal, int maxVal) {
    return std::max(minVal, std::min(maxVal, v));
}

inline float smoothstep01(float x, float a, float b) {
    float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

void rgbToPillowHsv(float r, float g, float b, float &h, float &s, float &v) {
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;

    float maxc = std::max({rf, gf, bf});
    float minc = std::min({rf, gf, bf});
    v = maxc;
    float d = maxc - minc;
    s = (maxc == 0.0f) ? 0.0f : (d / maxc);

    if (d == 0.0f) {
        h = 0.0f;
    } else if (maxc == rf) {
        h = (gf - bf) / d + (gf < bf ? 6.0f : 0.0f);
    } else if (maxc == gf) {
        h = (bf - rf) / d + 2.0f;
    } else {
        h = (rf - gf) / d + 4.0f;
    }
    h /= 6.0f;

    h *= 255.0f;
    s *= 255.0f;
    v *= 255.0f;
}

void pillowHsvToRgb(float h, float s, float v, float &r, float &g, float &b) {
    float hh = clampf(h / 255.0f, 0.0f, 1.0f);
    float ss = clampf(s / 255.0f, 0.0f, 1.0f);
    float vv = clampf(v / 255.0f, 0.0f, 1.0f);

    if (ss <= 1e-6f) {
        float c = vv * 255.0f;
        r = g = b = c;
        return;
    }

    int i = static_cast<int>(hh * 6.0f);
    float f = hh * 6.0f - i;
    float p = vv * (1.0f - ss);
    float q = vv * (1.0f - f * ss);
    float t = vv * (1.0f - (1.0f - f) * ss);

    float rf, gf, bf;
    switch (i % 6) {
        case 0: rf = vv; gf = t;  bf = p;  break;
        case 1: rf = q;  gf = vv; bf = p;  break;
        case 2: rf = p;  gf = vv; bf = t;  break;
        case 3: rf = p;  gf = q;  bf = vv; break;
        case 4: rf = t;  gf = p;  bf = vv; break;
        default: rf = vv; gf = p; bf = q;  break;
    }

    r = rf * 255.0f;
    g = gf * 255.0f;
    b = bf * 255.0f;
}

cv::Mat gaussianBlurCustom(const cv::Mat &src, float radius) {
    if (radius <= 0.01f) return src.clone();
    float sigma = std::max(0.1f, radius);
    int r = std::max(1, static_cast<int>(sigma * 3.0f));
    int ksize = r * 2 + 1;
    cv::Mat dst;
    cv::GaussianBlur(src, dst, cv::Size(ksize, ksize), sigma, sigma, cv::BORDER_REPLICATE);
    return dst;
}

// Luminance (CV_32FC1) of a CV_32FC3 RGB image.
cv::Mat lumaOf(const cv::Mat &img) {
    int n = img.rows * img.cols;
    cv::Mat L(img.rows, img.cols, CV_32FC1);
    const float* a = img.ptr<float>();
    float* l = L.ptr<float>();
    for (int i = 0; i < n; ++i)
        l[i] = 0.299f * a[i * 3] + 0.587f * a[i * 3 + 1] + 0.114f * a[i * 3 + 2];
    return L;
}

cv::Mat convolve3x3FindEdges(const cv::Mat &grayFloat) {
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f);
    cv::Mat dst;
    cv::filter2D(grayFloat, dst, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // Clamp to 0..255
    for (int i = 0; i < dst.rows * dst.cols; ++i) {
        dst.at<float>(i) = clampf(dst.at<float>(i), 0.0f, 255.0f);
    }
    return dst;
}

cv::Mat findEdgesFromL(const cv::Mat &rgbFloat) {
    return convolve3x3FindEdges(lumaOf(rgbFloat));
}

// Chroma (R-Y, G-Y, B-Y) as CV_32FC3.
static cv::Mat chromaOf(const cv::Mat &img, const cv::Mat &Y) {
    int n = img.rows * img.cols;
    cv::Mat ch(img.rows, img.cols, CV_32FC3);
    const float* a = img.ptr<float>();
    const float* y = Y.ptr<float>();
    float* o = ch.ptr<float>();
    for (int i = 0; i < n; ++i) {
        o[i * 3]     = a[i * 3]     - y[i];
        o[i * 3 + 1] = a[i * 3 + 1] - y[i];
        o[i * 3 + 2] = a[i * 3 + 2] - y[i];
    }
    return ch;
}

// --- Evaluation Struct & Function ---

struct Eval {
    float brightness;
    float contrast;
    float sharpness;
    float noise;
    float saturation;
    bool isGraphic;
    float darkRatio;
    float shadowRatio;
    float brightRatio;
    float highlightRatio;
    float haze;
};

// Analysis contracts for the tiled pipeline. These are deliberately plain
// value types: they own no cv::Mat storage and can be passed between stages
// without extending the lifetime of full-resolution buffers.
struct TileMetrics {
    cv::Rect core;
    float meanLuma = 0.0f;
    float sigmaLumaFine = 0.0f;
    float sigmaLumaCoarse = 0.0f;
    float edgeEnergy = 0.0f;
    float textureConfidence = 0.0f;
    float noiseConfidence = 0.0f;
};

struct NoiseEstimate {
    float sigmaLuma = 0.0f;
    float sigmaChroma1 = 0.0f;
    float sigmaChroma2 = 0.0f;
    float correlatedLuma = 0.0f;
    // Fraction of 16x16 blocks accepted by the fine-detail gate. This is
    // support coverage, NOT a calibrated probability/confidence value.
    float lumaSupportFraction = 0.0f;
    // Kept at zero until confidence is calibrated against labeled test data.
    float confidenceLuma = 0.0f;
    float confidenceChroma = 0.0f;
};

struct GlobalMetrics {
    Eval eval{};
    int sourceWidth = 0;
    int sourceHeight = 0;
    float sourceMeanLuma = 0.0f;
};

Eval evaluate(const cv::Mat &rgbFloat) {
    int w = rgbFloat.cols;
    int h = rgbFloat.rows;
    int n = w * h;

    double sumGray = 0.0;
    double sumGraySq = 0.0;
    int dark = 0, shadow = 0, bright = 0, highlight = 0;

    const float* ptr = rgbFloat.ptr<float>();

    for (int i = 0; i < n; ++i) {
        float r = ptr[i * 3];
        float g = ptr[i * 3 + 1];
        float b = ptr[i * 3 + 2];
        float y = 0.2989f * r + 0.5870f * g + 0.1140f * b;
        sumGray += y;
        sumGraySq += y * y;
        if (y < 45.0f) dark++;
        if (y < 80.0f) shadow++;
        if (y > 235.0f) bright++;
        if (y > 210.0f) highlight++;
    }

    float mean = static_cast<float>(sumGray / n);
    float variance = std::max(0.0f, static_cast<float>(sumGraySq / n - mean * mean));
    float stdDev = std::sqrt(variance);

    float brightness = clampf(mean / 2.55f, 0.0f, 100.0f);
    float contrast = clampf(stdDev / 2.55f, 0.0f, 100.0f);
    float darkRatio = dark / static_cast<float>(n);
    float shadowRatio = shadow / static_cast<float>(n);
    float brightRatio = bright / static_cast<float>(n);
    float highlightRatio = highlight / static_cast<float>(n);

    // Sharpness
    cv::Mat blur12 = gaussianBlurCustom(rgbFloat, 1.2f);
    const float* b12Ptr = blur12.ptr<float>();
    double sumAbs = 0.0;
    for (int i = 0; i < n; ++i) {
        float dr = std::abs(ptr[i * 3] - b12Ptr[i * 3]);
        float dg = std::abs(ptr[i * 3 + 1] - b12Ptr[i * 3 + 1]);
        float db = std::abs(ptr[i * 3 + 2] - b12Ptr[i * 3 + 2]);
        sumAbs += (dr + dg + db) / 3.0;
    }
    float sharpness = clampf(static_cast<float>((sumAbs / n) * 7.0), 0.0f, 100.0f);

    // Saturation
    double sumS = 0.0;
    for (int i = 0; i < n; ++i) {
        float hh, ss, vv;
        rgbToPillowHsv(ptr[i * 3], ptr[i * 3 + 1], ptr[i * 3 + 2], hh, ss, vv);
        sumS += ss;
    }
    float saturation = clampf(static_cast<float>((sumS / n) / 2.55), 0.0f, 100.0f);

    // Noise
    cv::Mat blur10 = gaussianBlurCustom(rgbFloat, 1.0f);
    const float* b10Ptr = blur10.ptr<float>();
    // Preserve the legacy residual-p75 proxy, but avoid allocating and sorting
    // one float per pixel. Residual is bounded to [0,255], so a 256-bin
    // histogram gives a deterministic nearest-rank approximation in O(N).
    // This remains a sharpening heuristic, NOT the authoritative noise model.
    int residualHist[256] = {};
    for (int i = 0; i < n; ++i) {
        float dr = std::abs(ptr[i * 3] - b10Ptr[i * 3]);
        float dg = std::abs(ptr[i * 3 + 1] - b10Ptr[i * 3 + 1]);
        float db = std::abs(ptr[i * 3 + 2] - b10Ptr[i * 3 + 2]);
        int v = clampi(static_cast<int>((dr + dg + db) / 3.0f + 0.5f), 0, 255);
        ++residualHist[v];
    }
    const int p75Rank = clampi(static_cast<int>(n * 0.75f), 0, n - 1);
    int cumulative = 0;
    int p75Value = 0;
    for (; p75Value < 255; ++p75Value) {
        cumulative += residualHist[p75Value];
        if (cumulative > p75Rank) break;
    }
    float noise = clampf(static_cast<float>(p75Value) * 3.0f, 0.0f, 100.0f);

    // Graphic detection was explicitly disabled in v4.3. Do not spend
    // memory/time constructing three full channels and edge maps for a
    // decision that is unconditionally false.
    const bool isGraphic = false;

    // Haze
    float haze = 0.0f;
    if (contrast < 48.0f) haze += (48.0f - contrast) / 48.0f * 0.50f;
    if (brightness > 48.0f && darkRatio < 0.18f) haze += 0.18f;
    if (saturation < 38.0f) haze += (38.0f - saturation) / 38.0f * 0.18f;
    haze = clampf(haze, 0.0f, 1.0f);

    return Eval{brightness, contrast, sharpness, noise, saturation, isGraphic,
                darkRatio, shadowRatio, brightRatio, highlightRatio, haze};
}

// --- Pipeline Stages ---

// v3 WB (unchanged): relative gray test, bright-neutral decision.
cv::Mat whiteBalance(const cv::Mat &img, const Eval &ev) {
    if (ev.isGraphic) return img;

    // Estimate on a small copy (<=640 max edge).
    cv::Mat small = img;
    int maxEdge = std::max(img.cols, img.rows);
    if (maxEdge > 640) {
        float s = 640.0f / maxEdge;
        cv::resize(img, small,
                   cv::Size(std::max(1, static_cast<int>(img.cols * s)),
                            std::max(1, static_cast<int>(img.rows * s))),
                   0, 0, cv::INTER_AREA);
    }

    int n = small.rows * small.cols;
    const float* p = small.ptr<float>();
    double sr = 0.0, sg = 0.0, sb = 0.0;        // global sums (fallback)
    double sgr = 0.0, sgg = 0.0, sgb = 0.0;     // gray-pixel sums
    double sbr = 0.0, sbg = 0.0, sbb = 0.0;     // BRIGHT gray-pixel sums
    int grayCount = 0;
    int brightCount = 0;
    for (int i = 0; i < n; ++i) {
        float r = p[i * 3], g = p[i * 3 + 1], b = p[i * 3 + 2];
        sr += r; sg += g; sb += b;
        float mx = std::max({r, g, b});
        float mn = std::min({r, g, b});
        float lum = 0.299f * r + 0.587f * g + 0.114f * b;
        float grayThr = std::max(25.0f, 0.20f * lum);
        if ((mx - mn) < grayThr && lum >= 40.0f && lum <= 220.0f) {
            sgr += r; sgg += g; sgb += b;
            ++grayCount;
            if (lum >= WB_BRIGHT_LUM) {
                sbr += r; sbg += g; sbb += b;
                ++brightCount;
            }
        }
    }

    float mr, mg, mb, k;
    if (grayCount > 0 && grayCount >= static_cast<int>(n * 0.03f)) {
        if (brightCount >= static_cast<int>(n * WB_BRIGHT_FRAC)) {
            mr = static_cast<float>(sbr / brightCount);
            mg = static_cast<float>(sbg / brightCount);
            mb = static_cast<float>(sbb / brightCount);
            k = 0.70f;
            float bmean = (mr + mg + mb) / 3.0f;
            if (bmean < 5.0f) return img;
            float bdev = std::max({std::abs(mr - bmean), std::abs(mg - bmean), std::abs(mb - bmean)});
            if (bdev < WB_BRIGHT_DEV) return img;
        } else {
            mr = static_cast<float>(sgr / grayCount);
            mg = static_cast<float>(sgg / grayCount);
            mb = static_cast<float>(sgb / grayCount);
            k = 0.70f;
            float gmean = (mr + mg + mb) / 3.0f;
            if (gmean < 5.0f) return img;
            float gdev = std::max({std::abs(mr - gmean), std::abs(mg - gmean), std::abs(mb - gmean)});
            if (gdev < WB_GRAY_DEV) return img;
        }
    } else {
        mr = static_cast<float>(sr / n);
        mg = static_cast<float>(sg / n);
        mb = static_cast<float>(sb / n);
        k = 0.40f;  // fallback: weaker, because global means are untrusted
        float fmean = (mr + mg + mb) / 3.0f;
        if (fmean < 5.0f) return img;
        float fdev = std::max({std::abs(mr - fmean), std::abs(mg - fmean), std::abs(mb - fmean)});
        if (fdev < WB_FALLBACK_DEV) return img;
    }

    float mean = (mr + mg + mb) / 3.0f;
    if (mean < 5.0f) return img;

    float gr = clampf(mean / std::max(1.0f, mr), 0.85f, 1.15f);
    float gg = clampf(mean / std::max(1.0f, mg), 0.85f, 1.15f);
    float gb = clampf(mean / std::max(1.0f, mb), 0.85f, 1.15f);

    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    float* o = out.ptr<float>();
    int px = img.rows * img.cols * 3;
    for (int i = 0; i < px; i += 3) {
        o[i]     = clampf(a[i]     * (1.0f + (gr - 1.0f) * k), 0.0f, 255.0f);
        o[i + 1] = clampf(a[i + 1] * (1.0f + (gg - 1.0f) * k), 0.0f, 255.0f);
        o[i + 2] = clampf(a[i + 2] * (1.0f + (gb - 1.0f) * k), 0.0f, 255.0f);
    }
    return out;
}

// v2 dark-AND-flat mask (unchanged): the dark-area specialist.
cv::Mat buildDarkFlatMask(const cv::Mat &img) {
    cv::Mat half;
    cv::resize(img, half,
               cv::Size(std::max(1, img.cols / 2), std::max(1, img.rows / 2)),
               0, 0, cv::INTER_AREA);

    int w = half.cols, h = half.rows, n = w * h;
    cv::Mat lum(h, w, CV_32FC1);
    const float* p = half.ptr<float>();
    float* l = lum.ptr<float>();
    for (int i = 0; i < n; ++i)
        l[i] = 0.299f * p[i * 3] + 0.587f * p[i * 3 + 1] + 0.114f * p[i * 3 + 2];

    cv::Mat ls = gaussianBlurCustom(lum, 1.0f);
    cv::Mat mean = gaussianBlurCustom(ls, 2.0f);
    cv::Mat lsSq, meanSq;
    cv::multiply(ls, ls, lsSq);
    meanSq = gaussianBlurCustom(lsSq, 2.0f);

    cv::Mat mask(h, w, CV_32FC1);
    const float* lPtr = lum.ptr<float>();
    const float* mPtr = mean.ptr<float>();
    const float* qPtr = meanSq.ptr<float>();
    float* o = mask.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float var = std::max(0.0f, qPtr[i] - mPtr[i] * mPtr[i]);
        float stdDev = std::sqrt(var);
        float darkW = 1.0f - smoothstep01(lPtr[i], 45.0f, 95.0f);
        float flatW = 1.0f - smoothstep01(stdDev, 4.0f, 12.0f);
        o[i] = darkW * flatW;  // soft AND: must be dark AND flat
    }

    cv::Mat feathered = gaussianBlurCustom(mask, 2.5f);
    cv::Mat full;
    cv::resize(feathered, full, img.size(), 0, 0, cv::INTER_LINEAR);
    return full;
}

// v2 dark-flat cleaning. v4.1: strength follows the measured noise gate t
// (t = 0 on a clean photo -> returns the image untouched).
cv::Mat darkFlatClean(const cv::Mat &img, const cv::Mat &mask, float t) {
    if (t <= 0.001f) return img;
    cv::Mat blurred = gaussianBlurCustom(img, 2.0f);
    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    const float* b = blurred.ptr<float>();
    const float* m = mask.ptr<float>();
    float* o = out.ptr<float>();
    int n = img.rows * img.cols;
    for (int i = 0; i < n; ++i) {
        float s = m[i] * 0.55f * t;  // cleaning strength knob (x noise gate)
        float r  = a[i * 3]     * (1.0f - s) + b[i * 3]     * s;
        float g  = a[i * 3 + 1] * (1.0f - s) + b[i * 3 + 1] * s;
        float bl = a[i * 3 + 2] * (1.0f - s) + b[i * 3 + 2] * s;
        float lum = 0.299f * r + 0.587f * g + 0.114f * bl;
        float d = m[i] * 0.45f * t;  // chroma suppression knob (x noise gate)
        o[i * 3]     = clampf(lum + (r - lum)  * (1.0f - d), 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(lum + (g - lum)  * (1.0f - d), 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(lum + (bl - lum) * (1.0f - d), 0.0f, 255.0f);
    }
    return out;
}

// --- v4.2 additions ---

// Per-block mean downsample (INTER_AREA) helper.
cv::Mat blockMean(const cv::Mat &src, int bw, int bh) {
    cv::Mat dst;
    cv::resize(src, dst, cv::Size(bw, bh), 0, 0, cv::INTER_AREA);
    return dst;
}

// v4.2: local noise-confidence map (session-2 decision, transcribed from
// the validated diagnostic port). Returns full-res CV_32FC1 scores in
// [0, 1]; high = noise-like, low = real structure. Two indicators only:
// align (residual/edge alignment) and coh (structure-tensor coherence).
// The r-term and E_fine-presence term were dropped in session 2 (no signal).
// Note: gaussianBlurCustom uses BORDER_REPLICATE where the diagnostic port
// used OpenCV defaults; p75 is nearest-rank here vs interpolated there --
// both differences are sub-threshold for the score thresholds.
cv::Mat buildNoiseScoreMap(const cv::Mat &luma, float gb_floor) {
    int W = luma.cols, H = luma.rows;
    int bw = std::max(1, W / SCORE_BLOCK);
    int bh = std::max(1, H / SCORE_BLOCK);

    cv::Mat g1 = gaussianBlurCustom(luma, 1.0f);
    cv::Mat g4 = gaussianBlurCustom(luma, 4.0f);

    // --- align ---
    cv::Mat gx4, gy4, gb;
    cv::Sobel(g4, gx4, CV_32F, 1, 0, 3);
    cv::Sobel(g4, gy4, CV_32F, 0, 1, 3);
    cv::magnitude(gx4, gy4, gb);
    // Sobel magnitude is bounded for 8-bit-range input. A fixed histogram
    // avoids a full-resolution float vector + nth_element (about 48 MB at
    // 12 MP), while retaining sub-gray-level precision for this normalizer.
    constexpr int ALIGN_HIST_BINS = 4096;
    constexpr float ALIGN_HIST_SCALE = 2.0f; // 0.5 input-level per bin
    uint64_t hist[ALIGN_HIST_BINS] = {};
    const uint64_t totalSamples = static_cast<uint64_t>(W) * H;
    for (int i = 0; i < H; ++i) {
        const float* p = gb.ptr<float>(i);
        for (int j = 0; j < W; ++j) {
            int bin = clampi(static_cast<int>(p[j] * ALIGN_HIST_SCALE + 0.5f),
                             0, ALIGN_HIST_BINS - 1);
            ++hist[bin];
        }
    }
    const uint64_t k75 = static_cast<uint64_t>(0.75 * (totalSamples - 1));
    uint64_t cumulative = 0;
    int p75Bin = 0;
    for (; p75Bin < ALIGN_HIST_BINS - 1; ++p75Bin) {
        cumulative += hist[p75Bin];
        if (cumulative > k75) break;
    }
    float p75 = std::max(static_cast<float>(p75Bin) / ALIGN_HIST_SCALE + gb_floor, 1e-6f);
    cv::Mat w;
    cv::divide(gb, p75, w);
    cv::min(w, 1.0, w);
    cv::Mat resid;
    cv::absdiff(luma, g4, resid);
    cv::Mat numB = blockMean(resid.mul(w), bw, bh);
    cv::Mat denB = blockMean(resid, bw, bh);
    cv::Mat alignB(bh, bw, CV_32FC1);
    for (int i = 0; i < bh; ++i) {
        const float* np = numB.ptr<float>(i);
        const float* dp = denB.ptr<float>(i);
        float* op = alignB.ptr<float>(i);
        for (int j = 0; j < bw; ++j)
            op[j] = (dp[j] > 1e-6f) ? np[j] / dp[j] : 0.0f;
    }

    // --- coh (structure tensor on g1, weighted by gradient magnitude) ---
    cv::Mat gx1, gy1, gm;
    cv::Sobel(g1, gx1, CV_32F, 1, 0, 3);
    cv::Sobel(g1, gy1, CV_32F, 0, 1, 3);
    cv::magnitude(gx1, gy1, gm);
    cv::Mat txx, tyy, txy;
    cv::multiply(gx1, gx1, txx);
    cv::multiply(gy1, gy1, tyy);
    cv::multiply(gx1, gy1, txy);
    cv::Mat jxx = gaussianBlurCustom(txx, 2.0f);
    cv::Mat jyy = gaussianBlurCustom(tyy, 2.0f);
    cv::Mat jxy = gaussianBlurCustom(txy, 2.0f);
    cv::Mat d = jxx - jyy;
    cv::Mat numC = d.mul(d) + 4.0f * jxy.mul(jxy);
    cv::sqrt(numC, numC);
    cv::Mat denC = jxx + jyy + 1e-6f;
    cv::Mat coh;
    cv::divide(numC, denC, coh);
    cv::Mat numW = blockMean(coh.mul(gm), bw, bh);
    cv::Mat denW = blockMean(gm, bw, bh);
    cv::Mat cohB(bh, bw, CV_32FC1);
    for (int i = 0; i < bh; ++i) {
        const float* np = numW.ptr<float>(i);
        const float* dp = denW.ptr<float>(i);
        float* op = cohB.ptr<float>(i);
        for (int j = 0; j < bw; ++j)
            op[j] = (dp[j] > 1e-6f) ? np[j] / dp[j] : 0.0f;
    }

    // --- score ---
    cv::Mat scoreB(bh, bw, CV_32FC1);
    for (int i = 0; i < bh; ++i) {
        const float* ap = alignB.ptr<float>(i);
        const float* cp = cohB.ptr<float>(i);
        float* op = scoreB.ptr<float>(i);
        for (int j = 0; j < bw; ++j) {
            float an = clampf(ap[j] / SCORE_ALIGN_NORM, 0.0f, 1.0f);
            float cn = clampf(cp[j] / SCORE_COH_NORM, 0.0f, 1.0f);
            op[j] = (1.0f - an) * (1.0f - cn);
        }
    }
    cv::Mat score;
    cv::resize(scoreB, score, luma.size(), 0, 0, cv::INTER_LINEAR);
    return score;
}

// v4.5 (S6-3): per-block periodicity of the HF residual r = L - G4.
// Weave (4px period): corr(4) ~= +1, corr(2) ~= -1 -> ~1. Noise (white
// or correlated): ~0 at both lags, or decaying without oscillation ->
// the difference rejects it. Same 16x16 grid as the score map.
cv::Mat buildPeriodMap(const cv::Mat &luma) {
    int W = luma.cols, H = luma.rows;
    int bw = std::max(1, W / SCORE_BLOCK);
    int bh = std::max(1, H / SCORE_BLOCK);
    cv::Mat g4 = gaussianBlurCustom(luma, 4.0f);
    cv::Mat r;
    cv::subtract(luma, g4, r);
    cv::Mat denB = blockMean(r.mul(r), bw, bh);
    auto lagCorr = [&](int lag, bool horizontal) {
        cv::Mat num = cv::Mat::zeros(H, W, CV_32FC1);
        if (horizontal) {
            cv::Mat a = r(cv::Range::all(), cv::Range(0, W - lag));
            cv::Mat b = r(cv::Range::all(), cv::Range(lag, W));
            cv::Mat prod;
            cv::multiply(a, b, prod);
            prod.copyTo(num(cv::Range::all(), cv::Range(0, W - lag)));
        } else {
            cv::Mat a = r(cv::Range(0, H - lag), cv::Range::all());
            cv::Mat b = r(cv::Range(lag, H), cv::Range::all());
            cv::Mat prod;
            cv::multiply(a, b, prod);
            prod.copyTo(num(cv::Range(0, H - lag), cv::Range::all()));
        }
        cv::Mat cb = blockMean(num, bw, bh);
        cv::Mat out(bh, bw, CV_32FC1);
        for (int i = 0; i < bh; ++i) {
            const float* np = cb.ptr<float>(i);
            const float* dp = denB.ptr<float>(i);
            float* op = out.ptr<float>(i);
            for (int j = 0; j < bw; ++j)
                op[j] = (dp[j] > 1e-6f) ? np[j] / dp[j] : 0.0f;
        }
        return out;
    };
    cv::Mat c2 = (lagCorr(2, true) + lagCorr(2, false)) * 0.5f;
    cv::Mat c4 = (lagCorr(4, true) + lagCorr(4, false)) * 0.5f;
    cv::Mat periodB = (c4 - c2) * 0.5f;
    cv::max(periodB, 0.0, periodB);
    cv::min(periodB, 1.0, periodB);
    cv::Mat period;
    cv::resize(periodB, period, luma.size(), 0, 0, cv::INTER_LINEAR);
    return period;
}

// v4.2: luma-only dark-flat cleaning, gated by the noise-score map.
// Chroma is EXACTLY preserved (same delta added to all channels).
cv::Mat darkFlatCleanMap(const cv::Mat &img, const cv::Mat &mask,
                         const cv::Mat &score, const cv::Mat &period) {
    cv::Mat blurred = gaussianBlurCustom(img, 2.0f);
    cv::Mat lumB = lumaOf(blurred);
    cv::Mat lumI = lumaOf(img);
    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    const float* m = mask.ptr<float>();
    const float* s = score.ptr<float>();
    const float* p = period.ptr<float>();
    const float* lb = lumB.ptr<float>();
    const float* li = lumI.ptr<float>();
    float* o = out.ptr<float>();
    int n = img.rows * img.cols;
    for (int i = 0; i < n; ++i) {
        float strength = m[i] * 0.55f * clampf(s[i] / SCORE_FULL, 0.0f, 1.0f);
        strength *= (1.0f - p[i]);  // v4.5 (S6-3): texture guard
        float d = (lb[i] - li[i]) * strength;
        o[i * 3]     = clampf(a[i * 3]     + d, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(a[i * 3 + 1] + d, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(a[i * 3 + 2] + d, 0.0f, 255.0f);
    }
    return out;
}

// --- v4.3 additions ---

// 16x16-block RMS via INTER_AREA means of squares.
cv::Mat blockRMS(const cv::Mat &x) {
    int bw = std::max(1, x.cols / 16);
    int bh = std::max(1, x.rows / 16);
    cv::Mat sq;
    cv::multiply(x, x, sq);
    cv::Mat d = blockMean(sq, bw, bh);
    cv::sqrt(d, d);
    return d;
}

struct SigmaInfo {
    float sigma, imm, gated, corr, lowFrac;
};

// v4.3 (F2): adaptive sigma -- see header for the fitted rationale.
SigmaInfo estimateSigmaAdaptive(const cv::Mat &img) {
    SigmaInfo si;
    si.imm = estimateSigma(img);
    cv::Mat L = lumaOf(img);
    cv::Mat g1 = gaussianBlurCustom(L, 1.0f);
    cv::Mat g4 = gaussianBlurCustom(L, 4.0f);
    cv::Mat ef = blockRMS(L - g1);
    cv::Mat h4 = blockRMS(L - g4);
    std::vector<float> vals;
    int total = 0;
    for (int i = 0; i < ef.rows; ++i) {
        const float* ep = ef.ptr<float>(i);
        const float* hp = h4.ptr<float>(i);
        for (int j = 0; j < ef.cols; ++j) {
            ++total;
            if (ep[j] < ADAPT_EF_GATE) vals.push_back(hp[j]);
        }
    }
    si.lowFrac = total ? (float)vals.size() / (float)total : 0.0f;
    if (si.lowFrac >= ADAPT_MIN_FRAC && !vals.empty()) {
        size_t mid = vals.size() / 2;
        std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
        si.gated = vals[mid];
    } else {
        si.gated = 0.0f;
    }
    si.corr = std::max(0.0f, ADAPT_H4_K * si.gated - ADAPT_FLOOR);
    si.sigma = std::max(si.imm, si.corr);
    return si;
}

// Immerkaer estimate on a single channel (for chroma noise).
float immerkaerChan(const cv::Mat &x) {
    static const float kdata[9] = {1, -2, 1, -2, 4, -2, 1, -2, 1};
    cv::Mat k(3, 3, CV_32F, (void*)kdata);
    cv::Mat resp;
    cv::filter2D(x, resp, CV_32F, k);
    cv::Mat absr;
    cv::absdiff(resp, cv::Scalar(0), absr);
    int bw = std::max(1, x.cols / 16);
    int bh = std::max(1, x.rows / 16);
    cv::Mat bm = blockMean(absr, bw, bh);
    std::vector<float> vals;
    for (int i = 0; i < bm.rows; ++i) {
        const float* p = bm.ptr<float>(i);
        for (int j = 0; j < bm.cols; ++j) vals.push_back(p[j]);
    }
    size_t q = (size_t)(0.25 * (vals.size() - 1));
    std::nth_element(vals.begin(), vals.begin() + q, vals.end());
    return vals[q] * 1.2533141f / 6.0f;
}

// Cb (idx 1) or Cr (idx 2) channel of an RGB float image (no +128 offset;
// the Laplacian is invariant to it).
cv::Mat chromaChannel(const cv::Mat &img, int idx) {
    int n = img.rows * img.cols;
    cv::Mat ch(img.rows, img.cols, CV_32FC1);
    const float* a = img.ptr<float>();
    float* o = ch.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float r = a[i * 3], g = a[i * 3 + 1], b = a[i * 3 + 2];
        o[i] = (idx == 1) ? (-0.1687f * r - 0.3313f * g + 0.5f * b)
                          : (0.5f * r - 0.4187f * g - 0.0813f * b);
    }
    return ch;
}

// Orthonormal opponent-color coordinates. They are used for noise measurement
// only; the existing reconstruction/cleaning path remains unchanged. Input
// RGB is currently sRGB-like code values, so these are not claimed to be
// physically linear-light noise estimates.
static void opponentChromaChannels(const cv::Mat &img, cv::Mat &c1, cv::Mat &c2) {
    c1.create(img.size(), CV_32FC1);
    c2.create(img.size(), CV_32FC1);
    const float *src = img.ptr<float>();
    float *p1 = c1.ptr<float>();
    float *p2 = c2.ptr<float>();
    const int n = img.rows * img.cols;
    constexpr float INV_SQRT2 = 0.70710678118f;
    constexpr float INV_SQRT6 = 0.40824829046f;
    for (int i = 0; i < n; ++i) {
        const float r = src[i * 3];
        const float g = src[i * 3 + 1];
        const float b = src[i * 3 + 2];
        p1[i] = (r - g) * INV_SQRT2;
        p2[i] = (r + g - 2.0f * b) * INV_SQRT6;
    }
}

// v4.3 (F2): fine-chroma-only cleaning (same guided params as NLM's fine
// chroma). Runs only when luma NLM is skipped but chroma is noisy.
cv::Mat chromaFineClean(const cv::Mat &img, float gate) {
    cv::Mat Y = lumaOf(img);
    cv::Mat chroma = chromaOf(img, Y);
    cv::Mat cs = guidedFast(Y, chroma, 3, FINE_CHROMA_EPS, 1 << 20);
    float blend = FINE_CHROMA_BLEND * gate;
    int n = img.rows * img.cols;
    cv::Mat out(img.rows, img.cols, CV_32FC3);
    const float* y = Y.ptr<float>();
    const float* c = chroma.ptr<float>();
    const float* s = cs.ptr<float>();
    float* o = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        o[i * 3]     = clampf(y[i] + c[i * 3]     * (1.0f - blend) + s[i * 3]     * blend, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(y[i] + c[i * 3 + 1] * (1.0f - blend) + s[i * 3 + 1] * blend, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(y[i] + c[i * 3 + 2] * (1.0f - blend) + s[i * 3 + 2] * blend, 0.0f, 255.0f);
    }
    return out;
}

// v4.3 (F2): guided luma smoothing for CORRELATED noise, masked to
// fine-smooth blocks (E_fine < 2.0). Chroma is rebuilt around the new luma,
// so each pixel's chroma is exactly preserved.
cv::Mat corrLumaClean(const cv::Mat &img, float corrSigma) {
    cv::Mat Y = lumaOf(img);
    cv::Mat g1 = gaussianBlurCustom(Y, 1.0f);
    cv::Mat ef = blockRMS(Y - g1);
    cv::Mat lowB(ef.rows, ef.cols, CV_32FC1);
    for (int i = 0; i < ef.rows; ++i) {
        const float* ep = ef.ptr<float>(i);
        float* op = lowB.ptr<float>(i);
        for (int j = 0; j < ef.cols; ++j) op[j] = (ep[j] < ADAPT_EF_GATE) ? 1.0f : 0.0f;
    }
    cv::Mat lowFull;
    cv::resize(lowB, lowFull, Y.size(), 0, 0, cv::INTER_LINEAR);
    float e = 1.6f * corrSigma;
    cv::Mat Yd = guidedFast(Y, Y, 2, e * e, 1 << 20);
    float gate = smoothstep01(corrSigma, CHROMA_GATE_LO, CHROMA_GATE_HI);
    int n = img.rows * img.cols;
    cv::Mat chroma = chromaOf(img, Y);
    cv::Mat out(img.rows, img.cols, CV_32FC3);
    const float* y = Y.ptr<float>();
    const float* yd = Yd.ptr<float>();
    const float* lf = lowFull.ptr<float>();
    const float* c = chroma.ptr<float>();
    float* o = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float b = 0.70f * gate * lf[i];
        float yn = y[i] * (1.0f - b) + yd[i] * b;
        o[i * 3]     = clampf(yn + c[i * 3],     0.0f, 255.0f);
        o[i * 3 + 1] = clampf(yn + c[i * 3 + 1], 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(yn + c[i * 3 + 2], 0.0f, 255.0f);
    }
    return out;
}

// v4.3 (F3): sharpMask with sharpening gain suppressed where the residual
// noise map says noise (same structure gate and darkMask as buildSharpMask).
cv::Mat buildSharpMaskScore(const cv::Mat &img, float sigma,
                            const cv::Mat &darkMask, const cv::Mat &score) {
    int n = img.rows * img.cols;
    cv::Mat L = lumaOf(img);
    cv::Mat Lb = gaussianBlurCustom(L, 1.2f);
    cv::Mat d(img.rows, img.cols, CV_32FC1);
    {
        const float* a = L.ptr<float>();
        const float* b = Lb.ptr<float>();
        float* o = d.ptr<float>();
        for (int i = 0; i < n; ++i) o[i] = std::abs(a[i] - b[i]);
    }
    cv::Mat structure = gaussianBlurCustom(d, 1.5f);
    float sg = std::max(sigma * 0.5f, 0.8f);
    cv::Mat mask(img.rows, img.cols, CV_32FC1);
    const float* st = structure.ptr<float>();
    const float* dm = darkMask.ptr<float>();
    const float* sc = score.ptr<float>();
    float* o = mask.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float gain = smoothstep01(st[i], STRUCT_LO * sg, STRUCT_HI * sg) * (1.0f - dm[i]) * (1.0f - sc[i]);
        o[i] = 1.0f - gain;
    }
    return mask;
}

// --- v4.4 additions ---

// Per-pixel-factor version of pilContrastMod (same pivot/mean logic).
cv::Mat pilContrastModMap(const cv::Mat &img, const cv::Mat &fmap,
                          float darkScale, float brightCap, float pivotCap = 255.0f) {
    int px = img.rows * img.cols;
    const float* a = img.ptr<float>();
    double sum = 0.0;
    for (int i = 0; i < px; ++i)
        sum += 0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2];
    float mean = std::min(clampf(static_cast<float>(sum / px), 0.0f, 255.0f), pivotCap);
    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();
    const float* fm = fmap.ptr<float>();
    for (int i = 0; i < px; ++i) {
        float lum = 0.299f * a[i * 3] + 0.587f * a[i * 3 + 1] + 0.114f * a[i * 3 + 2];
        float fp = fm[i];
        float f = (lum < mean) ? (1.0f + (fp - 1.0f) * darkScale)
                               : (1.0f + (fp - 1.0f) * brightCap);
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            o[idx] = clampf(mean * (1.0f - f) + a[idx] * f, 0.0f, 255.0f);
        }
    }
    return out;
}

// v4.4 (S5): noise-aware contrast. Original adaptiveContrast untouched.
cv::Mat adaptiveContrastS5(const cv::Mat &img, const Eval &ev,
                           float inputGate, const cv::Mat &score) {
    float c = ev.contrast;
    float factor = 1.0f;
    if (c < 30.0f) factor = 1.0f + (30.0f - c) / 50.0f;
    else if (c < 45.0f) factor = 1.0f + (45.0f - c) / 180.0f;
    factor = 1.0f + (factor - 1.0f) * 0.82f;
    if (factor <= 1.001f) return img;
    float darkScale = clampf(ev.brightness / CONTRAST_DARK_REF, CONTRAST_DARK_FLOOR, 1.0f);
    float fg = 1.0f + (factor - 1.0f) * (1.0f - CONTRAST_NOISE_CUT * inputGate);
    if (fg <= 1.001f) return img;
    if (score.empty()) return pilContrastMod(img, fg, darkScale, CONTRAST_BRIGHT_CAP, PIVOT_CAP);
    cv::Mat fmap(img.rows, img.cols, CV_32FC1);
    const float* sc = score.ptr<float>();
    float* fp = fmap.ptr<float>();
    for (int i = 0; i < img.rows * img.cols; ++i)
        fp[i] = 1.0f + (fg - 1.0f) * (1.0f - sc[i]);
    return pilContrastModMap(img, fmap, darkScale, CONTRAST_BRIGHT_CAP, PIVOT_CAP);
}

// --- v4 additions ---

// Mean luminance on a small copy (<=640 max edge): sets the curve strength.
float quickMeanLum(const cv::Mat &img) {
    cv::Mat small = img;
    int maxEdge = std::max(img.cols, img.rows);
    if (maxEdge > 640) {
        float s = 640.0f / maxEdge;
        cv::resize(img, small,
                   cv::Size(std::max(1, static_cast<int>(img.cols * s)),
                            std::max(1, static_cast<int>(img.rows * s))),
                   0, 0, cv::INTER_AREA);
    }
    int n = small.rows * small.cols;
    const float* p = small.ptr<float>();
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += 0.299 * p[i * 3] + 0.587 * p[i * 3 + 1] + 0.114 * p[i * 3 + 2];
    return static_cast<float>(sum / n);
}

float curveStrengthFor(float meanLum) {
    return clampf(CURVE_MIN_STRENGTH +
                  (1.0f - CURVE_MIN_STRENGTH) * (meanLum - CURVE_DARK_LUM) /
                  (CURVE_FULL_LUM - CURVE_DARK_LUM),
                  CURVE_MIN_STRENGTH, 1.0f);
}

// Robust luma noise sigma (0..255 scale). Immerkaer Laplacian response,
// averaged per 16x16-ish block; the 25th-percentile block (a flat one)
// gives the noise floor: sigma = value * sqrt(pi/2) / 6.
float estimateSigma(const cv::Mat &img) {
    cv::Mat L = lumaOf(img);
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
         1.0f, -2.0f,  1.0f,
        -2.0f,  4.0f, -2.0f,
         1.0f, -2.0f,  1.0f);
    cv::Mat r;
    cv::filter2D(L, r, CV_32F, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    float* rp = r.ptr<float>();
    int n = r.rows * r.cols;
    for (int i = 0; i < n; ++i) rp[i] = std::abs(rp[i]);

    int bw = std::max(1, L.cols / 16);
    int bh = std::max(1, L.rows / 16);
    cv::Mat rb;
    cv::resize(r, rb, cv::Size(bw, bh), 0, 0, cv::INTER_AREA);
    const float* rbp = rb.ptr<float>();
    std::vector<float> v(rbp, rbp + bw * bh);
    size_t k = (v.size() - 1) / 4;  // 25th percentile
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k] * 1.2533141f / 6.0f;  // sqrt(pi/2) / 6
}

static cv::Mat boxF(const cv::Mat &x, int k) {
    cv::Mat d;
    cv::boxFilter(x, d, -1, cv::Size(k, k), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    return d;
}

// Fast guided filter (He et al.). I = CV_32FC1 guide at full resolution,
// p = CV_32FC1 or CV_32FC3 input. a,b are computed on a copy whose max
// edge is scaleEdge (or full size when scaleEdge >= image size) and
// applied at full resolution. Returns the same channel count as p.
cv::Mat guidedFast(const cv::Mat &I, const cv::Mat &p, int r, float eps, int scaleEdge) {
    int H = I.rows, W = I.cols;
    float s = std::min(1.0f, static_cast<float>(scaleEdge) / static_cast<float>(std::max(H, W)));
    int w2 = std::max(1, static_cast<int>(W * s));
    int h2 = std::max(1, static_cast<int>(H * s));
    bool scaled = (w2 != W || h2 != H);

    cv::Mat Is;
    if (scaled) cv::resize(I, Is, cv::Size(w2, h2), 0, 0, cv::INTER_AREA);
    else Is = I;

    int k = 2 * r + 1;
    int n2 = w2 * h2;
    cv::Mat IsSq(h2, w2, CV_32FC1);
    {
        const float* a = Is.ptr<float>();
        float* o = IsSq.ptr<float>();
        for (int i = 0; i < n2; ++i) o[i] = a[i] * a[i];
    }
    cv::Mat mI = boxF(Is, k);
    cv::Mat mII = boxF(IsSq, k);
    cv::Mat vI(h2, w2, CV_32FC1);
    {
        const float* m1 = mI.ptr<float>();
        const float* m2 = mII.ptr<float>();
        float* o = vI.ptr<float>();
        for (int i = 0; i < n2; ++i) o[i] = m2[i] - m1[i] * m1[i];
    }

    std::vector<cv::Mat> chans;
    cv::split(p, chans);
    std::vector<cv::Mat> outs(chans.size());

    for (size_t c = 0; c < chans.size(); ++c) {
        cv::Mat ps;
        if (scaled) cv::resize(chans[c], ps, cv::Size(w2, h2), 0, 0, cv::INTER_AREA);
        else ps = chans[c];

        cv::Mat Ip(h2, w2, CV_32FC1);
        {
            const float* a = Is.ptr<float>();
            const float* b = ps.ptr<float>();
            float* o = Ip.ptr<float>();
            for (int i = 0; i < n2; ++i) o[i] = a[i] * b[i];
        }
        cv::Mat mp = boxF(ps, k);
        cv::Mat mIp = boxF(Ip, k);

        cv::Mat A(h2, w2, CV_32FC1), B(h2, w2, CV_32FC1);
        {
            const float* mi = mI.ptr<float>();
            const float* mpp = mp.ptr<float>();
            const float* mip = mIp.ptr<float>();
            const float* vv = vI.ptr<float>();
            float* ao = A.ptr<float>();
            float* bo = B.ptr<float>();
            for (int i = 0; i < n2; ++i) {
                float cov = mip[i] - mi[i] * mpp[i];
                float a = cov / (vv[i] + eps);
                ao[i] = a;
                bo[i] = mpp[i] - a * mi[i];
            }
        }
        cv::Mat mA = boxF(A, k);
        cv::Mat mB = boxF(B, k);
        if (scaled) {
            cv::Mat tA, tB;
            cv::resize(mA, tA, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
            cv::resize(mB, tB, cv::Size(W, H), 0, 0, cv::INTER_LINEAR);
            mA = tA; mB = tB;
        }
        outs[c].create(H, W, CV_32FC1);
        const float* ap = mA.ptr<float>();
        const float* bp = mB.ptr<float>();
        const float* ip = I.ptr<float>();
        float* op = outs[c].ptr<float>();
        int n = H * W;
        for (int i = 0; i < n; ++i) op[i] = ap[i] * ip[i] + bp[i];
    }

    cv::Mat out;
    cv::merge(outs, out);
    return out;
}

// v4 luma denoise: classical non-local means on LUMA ONLY, strength from the
// measured sigma; a uniform share of the original luma is kept (no
// staircase artefacts) and real texture is re-injected where structure
// exists. Fine chroma noise: small-radius luma-guided filter.
cv::Mat nlmDenoise(const cv::Mat &img, float sigma) {
    if (sigma < NLM_MIN_SIGMA) return img;  // clean photo: do not touch

    int rows = img.rows, cols = img.cols, n = rows * cols;
    float t = smoothstep01(sigma, 5.0f, 10.0f);   // 0 moderate noise .. 1 heavy noise
    float h = clampf((NLM_H_K + 0.30f * t) * sigma, 1.5f, NLM_H_MAX);
    float origKeep = NLM_ORIG_KEEP * (1.0f - 0.65f * t);

    cv::Mat Y = lumaOf(img);
    cv::Mat Yd(rows, cols, CV_32FC1);
    if (n > NLM_MAX_PIXELS) {
        // fast path for big photos: edge-preserving guided filter on luma
        float e = 1.6f * sigma;
        Yd = guidedFast(Y, Y, FAST_LUMA_RADIUS, e * e, 1 << 20);
    } else {
        cv::Mat Y8(rows, cols, CV_8UC1);
        const float* yp = Y.ptr<float>();
        uint8_t* y8 = Y8.ptr<uint8_t>();
        for (int i = 0; i < n; ++i)
            y8[i] = static_cast<uint8_t>(clampi(static_cast<int>(yp[i] + 0.5f), 0, 255));
        cv::Mat D8;
        cv::fastNlMeansDenoising(Y8, D8, h, 7, 21);
        const uint8_t* d8 = D8.ptr<uint8_t>();
        float* ydp = Yd.ptr<float>();
        for (int i = 0; i < n; ++i) ydp[i] = static_cast<float>(d8[i]);
    }

    // structure map (where real texture exists)
    cv::Mat Yb = gaussianBlurCustom(Y, 1.2f);
    cv::Mat absd(rows, cols, CV_32FC1);
    {
        const float* a = Y.ptr<float>();
        const float* b = Yb.ptr<float>();
        float* o = absd.ptr<float>();
        for (int i = 0; i < n; ++i) o[i] = std::abs(a[i] - b[i]);
    }
    cv::Mat structure = gaussianBlurCustom(absd, 1.5f);

    cv::Mat chroma = chromaOf(img, Y);
    cv::Mat cs = guidedFast(Y, chroma, 3, FINE_CHROMA_EPS, 1 << 20);

    cv::Mat out(rows, cols, CV_32FC3);
    const float* yp = Y.ptr<float>();
    float* ydp = Yd.ptr<float>();
    const float* stp = structure.ptr<float>();
    const float* chp = chroma.ptr<float>();
    const float* csp = cs.ptr<float>();
    float* o = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float y = yp[i];
        float yd = ydp[i];
        yd = yd + origKeep * (y - yd);
        float keepW = NLM_TEXTURE_KEEP * smoothstep01(stp[i], 1.6f * sigma, 3.2f * sigma);
        yd = yd + keepW * (y - yd);
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            float chr = chp[idx] * (1.0f - FINE_CHROMA_BLEND) + csp[idx] * FINE_CHROMA_BLEND;
            o[idx] = clampf(yd + chr, 0.0f, 255.0f);
        }
    }
    return out;
}

// v4: removes LARGE colour blotches only. Flattens the chroma band between a
// mid scale and a big scale (both luma-guided, so luma edges are kept) and
// leaves finer chroma detail alone. Strong deviations = real colour = kept.
// v4.1: strength follows the measured noise gate t (0 = clean photo = skip).
cv::Mat chromaBlotchClean(const cv::Mat &img, float t) {
    if (t <= 0.001f) return img;
    int n = img.rows * img.cols;
    cv::Mat Y = lumaOf(img);
    cv::Mat chroma = chromaOf(img, Y);
    cv::Mat low = guidedFast(Y, chroma, GF_MID_RADIUS, GF_EPS, GF_SCALE_EDGE);
    cv::Mat big = guidedFast(Y, chroma, GF_RADIUS, GF_EPS * 2.0f, GF_SCALE_EDGE);

    cv::Mat out(img.size(), CV_32FC3);
    const float* yp = Y.ptr<float>();
    const float* chp = chroma.ptr<float>();
    const float* lp = low.ptr<float>();
    const float* bp = big.ptr<float>();
    float* o = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float b0 = lp[i * 3]     - bp[i * 3];
        float b1 = lp[i * 3 + 1] - bp[i * 3 + 1];
        float b2 = lp[i * 3 + 2] - bp[i * 3 + 2];
        float dev = std::sqrt(b0 * b0 + b1 * b1 + b2 * b2);
        float w = CH_STRENGTH * t * (1.0f - smoothstep01(dev, CH_DEV_LO, CH_DEV_HI));
        o[i * 3]     = clampf(yp[i] + chp[i * 3]     - b0 * w, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(yp[i] + chp[i * 3 + 1] - b1 * w, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(yp[i] + chp[i * 3 + 2] - b2 * w, 0.0f, 255.0f);
    }
    return out;
}

// v4: halo-free local shadow lift. Only the guided BASE layer is lifted;
// detail stays untouched and deep blacks (base < ~14) are never lifted.
// v4.1: applied as a GAIN, so R:G:B ratios (saturation, warmth) are kept.
cv::Mat localShadowLift(const cv::Mat &img) {
    int n = img.rows * img.cols;
    cv::Mat Y = lumaOf(img);
    cv::Mat base = guidedFast(Y, Y, 8, 300.0f, 480);
    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    const float* bp = base.ptr<float>();
    float* o = out.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float b = bp[i];
        float tt = clampf(b / 255.0f, 0.0f, 1.0f);
        float lifted = 255.0f * std::pow(tt, LIFT_GAMMA);
        float gate = smoothstep01(b, 14.0f, 44.0f) * (1.0f - smoothstep01(b, 105.0f, 175.0f));
        float base2 = b + LIFT_STRENGTH * gate * (lifted - b);
        float f = clampf(base2 / std::max(b, 1.0f), 1.0f, 1.6f);
        o[i * 3]     = clampf(a[i * 3]     * f, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(a[i * 3 + 1] * f, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(a[i * 3 + 2] * f, 0.0f, 255.0f);
    }
    return out;
}

// v4.1: two-sided saturation anchor. Compares the mid-tone chroma-to-luma
// of the output with the pre-cleaning image (both low-passed so noise does
// not inflate it) and rescales chroma about luma by ONE global factor so
// the result lands at ~RESTORE_TARGET x the original. Hue is never changed.
cv::Mat chromaRestore(const cv::Mat &out, const cv::Mat &ref) {
    int n = out.rows * out.cols;
    cv::Mat Yo = lumaOf(out);
    cv::Mat Yr = lumaOf(ref);
    cv::Mat co = gaussianBlurCustom(chromaOf(out, Yo), 8.0f);
    cv::Mat cr = gaussianBlurCustom(chromaOf(ref, Yr), 8.0f);
    cv::Mat yo = gaussianBlurCustom(Yo, 8.0f);
    cv::Mat yr = gaussianBlurCustom(Yr, 8.0f);
    const float* cop = co.ptr<float>();
    const float* crp = cr.ptr<float>();
    const float* yop = yo.ptr<float>();
    const float* yrp = yr.ptr<float>();
    std::vector<float> ratios;
    ratios.reserve(n / 4 + 1);
    for (int i = 0; i < n; i += 4) {  // every 4th pixel is plenty for a median
        float so = std::sqrt(cop[i * 3] * cop[i * 3] + cop[i * 3 + 1] * cop[i * 3 + 1] +
                             cop[i * 3 + 2] * cop[i * 3 + 2]) / std::max(yop[i], 10.0f);
        float sr = std::sqrt(crp[i * 3] * crp[i * 3] + crp[i * 3 + 1] * crp[i * 3 + 1] +
                             crp[i * 3 + 2] * crp[i * 3 + 2]) / std::max(yrp[i], 10.0f);
        if (yrp[i] > 40.0f && yrp[i] < 215.0f && sr > 0.03f) ratios.push_back(so / sr);
    }
    if (ratios.size() < 500) return out;  // not enough mid-tone colour
    size_t mid = ratios.size() / 2;
    std::nth_element(ratios.begin(), ratios.begin() + mid, ratios.end());
    float ratio = ratios[mid];
    float k = clampf(RESTORE_TARGET / std::max(ratio, 1e-3f), RESTORE_MIN, RESTORE_MAX);
    if (std::abs(k - 1.0f) < 0.0005f) return out;

    cv::Mat res(out.size(), CV_32FC3);
    const float* a = out.ptr<float>();
    const float* y = Yo.ptr<float>();
    float* o = res.ptr<float>();
    for (int i = 0; i < n; ++i) {
        o[i * 3]     = clampf(y[i] + (a[i * 3]     - y[i]) * k, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(y[i] + (a[i * 3 + 1] - y[i]) * k, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(y[i] + (a[i * 3 + 2] - y[i]) * k, 0.0f, 255.0f);
    }
    return res;
}

// v4: SUPPRESSION mask for clarity / micro-detail / sharpening, CV_32FC1 in
// [0,1]. 0 = sharpen freely (real structure well above the measured noise),
// 1 = do not sharpen (flat/noisy zone, or inside the dark-flat mask).
cv::Mat buildSharpMask(const cv::Mat &img, float sigma, const cv::Mat &darkMask) {
    int n = img.rows * img.cols;
    cv::Mat L = lumaOf(img);
    cv::Mat Lb = gaussianBlurCustom(L, 1.2f);
    cv::Mat d(img.rows, img.cols, CV_32FC1);
    {
        const float* a = L.ptr<float>();
        const float* b = Lb.ptr<float>();
        float* o = d.ptr<float>();
        for (int i = 0; i < n; ++i) o[i] = std::abs(a[i] - b[i]);
    }
    cv::Mat structure = gaussianBlurCustom(d, 1.5f);
    float sg = std::max(sigma * 0.5f, 0.8f);
    cv::Mat mask(img.rows, img.cols, CV_32FC1);
    const float* st = structure.ptr<float>();
    const float* dm = darkMask.ptr<float>();
    float* o = mask.ptr<float>();
    for (int i = 0; i < n; ++i) {
        float gain = smoothstep01(st[i], STRUCT_LO * sg, STRUCT_HI * sg) * (1.0f - dm[i]);
        o[i] = 1.0f - gain;
    }
    return mask;
}

// Clarity: strength scaled per pixel by (1 - mask).
cv::Mat clarityBoost(const cv::Mat &img, const Eval &ev, const cv::Mat &mask) {
    cv::Mat blurred = gaussianBlurCustom(img, 3.0f);
    float strength = 0.055f; // NATURAL mode
    if (ev.contrast > 62.0f) strength *= 0.72f;
    if (ev.isGraphic) strength *= 0.72f;

    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    const float* b = blurred.ptr<float>();
    const float* m = mask.ptr<float>();
    float* o = out.ptr<float>();
    int n = img.rows * img.cols;
    for (int i = 0; i < n; ++i) {
        float s = strength * (1.0f - m[i]);
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            float detail = clampf(a[idx] - b[idx], -30.0f, 30.0f);
            o[idx] = clampf(a[idx] + detail * s, 0.0f, 255.0f);
        }
    }
    return out;
}

// Micro-detail: own Laplacian edge mask, additionally scaled by (1 - mask).
cv::Mat microDetail(const cv::Mat &img, const Eval &ev, const cv::Mat &mask) {
    cv::Mat blurred = gaussianBlurCustom(img, 1.0f);
    cv::Mat grayEdges = findEdgesFromL(img);

    int n = img.rows * img.cols;
    std::vector<float> edgeMask(n);
    const float* geP = grayEdges.ptr<float>();
    const float* mkP = mask.ptr<float>();
    for (int i = 0; i < n; ++i) {
        edgeMask[i] = clampf((geP[i] / 255.0f - 0.08f) * 3.0f, 0.0f, 1.0f);
    }

    float strength = 0.14f; // NATURAL
    if (ev.sharpness > 45.0f) strength *= 0.80f;
    if (ev.noise > 60.0f) strength *= 0.75f;

    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    const float* b = blurred.ptr<float>();
    float* o = out.ptr<float>();

    for (int i = 0; i < n; ++i) {
        float m = edgeMask[i] * (1.0f - mkP[i]);
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            float detail = clampf(a[idx] - b[idx], -18.0f, 18.0f);
            o[idx] = clampf(a[idx] + detail * strength * m, 0.0f, 255.0f);
        }
    }
    return out;
}

// Contrast around the image mean with modulated push (v3, unchanged).
cv::Mat pilContrastMod(const cv::Mat &img, float factor, float darkScale, float brightCap, float pivotCap) {
    int px = img.rows * img.cols;
    const float* a = img.ptr<float>();
    double sum = 0.0;
    for (int i = 0; i < px; ++i) {
        sum += 0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2];
    }
    float mean = std::min(clampf(static_cast<float>(sum / px), 0.0f, 255.0f), pivotCap);
    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();
    for (int i = 0; i < px; ++i) {
        float lum = 0.299f * a[i * 3] + 0.587f * a[i * 3 + 1] + 0.114f * a[i * 3 + 2];
        float f = (lum < mean) ? (1.0f + (factor - 1.0f) * darkScale)
                               : (1.0f + (factor - 1.0f) * brightCap);
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            o[idx] = clampf(mean * (1.0f - f) + a[idx] * f, 0.0f, 255.0f);
        }
    }
    return out;
}

cv::Mat adaptiveContrast(const cv::Mat &img, const Eval &ev) {
    float c = ev.contrast;
    float factor = 1.0f;
    if (c < 30.0f) factor = 1.0f + (30.0f - c) / 50.0f;
    else if (c < 45.0f) factor = 1.0f + (45.0f - c) / 180.0f;

    factor = 1.0f + (factor - 1.0f) * 0.82f; // NATURAL
    if (factor <= 1.001f) return img;
    float darkScale = clampf(ev.brightness / CONTRAST_DARK_REF, CONTRAST_DARK_FLOOR, 1.0f);
    return pilContrastMod(img, factor, darkScale, CONTRAST_BRIGHT_CAP);
}

cv::Mat colorDepth(const cv::Mat &img, const Eval &ev, const cv::Mat &mask) {
    (void)ev;
    int px = img.rows * img.cols;
    const float* a = img.ptr<float>();

    // Two-pass HSV conversion avoids retaining three floats per pixel.
    // At 12 MP the old tuple vector cost roughly 144 MB in addition to the
    // source and destination Mats. Recompute HSV in pass two instead.
    double sumSat = 0.0;
    for (int i = 0; i < px; ++i) {
        float hh, ss, vv;
        rgbToPillowHsv(a[i * 3], a[i * 3 + 1], a[i * 3 + 2], hh, ss, vv);
        sumSat += ss / 255.0;
    }

    float meanSat = static_cast<float>(sumSat / px);
    float maximum = 0.062f; // NATURAL
    float boost = 0.0f;

    if (meanSat < 0.28f) boost = std::min(maximum, (0.28f - meanSat) * 0.88f);
    else if (meanSat < 0.42f) boost = std::min(maximum, (0.42f - meanSat) * 0.44f);
    else boost = 0.010f;

    if (boost <= 0.0f) return img;

    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();
    const float* m = mask.ptr<float>();

    for (int i = 0; i < px; ++i) {
        float h, s, v;
        rgbToPillowHsv(a[i * 3], a[i * 3 + 1], a[i * 3 + 2], h, s, v);
        float protection = clampf((v / 255.0f - 0.68f) / 0.32f, 0.0f, 1.0f);
        float effective = boost * (1.0f - protection * 0.60f);
        effective *= (1.0f - (s / 255.0f) * 0.65f);  // vibrance
        bool skinHue = (h <= 28.0f || h >= 240.0f);
        if (skinHue && s >= 30.0f && s <= 200.0f && v >= 60.0f && v <= 240.0f)
            effective *= 0.45f;  // skin guard
        effective *= (1.0f - m[i]);  // dark-flat gate

        float newS = clampf(s * (1.0f + effective), 0.0f, 255.0f);

        float r, g, b;
        pillowHsvToRgb(h, newS, v, r, g, b);
        o[i * 3]     = clampf(r, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(g, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(b, 0.0f, 255.0f);
    }

    return out;
}

cv::Mat tonalDepth(const cv::Mat &img, const Eval &ev) {
    int px = img.rows * img.cols;
    const float* a = img.ptr<float>();

    float shadow = 0.015f;   // NATURAL
    float highlight = 0.010f;
    float blackPull = 0.018f * clampf(ev.brightness / DEEPBLACK_REF, DEEPBLACK_FLOOR, 1.0f);

    if (ev.shadowRatio > 0.18f) {
        shadow *= std::min(1.0f, ev.shadowRatio / 0.32f);
    } else {
        shadow *= 0.40f;
    }

    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();

    for (int i = 0; i < px; ++i) {
        float r = a[i * 3] / 255.0f;
        float g = a[i * 3 + 1] / 255.0f;
        float b = a[i * 3 + 2] / 255.0f;
        float lum = r * 0.299f + g * 0.587f + b * 0.114f;

        float shadowMask = clampf((0.50f - lum) / 0.50f, 0.0f, 1.0f);
        float highlightMask = clampf((lum - 0.74f) / 0.26f, 0.0f, 1.0f);
        float deepBlack = clampf((0.22f - lum) / 0.22f, 0.0f, 1.0f);

        shadowMask *= clampf((lum - 0.10f) / 0.20f, 0.0f, 1.0f);

        float rr = r - deepBlack * blackPull;
        float gg = g - deepBlack * blackPull;
        float bb = b - deepBlack * blackPull;

        rr += shadowMask * (shadow * 0.25f);
        gg += shadowMask * (shadow * 0.25f);
        bb += shadowMask * (shadow * 0.25f);

        rr -= highlightMask * highlight;
        gg -= highlightMask * highlight;
        bb -= highlightMask * highlight;

        o[i * 3] = clampf(rr * 255.0f, 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(gg * 255.0f, 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(bb * 255.0f, 0.0f, 255.0f);
    }
    return out;
}

cv::Mat unsharpMask(const cv::Mat &img, float radius, int percent, int threshold,
                    const cv::Mat &mask) {
    cv::Mat blurred = gaussianBlurCustom(img, radius);
    int px = img.rows * img.cols;
    const float* srcPx = img.ptr<float>();
    const float* blurPx = blurred.ptr<float>();
    const float* m = mask.ptr<float>();
    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();
    float amount = percent / 100.0f;
    for (int i = 0; i < px; ++i) {
        float am = amount * (1.0f - m[i]);
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            float diff = srcPx[idx] - blurPx[idx];
            if (std::abs(diff) >= threshold) {
                o[idx] = clampf(srcPx[idx] + diff * am, 0.0f, 255.0f);
            } else {
                o[idx] = srcPx[idx];
            }
        }
    }
    return out;
}

cv::Mat crystalSharpen(const cv::Mat &img, const Eval &ev, const cv::Mat &mask) {
    float s = ev.sharpness;
    float n = ev.noise;

    int minPercent = 45;
    int maxPercent = 65;

    int percent;
    if (s < 18.0f) percent = maxPercent;
    else if (s < 28.0f) percent = static_cast<int>(maxPercent * 0.92f);
    else if (s < 38.0f) percent = static_cast<int>(minPercent + (maxPercent - minPercent) * 0.72f);
    else if (s < 50.0f) percent = static_cast<int>(minPercent + (maxPercent - minPercent) * 0.48f);
    else if (s < 65.0f) percent = static_cast<int>(minPercent + (maxPercent - minPercent) * 0.22f);
    else percent = minPercent;

    if (n > 60.0f) percent = static_cast<int>(percent * 0.80f);

    float radius = ev.isGraphic ? 1.8f : 1.0f;
    int threshold = ev.isGraphic ? 4 : 10;
    if (ev.isGraphic) percent = static_cast<int>(percent * 0.90f);

    percent = clampi(percent, 40, maxPercent);

    return unsharpMask(img, radius, percent, threshold, mask);
}

cv::Mat fineCrisp(const cv::Mat &img, const Eval &ev, const cv::Mat &mask) {
    int percent = 8; // NATURAL
    if (ev.noise > 55.0f) percent = static_cast<int>(percent * 0.60f);
    return unsharpMask(img, 0.5f, percent, 8, mask);
}

cv::Mat highlightSafety(const cv::Mat &img, const cv::Mat &original) {
    int px = img.rows * img.cols;
    const float* arr = img.ptr<float>();
    const float* orig = original.ptr<float>();

    float protection = 0.30f; // NATURAL
    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();

    for (int i = 0; i < px; ++i) {
        float r = arr[i * 3];
        float g = arr[i * 3 + 1];
        float b = arr[i * 3 + 2];
        float lum = r * 0.299f + g * 0.587f + b * 0.114f;

        float mask = clampf((lum - 248.0f) / 7.0f, 0.0f, 1.0f);
        float t = mask * protection;

        o[i * 3] = r * (1.0f - t) + orig[i * 3] * t;
        o[i * 3 + 1] = g * (1.0f - t) + orig[i * 3 + 1] * t;
        o[i * 3 + 2] = b * (1.0f - t) + orig[i * 3 + 2] * t;
    }
    return out;
}

// --- User curve pre-stage (points unchanged since v2.1) ---
// Control points (0,0) (59,44) (129,133) (208,208) (255,255), PCHIP LUT,
// non-decreasing, same LUT on R, G, B (no hue shift).

static const uint8_t CURVE_LUT[256] = {
    0, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9, 9, 10, 11,
    11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 20, 20, 21, 22,
    22, 23, 24, 25, 25, 26, 27, 27, 28, 29, 30, 30, 31, 32, 33, 34,
    34, 35, 36, 37, 38, 39, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 62, 63, 64, 65, 67,
    68, 69, 71, 72, 73, 75, 76, 77, 79, 80, 82, 83, 84, 86, 87, 89,
    90, 91, 93, 94, 96, 97, 98, 100, 101, 103, 104, 105, 107, 108, 109, 111,
    112, 113, 115, 116, 117, 119, 120, 121, 122, 124, 125, 126, 127, 128, 130, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 148,
    149, 150, 151, 152, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
    164, 165, 166, 167, 168, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
    178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
};

// out = in + strength * (LUT[in] - in); strength 1.0 = the full curve.
cv::Mat applyCurve(const cv::Mat &img, float strength) {
    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    float* o = out.ptr<float>();
    int px = img.rows * img.cols * 3;
    for (int i = 0; i < px; ++i) {
        int idx = static_cast<int>(a[i] + 0.5f);
        idx = std::max(0, std::min(255, idx));
        o[i] = a[i] + strength * (static_cast<float>(CURVE_LUT[idx]) - a[i]);
    }
    return out;
}

#ifndef ENHANCER_NO_JNI
// --- Bitmap Conversion Utilities ---

// JNI bitmap access is deliberately restricted to RGBA_8888. Android's
// stride is bytes-per-row and may exceed width*4; never flatten rows as if
// they were tightly packed. Conversion uses byte order, not uint32 shifts.
struct AndroidBitmapPixels {
    JNIEnv *env = nullptr;
    jobject bitmap = nullptr;
    void *pixels = nullptr;
    bool locked = false;

    AndroidBitmapPixels(JNIEnv *e, jobject b) : env(e), bitmap(b) {}
    bool lock() {
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS || !pixels)
            return false;
        locked = true;
        return true;
    }
    ~AndroidBitmapPixels() {
        if (locked) AndroidBitmap_unlockPixels(env, bitmap);
    }
};

static void throwBitmapError(JNIEnv *env, const char *className, const char *message) {
    if (!env || env->ExceptionCheck()) return;
    jclass cls = env->FindClass(className);
    if (cls) {
        env->ThrowNew(cls, message);
        env->DeleteLocalRef(cls);
    }
}

bool bitmapToFloatRgb(JNIEnv *env, jobject bitmap, cv::Mat &matRgbFloat) {
    if (!env || !bitmap) {
        throwBitmapError(env, "java/lang/IllegalArgumentException", "Source bitmap is null");
        return false;
    }

    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throwBitmapError(env, "java/lang/IllegalArgumentException", "Cannot read source bitmap info");
        return false;
    }
    if (info.width == 0 || info.height == 0 ||
        info.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        info.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        info.width > std::numeric_limits<uint32_t>::max() / 4u ||
        info.stride < info.width * 4u || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwBitmapError(env, "java/lang/IllegalArgumentException",
                         "Source bitmap must be valid RGBA_8888 with a supported stride");
        return false;
    }

    // Allocate before locking so an allocation failure cannot leave pixels locked.
    matRgbFloat.create(static_cast<int>(info.height), static_cast<int>(info.width), CV_32FC3);
    AndroidBitmapPixels lock(env, bitmap);
    if (!lock.lock()) {
        throwBitmapError(env, "java/lang/IllegalStateException", "Cannot lock source bitmap pixels");
        return false;
    }

    const auto *base = static_cast<const uint8_t *>(lock.pixels);
    for (int y = 0; y < matRgbFloat.rows; ++y) {
        const uint8_t *src = base + static_cast<size_t>(y) * info.stride;
        float *dst = matRgbFloat.ptr<float>(y);
        for (int x = 0; x < matRgbFloat.cols; ++x) {
            const uint8_t *px = src + static_cast<size_t>(x) * 4u;
            dst[x * 3]     = static_cast<float>(px[0]); // R
            dst[x * 3 + 1] = static_cast<float>(px[1]); // G
            dst[x * 3 + 2] = static_cast<float>(px[2]); // B
        }
    }
    return true;
}

bool floatRgbToBitmap(JNIEnv *env, const cv::Mat &matRgbFloat, jobject bitmap) {
    if (!env || !bitmap || matRgbFloat.empty() || matRgbFloat.type() != CV_32FC3) {
        throwBitmapError(env, "java/lang/IllegalArgumentException", "Invalid destination bitmap or RGB image");
        return false;
    }

    AndroidBitmapInfo info{};
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
        throwBitmapError(env, "java/lang/IllegalArgumentException", "Cannot read destination bitmap info");
        return false;
    }
    if (info.width != static_cast<uint32_t>(matRgbFloat.cols) ||
        info.height != static_cast<uint32_t>(matRgbFloat.rows) ||
        info.width == 0 || info.height == 0 ||
        info.width > std::numeric_limits<uint32_t>::max() / 4u ||
        info.stride < info.width * 4u || info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        throwBitmapError(env, "java/lang/IllegalArgumentException",
                         "Destination bitmap must match image dimensions and be RGBA_8888");
        return false;
    }

    AndroidBitmapPixels lock(env, bitmap);
    if (!lock.lock()) {
        throwBitmapError(env, "java/lang/IllegalStateException", "Cannot lock destination bitmap pixels");
        return false;
    }

    auto *base = static_cast<uint8_t *>(lock.pixels);
    for (int y = 0; y < matRgbFloat.rows; ++y) {
        uint8_t *dst = base + static_cast<size_t>(y) * info.stride;
        const float *src = matRgbFloat.ptr<float>(y);
        for (int x = 0; x < matRgbFloat.cols; ++x) {
            uint8_t *px = dst + static_cast<size_t>(x) * 4u;
            px[0] = static_cast<uint8_t>(clampi(static_cast<int>(src[x * 3] + 0.5f), 0, 255));
            px[1] = static_cast<uint8_t>(clampi(static_cast<int>(src[x * 3 + 1] + 0.5f), 0, 255));
            px[2] = static_cast<uint8_t>(clampi(static_cast<int>(src[x * 3 + 2] + 0.5f), 0, 255));
            px[3] = 255;
        }
    }
    return true;
}

// --- Main JNI Function ---

extern "C" JNIEXPORT void JNICALL
Java_com_master_aistudio_utils_NativeSmartEnhancer_nativeEnhance(
        JNIEnv *env,
        jobject thiz,
        jobject srcBitmap,
        jobject dstBitmap) {
    (void)thiz;

    try {
    cv::Mat work;
    if (!bitmapToFloatRgb(env, srcBitmap, work)) return;

    // Keep the immutable original analysis reference. The user curve is a
    // processing transform and must not redefine the measured input metrics.
    cv::Mat analysisSrc = work;
    float preLum = quickMeanLum(analysisSrc);

    // 0. Adaptive user curve, applied only to the processing image.
    work = applyCurve(work, curveStrengthFor(preLum));

    // 1. Global evaluation on a bounded preview of the ORIGINAL image.
    cv::Mat evalSrc = analysisSrc;
    int maxEdge = std::max(work.cols, work.rows);
    if (maxEdge > 1280) {
        float scale = 1280.0f / maxEdge;
        int nw = std::max(1, static_cast<int>(work.cols * scale));
        int nh = std::max(1, static_cast<int>(work.rows * scale));
        cv::resize(analysisSrc, evalSrc, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    }
    Eval ev = evaluate(evalSrc);
    evalSrc.release();
    analysisSrc.release(); // no longer needed; work is the independent curve result

    // 2. Pipeline (v4.1 order).
    cv::Mat out = whiteBalance(work, ev);
    cv::Mat ref = out;                                // v4.1: colour reference for chromaRestore
    // v4.3 (F2): adaptive sigma + chroma gate.
    SigmaInfo si = estimateSigmaAdaptive(out);
    float sigma = si.sigma;
    float fineGate = 0.0f;
    NoiseEstimate noiseEstimate;
    noiseEstimate.sigmaLuma = sigma;
    noiseEstimate.correlatedLuma = si.corr;
    noiseEstimate.lumaSupportFraction = si.lowFrac;
    // Do not misrepresent the accepted-block fraction as confidence.
    // Confidence remains unknown until calibrated on synthetic + real data.
    noiseEstimate.confidenceLuma = 0.0f;
    {
        cv::Mat c1, c2;
        opponentChromaChannels(out, c1, c2);
        noiseEstimate.sigmaChroma1 = immerkaerChan(c1);
        noiseEstimate.sigmaChroma2 = immerkaerChan(c2);
        noiseEstimate.confidenceChroma = 0.0f; // no calibrated confidence model yet
        float chromaSigma = std::max(noiseEstimate.sigmaChroma1,
                                     noiseEstimate.sigmaChroma2);
        fineGate = smoothstep01(chromaSigma, CHROMA_GATE_LO, CHROMA_GATE_HI);
        c1.release();
        c2.release();
    }
    float noiseGate = smoothstep01(sigma, CHROMA_GATE_LO, CHROMA_GATE_HI);  // v4.1
    if (!ev.isGraphic) {
        out = nlmDenoise(out, sigma);                 // v4: luma NLM + fine chroma
        if (sigma < NLM_MIN_SIGMA && fineGate > 0.001f)
            out = chromaFineClean(out, fineGate);     // v4.3 (F2): chroma-only path
        if (si.corr >= 1.5f) out = corrLumaClean(out, si.corr);  // v4.3 (F2)
        out = chromaBlotchClean(out, noiseGate);      // v4.1: gated by measured noise
    }
    cv::Mat darkMask = buildDarkFlatMask(out);        // unchanged
    // v4.2: gate open -> full v4.1 path (with chroma suppression);
    // gate closed -> map-modulated luma-only path (chroma exact).
    cv::Mat noiseScore = buildNoiseScoreMap(lumaOf(out), ALIGN_FLOOR_K * sigma);
    if (noiseGate >= 1.0f) {
        out = darkFlatClean(out, darkMask, noiseGate);
    } else {
        cv::Mat periodMap = buildPeriodMap(lumaOf(out));
        out = darkFlatCleanMap(out, darkMask, noiseScore, periodMap);
        periodMap.release();
    }
    noiseScore.release();
    out = localShadowLift(out);                       // v4.1: multiplicative lift
    SigmaInfo si2 = estimateSigmaAdaptive(out);       // v4.3 (F2): residual adaptive sigma
    float sigma2 = si2.sigma;
    // v4.3 (F3): sharpening gated by the residual noise map.
    cv::Mat score2 = buildNoiseScoreMap(lumaOf(out), ALIGN_FLOOR_K * sigma2);
    cv::Mat sharpMask = buildSharpMaskScore(out, sigma2, darkMask, score2);
    out = clarityBoost(out, ev, sharpMask);
    out = microDetail(out, ev, sharpMask);
    out = adaptiveContrastS5(out, ev, noiseGate, score2);  // v4.4 (S5)
    out = colorDepth(out, ev, darkMask);
    darkMask.release();
    score2.release();
    out = tonalDepth(out, ev);
    out = crystalSharpen(out, ev, sharpMask);
    out = fineCrisp(out, ev, sharpMask);
    if (!ev.isGraphic) out = chromaRestore(out, ref); // v4.1: saturation anchor
    ref.release();
    out = highlightSafety(out, work);

    // 3. Render back to Bitmap.
    (void)floatRgbToBitmap(env, out, dstBitmap);
    } catch (const cv::Exception &e) {
        throwBitmapError(env, "java/lang/RuntimeException", e.what());
    } catch (const std::bad_alloc &) {
        throwBitmapError(env, "java/lang/OutOfMemoryError", "Native image processing ran out of memory");
    } catch (const std::exception &e) {
        throwBitmapError(env, "java/lang/RuntimeException", e.what());
    } catch (...) {
        throwBitmapError(env, "java/lang/RuntimeException", "Unknown native image processing error");
    }
}
#endif // ENHANCER_NO_JNI
