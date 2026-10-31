#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>

// --- Helper Functions ---

inline float clampf(float v, float minVal, float maxVal) {
    return std::max(minVal, std::min(maxVal, v));
}

inline int clampi(int v, int minVal, int maxVal) {
    return std::max(minVal, std::min(maxVal, v));
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
    int px = rgbFloat.rows * rgbFloat.cols;
    cv::Mat grayFloat(rgbFloat.rows, rgbFloat.cols, CV_32FC1);
    
    const float* src = rgbFloat.ptr<float>();
    float* dst = grayFloat.ptr<float>();

    for (int i = 0; i < px; ++i) {
        float r = src[i * 3];
        float g = src[i * 3 + 1];
        float b = src[i * 3 + 2];
        dst[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    return convolve3x3FindEdges(grayFloat);
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

Eval evaluate(const cv::Mat &rgbFloat) {
    int w = rgbFloat.cols;
    int h = rgbFloat.rows;
    int n = w * h;

    double sumGray = 0.0;
    double sumGraySq = 0.0;
    int dark = 0, shadow = 0, bright = 0, highlight = 0;

    std::vector<float> gray(n);
    const float* ptr = rgbFloat.ptr<float>();

    for (int i = 0; i < n; ++i) {
        float r = ptr[i * 3];
        float g = ptr[i * 3 + 1];
        float b = ptr[i * 3 + 2];
        float y = 0.2989f * r + 0.5870f * g + 0.1140f * b;
        gray[i] = y;
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
    std::vector<float> residual(n);
    for (int i = 0; i < n; ++i) {
        float dr = std::abs(ptr[i * 3] - b10Ptr[i * 3]);
        float dg = std::abs(ptr[i * 3 + 1] - b10Ptr[i * 3 + 1]);
        float db = std::abs(ptr[i * 3 + 2] - b10Ptr[i * 3 + 2]);
        residual[i] = (dr + dg + db) / 3.0f;
    }
    std::sort(residual.begin(), residual.end());
    int p75Idx = clampi(static_cast<int>(n * 0.75f), 0, n - 1);
    float noise = clampf(residual[p75Idx] * 3.0f, 0.0f, 100.0f);

    // Graphic check
    std::vector<float> rCh(n), gCh(n), bCh(n);
    for (int i = 0; i < n; ++i) {
        rCh[i] = ptr[i * 3];
        gCh[i] = ptr[i * 3 + 1];
        bCh[i] = ptr[i * 3 + 2];
    }
    cv::Mat rMat(h, w, CV_32FC1, rCh.data());
    cv::Mat gMat(h, w, CV_32FC1, gCh.data());
    cv::Mat bMat(h, w, CV_32FC1, bCh.data());

    cv::Mat er = convolve3x3FindEdges(rMat);
    cv::Mat eg = convolve3x3FindEdges(gMat);
    cv::Mat eb = convolve3x3FindEdges(bMat);

    float edgeRatio = 0.0f;
    float strongEdgeRatio = 0.0f;
    const float* erP = er.ptr<float>();
    const float* egP = eg.ptr<float>();
    const float* ebP = eb.ptr<float>();

    for (int i = 0; i < n; ++i) {
        float v = 0.299f * erP[i] + 0.587f * egP[i] + 0.114f * ebP[i];
        if (v > 80.0f) edgeRatio += 1.0f;
        if (v > 150.0f) strongEdgeRatio += 1.0f;
    }
    edgeRatio /= n;
    strongEdgeRatio /= n;
    bool isGraphic = (edgeRatio > 0.065f && strongEdgeRatio > 0.014f);

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

cv::Mat gentleDenoise(const cv::Mat &img, const Eval &ev) {
    if (ev.noise < 38.0f) return img;
    float strength = (ev.noise < 55.0f) ? 0.08f : ((ev.noise < 72.0f) ? 0.14f : 0.20f);
    
    cv::Mat blurred = gaussianBlurCustom(img, 0.75f);
    cv::Mat out(img.size(), CV_32FC3);

    const float* a = img.ptr<float>();
    const float* b = blurred.ptr<float>();
    float* o = out.ptr<float>();
    int px = img.rows * img.cols * 3;

    float u = 1.0f - strength;
    for (int i = 0; i < px; ++i) {
        o[i] = clampf(a[i] * u + b[i] * strength, 0.0f, 255.0f);
    }
    return out;
}

inline float smoothstep01(float x, float a, float b) {
    float t = clampf((x - a) / (b - a), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// v2: conservative global white balance (gray-world on a small copy).
// Skips itself when: the image is a graphic (intentional colors),
// the image is near-black (no reliable cast), or no cast is measurable.
// Correction is capped and partial on purpose: a full neutralization
// would destroy intentional warmth (sunsets, tungsten mood).
cv::Mat whiteBalance(const cv::Mat &img, const Eval &ev) {
    if (ev.isGraphic) return img;

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
    double sr = 0.0, sg = 0.0, sb = 0.0;
    for (int i = 0; i < n; ++i) {
        sr += p[i * 3];
        sg += p[i * 3 + 1];
        sb += p[i * 3 + 2];
    }
    float mr = static_cast<float>(sr / n);
    float mg = static_cast<float>(sg / n);
    float mb = static_cast<float>(sb / n);
    float mean = (mr + mg + mb) / 3.0f;
    if (mean < 5.0f) return img;

    float dev = std::max({std::abs(mr - mean), std::abs(mg - mean), std::abs(mb - mean)});
    if (dev < 8.0f) return img;  // no measurable cast: do not touch

    float gr = clampf(mean / std::max(1.0f, mr), 0.80f, 1.25f);
    float gg = clampf(mean / std::max(1.0f, mg), 0.80f, 1.25f);
    float gb = clampf(mean / std::max(1.0f, mb), 0.80f, 1.25f);
    const float k = 0.70f;  // partial correction

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

// v2: dark-AND-flat mask, single-channel float in [0,1].
// Dark  = luminance weight, full below 45, gone by 95 (the code's own
//         dark<45 / shadow<80 vocabulary, made soft).
// Flat  = local standard deviation of the LIGHTLY SMOOTHED luminance.
//         The smoothing matters: raw grain would raise the local std and
//         disqualify the very sky we want to treat. Measuring flatness
//         after a radius-1.0 blur removes grain but keeps real structure
//         (hair, edges, textures) visible to the std test.
// Built at half resolution (the mask is smooth by construction) and
// upsampled with bilinear interpolation, then feathered.
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

// v2: treatment applied inside the mask only (see JNI section).
// One blur-blend does two jobs: it denoises AND it applies negative
// clarity (pulling pixels toward their local mean removes micro-contrast,
// and grain IS micro-contrast). Chroma suppression then removes the
// colored speckles and restores solid blacks. No extra black pull here:
// tonalDepth's deepBlack already pulls deep blacks globally.
cv::Mat darkFlatClean(const cv::Mat &img, const cv::Mat &mask) {
    cv::Mat blurred = gaussianBlurCustom(img, 2.0f);
    cv::Mat out(img.size(), CV_32FC3);
    const float* a = img.ptr<float>();
    const float* b = blurred.ptr<float>();
    const float* m = mask.ptr<float>();
    float* o = out.ptr<float>();
    int n = img.rows * img.cols;
    for (int i = 0; i < n; ++i) {
        float s = m[i] * 0.55f;  // cleaning strength knob
        float r  = a[i * 3]     * (1.0f - s) + b[i * 3]     * s;
        float g  = a[i * 3 + 1] * (1.0f - s) + b[i * 3 + 1] * s;
        float bl = a[i * 3 + 2] * (1.0f - s) + b[i * 3 + 2] * s;
        float lum = 0.299f * r + 0.587f * g + 0.114f * bl;
        float d = m[i] * 0.45f;  // chroma suppression knob
        o[i * 3]     = clampf(lum + (r - lum)  * (1.0f - d), 0.0f, 255.0f);
        o[i * 3 + 1] = clampf(lum + (g - lum)  * (1.0f - d), 0.0f, 255.0f);
        o[i * 3 + 2] = clampf(lum + (bl - lum) * (1.0f - d), 0.0f, 255.0f);
    }
    return out;
}

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

cv::Mat microDetail(const cv::Mat &img, const Eval &ev) {
    cv::Mat blurred = gaussianBlurCustom(img, 1.0f);
    cv::Mat grayEdges = findEdgesFromL(img);
    
    int n = img.rows * img.cols;
    std::vector<float> edgeMask(n);
    const float* geP = grayEdges.ptr<float>();
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
        float m = edgeMask[i];
        for (int c = 0; c < 3; ++c) {
            int idx = i * 3 + c;
            float detail = clampf(a[idx] - b[idx], -18.0f, 18.0f);
            o[idx] = clampf(a[idx] + detail * strength * m, 0.0f, 255.0f);
        }
    }
    return out;
}

cv::Mat pilContrast(const cv::Mat &img, float factor) {
    int px = img.rows * img.cols;
    const float* a = img.ptr<float>();
    
    double sum = 0.0;
    for (int i = 0; i < px; ++i) {
        sum += 0.299 * a[i * 3] + 0.587 * a[i * 3 + 1] + 0.114 * a[i * 3 + 2];
    }
    float mean = clampf(static_cast<float>(sum / px), 0.0f, 255.0f);

    cv::Mat out(img.size(), CV_32FC3);
    float* o = out.ptr<float>();
    for (int i = 0; i < px * 3; ++i) {
        o[i] = clampf(mean * (1.0f - factor) + a[i] * factor, 0.0f, 255.0f);
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
    return pilContrast(img, factor);
}

cv::Mat colorDepth(const cv::Mat &img, const Eval &ev, const cv::Mat &mask) {
    int px = img.rows * img.cols;
    const float* a = img.ptr<float>();

    std::vector<std::tuple<float, float, float>> hsv(px);
    double sumSat = 0.0;

    for (int i = 0; i < px; ++i) {
        float hh, ss, vv;
        rgbToPillowHsv(a[i * 3], a[i * 3 + 1], a[i * 3 + 2], hh, ss, vv);
        hsv[i] = {hh, ss, vv};
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
        auto [h, s, v] = hsv[i];
        float protection = clampf((v / 255.0f - 0.68f) / 0.32f, 0.0f, 1.0f);
        float effective = boost * (1.0f - protection * 0.60f);
        effective *= (1.0f - (s / 255.0f) * 0.65f);  // (a) vibrance
        bool skinHue = (h <= 28.0f || h >= 240.0f);
        if (skinHue && s >= 30.0f && s <= 200.0f && v >= 60.0f && v <= 240.0f)
            effective *= 0.45f;  // (b) skin guard
        effective *= (1.0f - m[i]);  // (c) mask gate

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
    float blackPull = 0.018f;

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

// --- Bitmap Conversion Utilities ---

void bitmapToFloatRgb(JNIEnv *env, jobject bitmap, cv::Mat &matRgbFloat) {
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    matRgbFloat.create(info.height, info.width, CV_32FC3);
    float* dst = matRgbFloat.ptr<float>();
    const uint32_t* src = static_cast<const uint32_t*>(pixels);

    int total = info.height * info.width;
    for (int i = 0; i < total; ++i) {
        uint32_t p = src[i];
        dst[i * 3]     = static_cast<float>((p >> 16) & 0xFF); // R
        dst[i * 3 + 1] = static_cast<float>((p >> 8)  & 0xFF); // G
        dst[i * 3 + 2] = static_cast<float>(p         & 0xFF); // B
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

void floatRgbToBitmap(JNIEnv *env, const cv::Mat &matRgbFloat, jobject bitmap) {
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);

    uint32_t* dst = static_cast<uint32_t*>(pixels);
    const float* src = matRgbFloat.ptr<float>();

    int total = info.height * info.width;
    for (int i = 0; i < total; ++i) {
        int r = clampi(static_cast<int>(src[i * 3]), 0, 255);
        int g = clampi(static_cast<int>(src[i * 3 + 1]), 0, 255);
        int b = clampi(static_cast<int>(src[i * 3 + 2]), 0, 255);
        dst[i] = (0xFF << 24) | (r << 16) | (g << 8) | b;
    }

    AndroidBitmap_unlockPixels(env, bitmap);
}

// --- Main JNI Function ---

extern "C" JNIEXPORT void JNICALL
Java_com_master_aistudio_utils_NativeSmartEnhancer_nativeEnhance(
        JNIEnv *env,
        jobject thiz,
        jobject srcBitmap,
        jobject dstBitmap) {

    cv::Mat work;
    bitmapToFloatRgb(env, srcBitmap, work);

    // 1. Evaluate on 1280 max edge preview if larger
    cv::Mat evalSrc = work;
    int maxEdge = std::max(work.cols, work.rows);
    if (maxEdge > 1280) {
        float scale = 1280.0f / maxEdge;
        int nw = std::max(1, static_cast<int>(work.cols * scale));
        int nh = std::max(1, static_cast<int>(work.rows * scale));
        cv::resize(work, evalSrc, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
    }

    Eval ev = evaluate(evalSrc);

    // 2. Pipeline Execution (v2 order)
    cv::Mat out = gentleDenoise(work, ev);
    out = whiteBalance(out, ev);                 // v2 NEW
    cv::Mat darkMask = buildDarkFlatMask(out);   // v2 NEW
    out = darkFlatClean(out, darkMask);          // v2 NEW
    out = clarityBoost(out, ev, darkMask);       // v2 signature
    out = microDetail(out, ev);                  // unchanged
    out = adaptiveContrast(out, ev);             // unchanged
    out = colorDepth(out, ev, darkMask);         // v2 signature
    out = tonalDepth(out, ev);                   // unchanged
    out = crystalSharpen(out, ev, darkMask);     // v2 signature
    out = fineCrisp(out, ev, darkMask);          // v2 signature
    out = highlightSafety(out, work);            // unchanged

    // 3. Render back to Bitmap
    floatRgbToBitmap(env, out, dstBitmap);
}
