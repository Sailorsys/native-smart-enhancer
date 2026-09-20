#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>

void bitmapToMat(JNIEnv *env, jobject bitmap, cv::Mat &mat) {
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    mat.create(info.height, info.width, CV_8UC4);
    memcpy(mat.data, pixels, info.height * info.width * 4);
    AndroidBitmap_unlockPixels(env, bitmap);
}

void matToBitmap(JNIEnv *env, const cv::Mat &mat, jobject bitmap) {
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    memcpy(pixels, mat.data, info.height * info.width * 4);
    AndroidBitmap_unlockPixels(env, bitmap);
}

extern "C" JNIEXPORT void JNICALL
Java_com_master_aistudio_utils_NativeSmartEnhancer_nativeEnhance(
        JNIEnv *env,
        jobject thiz,
        jobject srcBitmap,
        jobject dstBitmap) {

    cv::Mat srcRGBA;
    bitmapToMat(env, srcBitmap, srcRGBA);

    // 1. تحويل الصورة إلى RGB
    cv::Mat srcRGB;
    cv::cvtColor(srcRGBA, srcRGB, cv::COLOR_RGBA2RGB);

    // 2. التحويل إلى فضاء الألوان LAB (لفصل الإضاءة L عن قنوات الألوان a, b)
    cv::Mat labImg;
    cv::cvtColor(srcRGB, labImg, cv::COLOR_RGB2Lab);

    std::vector<cv::Mat> labChannels;
    cv::split(labImg, labChannels); // labChannels[0] = L, [1] = a, [2] = b

    // --- أ. تطبيق التباين التكيفي المتقدم (CLAHE) على قناة الإضاءة فقط ---
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.2, cv::Size(8, 8));
    cv::Mat lClahe;
    clahe->apply(labChannels[0], lClahe);

    // --- ب. رفع حدة التفاصيل الدقيقة (Micro-Crisp & Detail Boost) ---
    cv::Mat lFloat, lBlur, lDetail;
    lClahe.convertTo(lFloat, CV_32FC1);

    // تفاصيل دقيقة جداً (نصف قطر صغير)
    cv::GaussianBlur(lFloat, lBlur, cv::Size(0, 0), 1.2);
    cv::subtract(lFloat, lBlur, lDetail);
    cv::Mat lEnhancedFloat = lFloat + (lDetail * 0.75f); // تعزيز التفاصيل 75%

    // بروتوكول Clarity للمناطق المتوسطة (نصف قطر أوسع)
    cv::Mat lBlurMacro, lMacroDetail;
    cv::GaussianBlur(lEnhancedFloat, lBlurMacro, cv::Size(0, 0), 4.5);
    cv::subtract(lEnhancedFloat, lBlurMacro, lMacroDetail);
    lEnhancedFloat = lEnhancedFloat + (lMacroDetail * 0.25f);

    // ضبط الحدود لمنع حرق الإضاءة أو طمس الظلال
    cv::threshold(lEnhancedFloat, lEnhancedFloat, 255.0, 255.0, cv::THRESH_TRUNC);
    cv::max(lEnhancedFloat, 0.0f, lEnhancedFloat);
    lEnhancedFloat.convertTo(labChannels[0], CV_8UC1);

    // إعادة دمج قنوات LAB وتحويلها إلى RGB
    cv::Mat enhancedLab, enhancedRGB;
    cv::merge(labChannels, enhancedLab);
    cv::cvtColor(enhancedLab, enhancedRGB, cv::COLOR_Lab2RGB);

    // --- ج. تعزيز تشبع وحيوية الألوان (Vibrance Boost في HSV) ---
    cv::Mat hsvImg;
    cv::cvtColor(enhancedRGB, hsvImg, cv::COLOR_RGB2HSV);

    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImg, hsvChannels);

    // زيادة تشبع الألوان (قناة S) بنسبة 15% تكيفياً
    cv::Mat sFloat;
    hsvChannels[1].convertTo(sFloat, CV_32FC1);
    sFloat = sFloat * 1.15f;
    cv::threshold(sFloat, sFloat, 255.0, 255.0, cv::THRESH_TRUNC);
    sFloat.convertTo(hsvChannels[1], CV_8UC1);

    cv::merge(hsvChannels, hsvImg);
    cv::cvtColor(hsvImg, enhancedRGB, cv::COLOR_HSV2RGB);

    // 3. تحويل النتيجة النهائية إلى RGBA وتصديرها للـ Bitmap
    cv::Mat dstRGBA;
    cv::cvtColor(enhancedRGB, dstRGBA, cv::COLOR_RGB2RGBA);

    matToBitmap(env, dstRGBA, dstBitmap);
}
