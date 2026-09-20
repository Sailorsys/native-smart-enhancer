#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>

// تحويل Android Bitmap إلى cv::Mat (RGBA)
void bitmapToMat(JNIEnv *env, jobject bitmap, cv::Mat &mat) {
    AndroidBitmapInfo info;
    void *pixels;
    AndroidBitmap_getInfo(env, bitmap, &info);
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    mat.create(info.height, info.width, CV_8UC4);
    memcpy(mat.data, pixels, info.height * info.width * 4);
    AndroidBitmap_unlockPixels(env, bitmap);
}

// نقل cv::Mat المكتملة إلى Android Bitmap
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

    // 1. تحويل الصورة إلى RGB ثم إلى Float32 للمعالجة عالية الدقة
    cv::Mat srcRGB, floatImg;
    cv::cvtColor(srcRGBA, srcRGB, cv::COLOR_RGBA2RGB);
    srcRGB.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

    // 2. حساب مستوى التباين والسطوع لضبط القوة تلقائياً (Evaluation)
    cv::Mat gray;
    cv::cvtColor(srcRGBA, gray, cv::COLOR_RGBA2GRAY);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    float contrast = stddev[0]; // التباين الحالي

    // 3. مرحلة Clarity Boost (إبراز التفاصيل بناءً على التباين)
    float clarityStrength = (contrast > 60.0f) ? 0.04f : 0.065f;
    cv::Mat blurredClarity, detail;
    cv::GaussianBlur(floatImg, blurredClarity, cv::Size(0, 0), 3.0);
    cv::subtract(floatImg, blurredClarity, detail);
    cv::scaleAdd(detail, clarityStrength, floatImg, floatImg);

    // 4. مرحلة Crystal Sharpen & Micro Detail (رفع الحدة التكيفية)
    cv::Mat blurredSharpen, sharp;
    cv::GaussianBlur(floatImg, blurredSharpen, cv::Size(0, 0), 1.0);
    cv::addWeighted(floatImg, 1.35, blurredSharpen, -0.35, 0, floatImg);

    // 5. ضبط التباين والعمق (Tonal Depth & Adaptive Contrast)
    // تطبيق ضغط التباين المطاطي لحماية المناطق المضيئة والغامقة
    cv::Mat dstRGB;
    floatImg = (floatImg - 0.5f) * 1.08f + 0.5f; // زيادة التباين بنسبة 8%
    cv::threshold(floatImg, floatImg, 1.0, 1.0, cv::THRESH_TRUNC); // منع حرق الألوان الفاتحة
    cv::max(floatImg, 0.0f, floatImg);                           // منع طمس الظلال

    // 6. التحويل العكسي لـ UINT8 و RGBA
    floatImg.convertTo(dstRGB, CV_8UC3, 255.0);
    cv::Mat dstRGBA;
    cv::cvtColor(dstRGB, dstRGBA, cv::COLOR_RGB2RGBA);

    // كتابة النتيجة النهائية في الـ Bitmap المخرجة
    matToBitmap(env, dstRGBA, dstBitmap);
}

