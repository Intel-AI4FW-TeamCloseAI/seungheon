package com.example.onnxdemo

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * YOLO를 이용해 얼굴을 검출하고, 검출된 얼굴을 확장/크롭하여
 * 이진 분류 모델(deepfake_binary_s128_e5_early.onnx)로 Fake/Real을 판별하는 클래스.
 *
 * (1) detectFaces(): Bitmap → 얼굴 바운딩박스 목록
 * (2) cropFaceArea(): 바운딩박스 확장 후 Crop
 * (3) classifyBitmap(): 분류 모델로 Fake/Real 예측
 * (4) detectAndClassify(): 전체 흐름 (여러 얼굴 → 각각 분류) 또는 얼굴 無 시 전체 이미지 분류
 */
class DeepfakeDetector(
    private val yoloSession: OrtSession,  // yolov11n-face.onnx
    private val clsSession: OrtSession    // deepfake_binary_s128_e5_early.onnx
) {

    /**
     * 검출된 얼굴 바운딩박스 정보
     */
    data class FaceBox(
        val x1: Float,
        val y1: Float,
        val x2: Float,
        val y2: Float,
        val score: Float,
        val cls: Float
    )

    /**
     * detectAndClassify(...) 의 결과 한 건
     *
     * @param faceIndex   - 몇 번째 얼굴인지 (없으면 -1)
     * @param label       - "Fake" or "Real"
     * @param confidence  - 예측 확률(0.0 ~ 1.0)
     * @param message     - 예를 들어 "Face #0: Fake (95.12%)" 같은 문자열
     */
    data class DetectionResult(
        val faceIndex: Int,
        val label: String,
        val confidence: Float,
        val message: String
    )

    /**
     * YOLO 모델에 넣기 위한 전처리
     * (가정) 입력 크기: 640x640, 스케일만 [0,1], 별도 mean/std 없음
     */
    private fun preprocessForYolo(bitmap: Bitmap, targetSize: Int = 640): OnnxTensor {
        val resized = Bitmap.createScaledBitmap(bitmap, targetSize, targetSize, true)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * targetSize * targetSize)

        val pixels = IntArray(targetSize * targetSize)
        resized.getPixels(pixels, 0, targetSize, 0, 0, targetSize, targetSize)
        for (i in pixels.indices) {
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF) / 255f
            val g = ((px shr 8) and 0xFF) / 255f
            val b = (px and 0xFF) / 255f

            // YOLOv5/8 기본: RGB 순서, 0~1 범위
            floatBuffer.put(r)
            floatBuffer.put(g)
            floatBuffer.put(b)
        }
        floatBuffer.rewind()

        val env = OrtEnvironment.getEnvironment()
        return OnnxTensor.createTensor(
            env,
            floatBuffer,
            longArrayOf(1, 3, targetSize.toLong(), targetSize.toLong())
        )
    }

    /**
     * YOLO로 얼굴 검출
     * (가정) 출력 텐서 shape = [1][N][6] -> (x1, y1, x2, y2, score, class)
     *       이미 NMS 포함된 모델로 export됐다고 가정
     *
     * @param confThreshold Confidence threshold (기본 0.25)
     */
    private fun detectFaces(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f
    ): List<FaceBox> {
        val inputTensor = preprocessForYolo(bitmap, 640)

        // YOLO 모델 입력 노드 이름이 "images" 등일 수도 있으니 확인 필요
        val output = yoloSession.run(mapOf("images" to inputTensor))
        // 가정: output[0]의 shape = [1][N][6]
        val rawOutput = output[0].value as Array<Array<FloatArray>>

        val boxes = mutableListOf<FaceBox>()
        for (detection in rawOutput[0]) {
            val x1 = detection[0]
            val y1 = detection[1]
            val x2 = detection[2]
            val y2 = detection[3]
            val score = detection[4]
            val cls = detection[5]

            if (score >= confThreshold) {
                boxes.add(FaceBox(x1, y1, x2, y2, score, cls))
            }
        }
        return boxes
    }

    /**
     * 얼굴 영역을 상하좌우로 확장하여 Crop
     *
     * @param extendRatio 0.5 => 얼굴 높이의 50%, 폭의 50% 등을 확장
     *                    (필요에 따라 위/아래 개별 조정 가능)
     */
    private fun cropFaceArea(
        src: Bitmap,
        box: FaceBox,
        extendRatio: Float = 0.5f
    ): Bitmap {
        val imgW = src.width
        val imgH = src.height

        val faceW = box.x2 - box.x1
        val faceH = box.y2 - box.y1

        // ----------------------
        // 상하 확장
        // ----------------------
        // 위쪽: faceHeight * extend_ratio * 0.7
        val newY1 = (box.y1 - faceH * extendRatio * 0.7f).coerceAtLeast(0f)
        // 아래쪽: faceHeight * extend_ratio
        val newY2 = (box.y2 + faceH * extendRatio).coerceAtMost(imgH.toFloat())

        // ----------------------
        // 좌우 확장
        // ----------------------
        // 좌우: faceWidth * extend_ratio * 0.7
        val extendW = faceW * extendRatio * 0.7f
        val newX1 = (box.x1 - extendW).coerceAtLeast(0f)
        val newX2 = (box.x2 + extendW).coerceAtMost(imgW.toFloat())

        // 최종 좌표 (int로 변환)
        val left = newX1.toInt()
        val top = newY1.toInt()
        val width = (newX2 - newX1).toInt()
        val height = (newY2 - newY1).toInt()

        // width 혹은 height가 0 이하이면 잘못된 경우이므로 원본 반환
        return if (width > 0 && height > 0) {
            Bitmap.createBitmap(src, left, top, width, height)
        } else {
            src
        }
    }


    /**
     * 이진 분류 모델(deepfake_binary_s128_e5_early.onnx) 전처리 & 추론
     * - 입력 크기: 128x128
     * - 정규화: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
     * - 출력 shape=[1,2] => [logitsFake, logitsReal] (가정)
     */
    private fun classifyBitmap(bitmap: Bitmap): Pair<Int, Float> {
        // (1) Resize & Normalize
        val resized = Bitmap.createScaledBitmap(bitmap, 128, 128, true)
        val floatBuffer = FloatBuffer.allocate(1 * 3 * 128 * 128)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val pixels = IntArray(128 * 128)
        resized.getPixels(pixels, 0, 128, 0, 0, 128, 128)

        var index = 0
        pixels.indices.forEach { i ->
            val px = pixels[i]
            val r = ((px shr 16) and 0xFF) / 255f
            val g = ((px shr 8) and 0xFF) / 255f
            val b = (px and 0xFF) / 255f

            val rr = (r - mean[0]) / std[0]
            val gg = (g - mean[1]) / std[1]
            val bb = (b - mean[2]) / std[2]

            floatBuffer.put(rr)
            floatBuffer.put(gg)
            floatBuffer.put(bb)
            index++
        }
        floatBuffer.rewind()

        val env = OrtEnvironment.getEnvironment()
        val inputTensor = OnnxTensor.createTensor(env, floatBuffer, longArrayOf(1, 3, 128, 128))

        // (2) 추론
        // 가정: 입력 노드 이름이 "input"
        val output = clsSession.run(mapOf("input" to inputTensor))
        val logits = output[0].value as Array<FloatArray> // shape [1,2]
        val logit = logits[0]                            // [FakeLogit, RealLogit]

        // (3) Softmax -> argmax
        val expFake = exp(logit[0].toDouble()).toFloat()
        val expReal = exp(logit[1].toDouble()).toFloat()
        val sumExp = expFake + expReal
        val probFake = expFake / sumExp
        val probReal = expReal / sumExp

        // 0:Fake, 1:Real
        return if (probFake >= probReal) {
            0 to probFake
        } else {
            1 to probReal
        }
    }

    /**
     * 전체 이미지를 입력받아:
     * 1) 얼굴 검출
     * 2) 얼굴 있다면 각 얼굴을 크롭 & 분류
     * 3) 얼굴 없다면 전체 이미지 분류
     * 4) (0: Fake, 1: Real) + 확률
     *
     * 여러 얼굴이 있을 경우, 얼굴마다 DetectionResult를 리스트로 반환
     */
    fun detectAndClassify(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f,
        extendRatio: Float = 0.5f
    ): List<DetectionResult> {
        val faceBoxes = detectFaces(bitmap, confThreshold)

        // 얼굴이 하나도 없으면 -> 전체 이미지 분류
        if (faceBoxes.isEmpty()) {
            val (cls, score) = classifyBitmap(bitmap)
            val labelStr = if (cls == 0) "Fake" else "Real"
            val msg = "No Face Detected → $labelStr (%.2f%%)".format(score * 100)
            return listOf(
                DetectionResult(
                    faceIndex = -1,
                    label = labelStr,
                    confidence = score,
                    message = msg
                )
            )
        }

        // 얼굴이 있으면 -> 각각 크롭해서 분류
        val results = mutableListOf<DetectionResult>()
        faceBoxes.forEachIndexed { index, box ->
            val cropped = cropFaceArea(bitmap, box, extendRatio)
            val (cls, score) = classifyBitmap(cropped)

            val labelStr = if (cls == 0) "Fake" else "Real"
            val msg = "Face #$index: $labelStr (%.2f%%)".format(score * 100)

            results.add(
                DetectionResult(
                    faceIndex = index,
                    label = labelStr,
                    confidence = score,
                    message = msg
                )
            )
        }
        return results
    }
}
