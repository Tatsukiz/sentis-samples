using System;
using Unity.Mathematics;
using Unity.InferenceEngine;
using UnityEngine;

public class FaceDetection : MonoBehaviour
{
    public enum InputMode
    {
        Image,
        Webcam
    }

    public FacePreview[] facePreviews;
    public ImagePreview imagePreview;
    public ModelAsset faceDetector;
    public TextAsset anchorsCSV;

    [Header("Input")]
    [Tooltip("The input source is resolved once on Start, changing this during play mode has no effect")]
    public InputMode inputMode = InputMode.Webcam;
    [Tooltip("Used when inputMode is Image, and as a fallback when no webcam is available")]
    public Texture2D imageTexture;

    [Header("Webcam")]
    [Tooltip("Leave empty to use the first device reported by the platform")]
    public string webcamDeviceName = "";
    public Vector2Int webcamResolution = new Vector2Int(1280, 720);
    public int webcamFPS = 30;
    [Tooltip("Mirror the preview and the detections, as expected of a front facing camera")]
    public bool mirrorWebcam;

    public float iouThreshold = 0.3f;
    public float scoreThreshold = 0.5f;

    const int k_NumAnchors = 896;
    float[,] m_Anchors;

    const int k_NumKeypoints = 6;
    const int detectorInputSize = 128;

    Worker m_FaceDetectorWorker;
    Tensor<float> m_DetectorInput;
    Awaitable m_DetectAwaitable;

    WebCamTexture m_WebcamTexture;
    bool m_Mirror;

    float m_TextureWidth;
    float m_TextureHeight;

    public async void Start()
    {
        m_Anchors = BlazeUtils.LoadAnchors(anchorsCSV.text, k_NumAnchors);

        var faceDetectorModel = ModelLoader.Load(faceDetector);

        // post process the model to filter scores + nms select the best faces
        var graph = new FunctionalGraph();
        var input = graph.AddInput(faceDetectorModel, 0);
        var outputs = Functional.Forward(faceDetectorModel, 2 * input - 1);
        var boxes = outputs[0]; // (1, 896, 16)
        var scores = outputs[1]; // (1, 896, 1)
        var anchorsData = new float[k_NumAnchors * 4];
        Buffer.BlockCopy(m_Anchors, 0, anchorsData, 0, anchorsData.Length * sizeof(float));
        var anchors = Functional.Constant(new TensorShape(k_NumAnchors, 4), anchorsData);
        var idx_scores_boxes = BlazeUtils.NMSFiltering(boxes, scores, anchors, detectorInputSize, iouThreshold, scoreThreshold);
        faceDetectorModel = graph.Compile(idx_scores_boxes.Item1, idx_scores_boxes.Item2, idx_scores_boxes.Item3);

        m_FaceDetectorWorker = new Worker(faceDetectorModel, BackendType.GPUCompute);

        m_DetectorInput = new Tensor<float>(new TensorShape(1, detectorInputSize, detectorInputSize, 3));

        try
        {
            Texture sourceTexture = imageTexture;

            if (inputMode == InputMode.Webcam)
            {
                var webcamTexture = await StartWebcam();
                if (webcamTexture != null)
                {
                    sourceTexture = webcamTexture;
                    m_Mirror = mirrorWebcam;
                }
                else
                {
                    Debug.LogWarning("No webcam available, falling back to imageTexture.", this);
                }
            }

            if (sourceTexture == null)
            {
                Debug.LogError($"No input texture for input mode {inputMode}, assign imageTexture or connect a webcam.", this);
                return;
            }

            while (true)
            {
                // the webcam delivers a new image only every few frames, no need to run the model in between
                if (m_WebcamTexture != null && !m_WebcamTexture.didUpdateThisFrame)
                {
                    await Awaitable.NextFrameAsync(destroyCancellationToken);
                    continue;
                }

                m_DetectAwaitable = Detect(sourceTexture);
                await m_DetectAwaitable;
            }
        }
        catch (OperationCanceledException)
        {
            // the component was destroyed while detecting
        }
        finally
        {
            m_FaceDetectorWorker.Dispose();
            m_DetectorInput.Dispose();
            StopWebcam();
        }
    }

    async Awaitable<WebCamTexture> StartWebcam()
    {
        if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
        {
            var authorization = Application.RequestUserAuthorization(UserAuthorization.WebCam);
            while (!authorization.isDone)
                await Awaitable.NextFrameAsync(destroyCancellationToken);

            if (!Application.HasUserAuthorization(UserAuthorization.WebCam))
                return null;
        }

        var devices = WebCamTexture.devices;
        if (devices.Length == 0)
            return null;

        var deviceName = devices[0].name;
        if (!string.IsNullOrEmpty(webcamDeviceName))
        {
            var found = false;
            foreach (var device in devices)
                found |= device.name == webcamDeviceName;

            if (found)
                deviceName = webcamDeviceName;
            else
                Debug.LogWarning($"Webcam '{webcamDeviceName}' not found, using '{deviceName}' instead.", this);
        }

        m_WebcamTexture = new WebCamTexture(deviceName, webcamResolution.x, webcamResolution.y, webcamFPS);
        m_WebcamTexture.Play();

        // the actual resolution is only known once the device has delivered its first frame
        var timeout = Time.realtimeSinceStartup + 5f;
        while (m_WebcamTexture.width <= 16)
        {
            if (Time.realtimeSinceStartup > timeout)
            {
                Debug.LogWarning($"Webcam '{deviceName}' did not start within 5s.", this);
                StopWebcam();
                return null;
            }

            await Awaitable.NextFrameAsync(destroyCancellationToken);
        }

        return m_WebcamTexture;
    }

    void StopWebcam()
    {
        if (m_WebcamTexture == null)
            return;

        m_WebcamTexture.Stop();
        Destroy(m_WebcamTexture);
        m_WebcamTexture = null;
    }

    Vector3 ImageToWorld(Vector2 position)
    {
        var world = (position - 0.5f * new Vector2(m_TextureWidth, m_TextureHeight)) / m_TextureHeight;
        if (m_Mirror)
            world.x = -world.x;
        return world;
    }

    async Awaitable Detect(Texture texture)
    {
        m_TextureWidth = texture.width;
        m_TextureHeight = texture.height;
        imagePreview.SetTexture(texture, m_Mirror);

        var size = Mathf.Max(texture.width, texture.height);

        // The affine transformation matrix to go from tensor coordinates to image coordinates
        var scale = size / (float)detectorInputSize;
        var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))), BlazeUtils.ScaleMatrix(new Vector2(scale, -scale)));
        BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);

        m_FaceDetectorWorker.Schedule(m_DetectorInput);

        var outputIndicesAwaitable = (m_FaceDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
        var outputScoresAwaitable = (m_FaceDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
        var outputBoxesAwaitable = (m_FaceDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

        using var outputIndices = await outputIndicesAwaitable;
        using var outputScores = await outputScoresAwaitable;
        using var outputBoxes = await outputBoxesAwaitable;

        var numFaces = outputIndices.shape.length;

        for (var i = 0; i < facePreviews.Length; i++)
        {
            var active = i < numFaces;
            facePreviews[i].SetActive(active);
            if (!active)
                continue;

            var idx = outputIndices[i];

            var anchorPosition = detectorInputSize * new float2(m_Anchors[idx, 0], m_Anchors[idx, 1]);

            var box_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 0], outputBoxes[0, i, 1]));
            var boxTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 0] + 0.5f * outputBoxes[0, i, 2], outputBoxes[0, i, 1] + 0.5f * outputBoxes[0, i, 3]));

            var boxSize = 2f * (boxTopRight_ImageSpace - box_ImageSpace);
            facePreviews[i].SetBoundingBox(true, ImageToWorld(box_ImageSpace), boxSize / texture.height);

            for (var j = 0; j < k_NumKeypoints; j++)
            {
                var position_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBoxes[0, i, 4 + 2 * j + 0], outputBoxes[0, i, 4 + 2 * j + 1]));
                facePreviews[i].SetKeypoint(j, true, ImageToWorld(position_ImageSpace));
            }
        }

        // if no faces are recognized then the awaitable outputs return synchronously so we need to add an extra frame await here to allow the main thread to run
        if (numFaces == 0)
            await Awaitable.NextFrameAsync();
    }

    void OnDestroy()
    {
        m_DetectAwaitable?.Cancel();
    }
}
