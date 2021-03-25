using System.Collections;
using System.Collections.Generic;
using NUnit;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.TestTools;
using UnityEngine.TestTools.Graphics;
using UnityEngine.Rendering.HighDefinition;
using UnityEngine.SceneManagement;
using UnityEngine.Events;
using System.IO;

public class HDRP_GraphicTestRunner
{
    [UnityTest]
    [PrebuildSetup("SetupGraphicsTestCases")]
    [UseGraphicsTestCases]
    [Timeout(450 * 1000)] // Set timeout to 450 sec. to handle complex scenes with many shaders (previous timeout was 300s)
    public IEnumerator Run(GraphicsTestCase testCase)
    {
        // Debug.Log($"Load Scene : {testCase.ScenePath}");
        SceneManager.LoadScene(testCase.ScenePath);

        // Arbitrary wait for 5 frames for the scene to load, and other stuff to happen (like Realtime GI to appear ...)
        for (int i=0 ; i<5 ; ++i)
            yield return new WaitForEndOfFrame();

        // Load the test settings
        var settings = GameObject.FindObjectOfType<HDRP_TestSettings>();

        var camera = GameObject.FindGameObjectWithTag("MainCamera").GetComponent<Camera>();
        if (camera == null) camera = GameObject.FindObjectOfType<Camera>();
        if (camera == null)
        {
            Assert.Fail("Missing camera for graphic tests.");
        }

        // Grab the HDCamera
        HDCamera hdCamera = HDCamera.GetOrCreate(camera);

        bool useBackBuffer = settings.ImageComparisonSettings.UseBackBuffer;

        if (useBackBuffer)
            GameViewUtils.SetGameViewSize(settings.ImageComparisonSettings.TargetWidth, settings.ImageComparisonSettings.TargetHeight);

        Time.captureFramerate = settings.captureFramerate;

        int waitFrames = settings.waitFrames;

        if (XRGraphicsAutomatedTests.enabled)
        {
            waitFrames = Unity.Testing.XR.Runtime.ConfigureMockHMD.SetupTest(settings.xrCompatible, waitFrames, settings.ImageComparisonSettings);

            // Increase tolerance to account for slight changes due to float precision
            settings.ImageComparisonSettings.AverageCorrectnessThreshold *= settings.xrThresholdMultiplier;
            settings.ImageComparisonSettings.PerPixelCorrectnessThreshold *= settings.xrThresholdMultiplier;

            // Increase number of volumetric slices to compensate for initial half-resolution due to XR single-pass optimization
            foreach (var volume in GameObject.FindObjectsOfType<Volume>())
            {
                if (volume.profile.TryGet<Fog>(out Fog fog))
                    fog.volumeSliceCount.value *= 2;
            }
        }

        if (settings.doBeforeTest != null)
        {
            settings.doBeforeTest.Invoke();

            // Wait again one frame, to be sure.
            yield return WaitFunction( useBackBuffer );
        }

        hdCamera.Reset();

        if (settings.waitForFrameCountMultiple)
        {
            // Get HDRP instance
            var hdrp = RenderPipelineManager.currentPipeline as HDRenderPipeline;

            // When we capture from the back buffer, there is no requirement of compensation frames
            // Else, given that we will render two frames, we need to compensate for them in the waiting
            var frameCountOffset = useBackBuffer ? 0 : 2;

            while (((hdCamera.cameraFrameCount + frameCountOffset) % (uint)settings.frameCountMultiple) != 0)
                WaitFunction( useBackBuffer );
        }

        // Force clear all the history buffers
        if (useBackBuffer)
            hdCamera.RequestClearHistoryBuffers();

        for (int i=0; i<waitFrames; ++i)
            yield return WaitFunction( useBackBuffer );

        var settingsSG = (GameObject.FindObjectOfType<HDRP_TestSettings>() as HDRP_ShaderGraph_TestSettings);
        if (settingsSG == null || !settingsSG.compareSGtoBI)
        {
            // Standard Test
            ImageAssert.AreEqual(testCase.ReferenceImage, camera, settings?.ImageComparisonSettings);

            // For some reason, tests on mac os have started failing with render graph enabled by default.
            // Some tests have 400+ gcalloc in them. Unfortunately it's not reproductible outside of command line so it's impossible to debug.
            // That's why we don't test on macos anymore.
            if (settings.checkMemoryAllocation && SystemInfo.graphicsDeviceType != GraphicsDeviceType.Metal)
            {
                // Does it allocate memory when it renders what's on camera?
                bool allocatesMemory = false;
                try
                {
                    // GC alloc from Camera.CustomRender (case 1206364)
                    int gcAllocThreshold = 0;

                    ImageAssert.AllocatesMemory(camera, settings?.ImageComparisonSettings, gcAllocThreshold);
                }
                catch (AssertionException)
                {
                    allocatesMemory = true;
                }
                if (allocatesMemory)
                    Assert.Fail("Allocated memory when rendering what is on camera");
            }
        }
        else
        {
            if (settingsSG.sgObjs == null)
            {
                Assert.Fail("Missing Shader Graph objects in test scene.");
            }
            if (settingsSG.biObjs == null)
            {
                Assert.Fail("Missing comparison objects in test scene.");
            }

            settingsSG.sgObjs.SetActive(true);
            settingsSG.biObjs.SetActive(false);
            yield return WaitFunction( useBackBuffer );
            yield return WaitFunction( useBackBuffer );
            bool sgFail = false;
            bool biFail = false;

            // First test: Shader Graph
            try
            {
                ImageAssert.AreEqual(testCase.ReferenceImage, camera, (settings != null) ? settings.ImageComparisonSettings : null);
            }
            catch (AssertionException)
            {
                sgFail = true;
            }

            settingsSG.sgObjs.SetActive(false);
            settingsSG.biObjs.SetActive(true);
            settingsSG.biObjs.transform.position = settingsSG.sgObjs.transform.position; // Move to the same location.
            yield return WaitFunction( useBackBuffer );
            yield return WaitFunction( useBackBuffer );

            // Second test: HDRP/Lit Materials
            try
            {
                ImageAssert.AreEqual(testCase.ReferenceImage, camera, (settings != null) ? settings.ImageComparisonSettings : null);
            }
            catch (AssertionException)
            {
                biFail = true;
            }

            // Informs which ImageAssert failed, if any.
            if (sgFail && biFail) Assert.Fail("Both Shader Graph and Non-Shader Graph Objects failed to match the reference image");
            else if (sgFail) Assert.Fail("Shader Graph Objects failed.");
            else if (biFail) Assert.Fail("Non-Shader Graph Objects failed to match Shader Graph objects.");
        }
    }

    void SetViewSize( int width, int height)
    {
#if UNITY_EDITOR
        GameViewUtils.SetGameViewSize(width, height);
#else
        Screen.SetResolution(width, height, Screen.fullScreenMode);
#endif
    }

    YieldInstruction WaitFunction(bool isBackbuffer)
    {
        return isBackbuffer ? new WaitForEndOfFrame() : null;
    }

#if UNITY_EDITOR


    [TearDown]
    public void DumpImagesInEditor()
    {
        UnityEditor.TestTools.Graphics.ResultsUtility.ExtractImagesFromTestProperties(TestContext.CurrentContext.Test);
    }

    [TearDown]
    public void TearDownXR()
    {
        XRGraphicsAutomatedTests.running = false;
    }
#endif

}
