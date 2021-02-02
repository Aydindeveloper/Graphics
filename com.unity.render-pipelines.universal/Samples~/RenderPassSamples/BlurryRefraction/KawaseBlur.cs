using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

[System.Serializable]
public class KawaseBlurSettings
{
    public RenderPassEvent renderPassEvent = RenderPassEvent.AfterRenderingSkybox;
    public Material blurMaterial = null;

    [Range(1, 15)]
    public int blurPasses = 2;

    [Range(1, 4)]
    public int downsample = 1;

    public bool copyToFramebuffer;
    public string targetName = "_blurTexture";
}
class KawaseBlur : ScriptableRenderPass
{
    string profilerTag;

    int tmpId1;
    int tmpId2;

    RenderTargetIdentifier tmpRT1;
    RenderTargetIdentifier tmpRT2;

    RenderTargetIdentifier cameraColorTexture;
    KawaseBlurSettings settings;

    public KawaseBlur(string profilerTag, KawaseBlurSettings settings)
    {
        this.profilerTag = profilerTag;
        this.settings = settings;
        renderPassEvent = settings.renderPassEvent;
    }

    // This method is called before executing the render pass.
    // It can be used to configure render targets and their clear state. Also to create temporary render target textures.
    // When empty this render pass will render to the active camera render target.
    // You should never call CommandBuffer.SetRenderTarget. Instead call <c>ConfigureTarget</c> and <c>ConfigureClear</c>.
    // The render pipeline will ensure target setup and clearing happens in a performant manner.
    public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
    {
        var desc = renderingData.cameraData.cameraTargetDescriptor;
        var width = desc.width;
        var height = desc.height;
        desc.width = width / settings.downsample;
        desc.height = height / settings.downsample;

        tmpId1 = Shader.PropertyToID("tmpBlurRT1");
        tmpId2 = Shader.PropertyToID("tmpBlurRT2");
        cmd.GetTemporaryRT(tmpId1, desc, FilterMode.Bilinear);
        cmd.GetTemporaryRT(tmpId2, desc, FilterMode.Bilinear);

        desc.width = width;
        desc.height = height;

        tmpRT1 = new RenderTargetIdentifier(tmpId1);
        tmpRT2 = new RenderTargetIdentifier(tmpId2);

        ConfigureTarget(tmpRT1);
        ConfigureTarget(tmpRT2);
    }

    // Here you can implement the rendering logic.
    // Use <c>ScriptableRenderContext</c> to issue drawing commands or execute command buffers
    // https://docs.unity3d.com/ScriptReference/Rendering.ScriptableRenderContext.html
    // You don't have to call ScriptableRenderContext.submit, the render pipeline will call it at specific points in the pipeline.
    public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
    {
        cameraColorTexture = ((UniversalRenderPipelineAsset)GraphicsSettings.currentRenderPipeline).scriptableRenderer.cameraColorTarget;
        CommandBuffer cmd = CommandBufferPool.Get(profilerTag);

        RenderTextureDescriptor opaqueDesc = renderingData.cameraData.cameraTargetDescriptor;
        opaqueDesc.depthBufferBits = 0;

        // first pass
        // cmd.GetTemporaryRT(tmpId1, opaqueDesc, FilterMode.Bilinear);
        cmd.SetGlobalFloat("_offset", 1.5f);
        //cmd.Blit(cameraColorTexture, tmpRT1, settings.blurMaterial);
        cmd.SetGlobalTexture("_BaseMap", cameraColorTexture);
        cmd.SetRenderTarget(tmpRT1);
        cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, settings.blurMaterial, 0, -1);

        for (var i = 1; i < settings.blurPasses - 1; i++)
        {
            cmd.SetGlobalFloat("_offset", 0.5f + i);
            //cmd.Blit(tmpRT1, tmpRT2, settings.blurMaterial);
            cmd.SetGlobalTexture("_BaseMap", tmpRT1);
            cmd.SetRenderTarget(tmpRT2);
            cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, settings.blurMaterial, 0, -1);
            // pingpong
            var rttmp = tmpRT1;
            tmpRT1 = tmpRT2;
            tmpRT2 = rttmp;
        }

        // final pass
        cmd.SetGlobalFloat("_offset", 0.5f + settings.blurPasses - 1f);
        if (settings.copyToFramebuffer)
        {
            //cmd.Blit(tmpRT1, cameraColorTexture, settings.blurMaterial);
            cmd.SetGlobalTexture("_BaseMap", tmpRT1);
            cmd.SetRenderTarget(cameraColorTexture);
            cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, settings.blurMaterial, 0, -1);
        }
        else
        {
            //cmd.Blit(tmpRT1, tmpRT2, settings.blurMaterial);
            cmd.SetGlobalTexture("_BaseMap", tmpRT1);
            cmd.SetRenderTarget(tmpRT2);
            cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, settings.blurMaterial, 0, -1);
            cmd.SetGlobalTexture(settings.targetName, tmpRT2);
        }

        context.ExecuteCommandBuffer(cmd);

        CommandBufferPool.Release(cmd);
    }

    // Cleanup any allocated resources that were created during the execution of this render pass.
    public override void OnCameraCleanup(CommandBuffer cmd)
    {
    }
}
