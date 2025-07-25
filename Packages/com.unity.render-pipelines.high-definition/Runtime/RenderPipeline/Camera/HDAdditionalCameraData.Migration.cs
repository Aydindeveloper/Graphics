using System;
using UnityEngine.Serialization;

namespace UnityEngine.Rendering.HighDefinition
{
    public partial class HDAdditionalCameraData : IVersionable<HDAdditionalCameraData.Version>
    {
        /// <summary>
        /// Define migration versions of the HDAdditionalCameraData
        /// </summary>
        protected enum Version
        {
            /// <summary>Version Step.</summary>
            None,
            /// <summary>Version Step.</summary>
            First,
            /// <summary>Version Step.</summary>
            SeparatePassThrough,
            /// <summary>Version Step.</summary>
            UpgradingFrameSettingsToStruct,
            /// <summary>Version Step.</summary>
            AddAfterPostProcessFrameSetting,
            /// <summary>Version Step.</summary>
            AddFrameSettingSpecularLighting, // Not used anymore
            /// <summary>Version Step.</summary>
            AddReflectionSettings,
            /// <summary>Version Step.</summary>
            AddCustomPostprocessAndCustomPass,
            /// <summary>Version Step.</summary>
            UpdateMSAA,
            /// <summary>Version Step.</summary>
            UpdatePhysicalCameraPropertiesToCore
        }

        [SerializeField, FormerlySerializedAs("version")]
        [ExcludeCopy]
        Version m_Version = MigrationDescription.LastVersion<Version>();

        private static readonly MigrationDescription<Version, HDAdditionalCameraData> k_Migration = MigrationDescription.New(
            MigrationStep.New(Version.SeparatePassThrough, (HDAdditionalCameraData data) =>
            {
#pragma warning disable 618 // Type or member is obsolete
                switch ((int)data.m_ObsoleteRenderingPath)
#pragma warning restore 618
                {
                    case 0: //former RenderingPath.UseGraphicsSettings
                        data.fullscreenPassthrough = false;
                        data.customRenderingSettings = false;
                        break;
                    case 1: //former RenderingPath.Custom
                        data.fullscreenPassthrough = false;
                        data.customRenderingSettings = true;
                        break;
                    case 2: //former RenderingPath.FullscreenPassthrough
                        data.fullscreenPassthrough = true;
                        data.customRenderingSettings = false;
                        break;
                }
            }),
            MigrationStep.New(Version.UpgradingFrameSettingsToStruct, (HDAdditionalCameraData data) =>
            {
#pragma warning disable 618 // Type or member is obsolete
                if (data.m_ObsoleteFrameSettings != null)
                    FrameSettings.MigrateFromClassVersion(ref data.m_ObsoleteFrameSettings, ref data.renderingPathCustomFrameSettings, ref data.renderingPathCustomFrameSettingsOverrideMask);
#pragma warning restore 618
            }),
            MigrationStep.New(Version.AddAfterPostProcessFrameSetting, (HDAdditionalCameraData data) =>
            {
                FrameSettings.MigrateToAfterPostprocess(ref data.renderingPathCustomFrameSettings);
            }),
            MigrationStep.New(Version.AddReflectionSettings, (HDAdditionalCameraData data) =>
                FrameSettings.MigrateToDefaultReflectionSettings(ref data.renderingPathCustomFrameSettings)
                ),
            MigrationStep.New(Version.AddCustomPostprocessAndCustomPass, (HDAdditionalCameraData data) =>
            {
                FrameSettings.MigrateToCustomPostprocessAndCustomPass(ref data.renderingPathCustomFrameSettings);
            }),
            MigrationStep.New(Version.UpdateMSAA, (HDAdditionalCameraData data) =>
            {
                FrameSettings.MigrateMSAA(ref data.renderingPathCustomFrameSettings, ref data.renderingPathCustomFrameSettingsOverrideMask);
            }),
            MigrationStep.New(Version.UpdatePhysicalCameraPropertiesToCore, (HDAdditionalCameraData data) =>
            {
                var camera = data.GetComponent<Camera>();
#pragma warning disable CS0618 // Type or member is obsolete
                var physicalProps = data.physicalParameters;
#pragma warning restore CS0618
                if (camera != null)
                {
                    camera.iso = physicalProps.iso;
                    camera.shutterSpeed = physicalProps.shutterSpeed;
                    camera.aperture = physicalProps.aperture;
                    camera.focusDistance = physicalProps.focusDistance;
                    camera.bladeCount = physicalProps.bladeCount;
                    camera.curvature = physicalProps.curvature;
                    camera.barrelClipping = physicalProps.barrelClipping;
                    camera.anamorphism = physicalProps.anamorphism;
                }
            })
        );

        Version IVersionable<Version>.version { get => m_Version; set => m_Version = value; }

        void Awake() => k_Migration.Migrate(this);

#pragma warning disable 649 // Field never assigned
        [SerializeField, FormerlySerializedAs("renderingPath"), Obsolete("For Data Migration. #from(2021.1)")]
        [ExcludeCopy]
        int m_ObsoleteRenderingPath;
        [SerializeField]
        [FormerlySerializedAs("serializedFrameSettings"), FormerlySerializedAs("m_FrameSettings"), Obsolete("For Data Migration. #from(2021.1)")]
        [ExcludeCopy]
        ObsoleteFrameSettings m_ObsoleteFrameSettings;
#pragma warning restore 649
    }
}
