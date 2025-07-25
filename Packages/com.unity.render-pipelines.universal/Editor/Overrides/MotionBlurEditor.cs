using UnityEngine.Rendering.Universal;

namespace UnityEditor.Rendering.Universal
{
    [CustomEditor(typeof(MotionBlur))]
    sealed class MotionBlurEditor : VolumeComponentEditor
    {
        SerializedDataParameter m_Mode;
        SerializedDataParameter m_Quality;
        SerializedDataParameter m_Intensity;
        SerializedDataParameter m_Clamp;
        SerializedDataParameter m_Separation;


        public override void OnEnable()
        {
            var o = new PropertyFetcher<MotionBlur>(serializedObject);

            m_Mode = Unpack(o.Find(x => x.mode));
            m_Quality = Unpack(o.Find(x => x.quality));
            m_Intensity = Unpack(o.Find(x => x.intensity));
            m_Clamp = Unpack(o.Find(x => x.clamp));
            m_Separation = Unpack(o.Find(x => x.separation));

        }

        public override void OnInspectorGUI()
        {
            //PropertyField(m_Mode);

            //if (m_Mode.value.intValue == (int)MotionBlurMode.CameraOnly)
            //{
            PropertyField(m_Mode);
            PropertyField(m_Quality);
            PropertyField(m_Intensity);
            PropertyField(m_Clamp);
            PropertyField(m_Separation);

            //}
            //else
            //{
            //    EditorGUILayout.HelpBox("Object motion blur is not supported on the Universal Render Pipeline yet.", MessageType.Info);
            //}
        }
    }
}
