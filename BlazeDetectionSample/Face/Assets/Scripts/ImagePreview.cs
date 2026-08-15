using System;
using UnityEngine;

public class ImagePreview : MonoBehaviour
{
    public GameObject imageQuad;

    public void SetTexture(Texture texture, bool mirror = false)
    {
        var material = imageQuad.GetComponent<MeshRenderer>().material;
        material.mainTexture = texture;
        // mirror by flipping the uvs, flipping the quad would invert the winding order and cull it
        material.mainTextureScale = new Vector2(mirror ? -1f : 1f, 1f);
        material.mainTextureOffset = new Vector2(mirror ? 1f : 0f, 0f);
        var aspectRatio = texture.width / (float)texture.height;
        imageQuad.transform.localScale = new Vector3(aspectRatio, 1f, 1f);
    }
}
