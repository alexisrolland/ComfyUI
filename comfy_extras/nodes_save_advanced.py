import io
import av
import os
import json
import torch
import struct
import zlib
import logging
import numpy as np
import folder_paths
from comfy_api.latest import IO
from typing_extensions import override
from comfy_api.latest import ComfyExtension
from comfy.cli_args import args

def create_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Creates a valid PNG chunk with Length, Type, Data, and CRC32."""
    chunk = struct.pack('>I', len(data)) + chunk_type + data
    crc = zlib.crc32(chunk_type + data) & 0xffffffff
    return chunk + struct.pack('>I', crc)

def inject_comfy_metadata_png(png_bytes, prompt=None, extra_pnginfo=None):
    # IEND chunk is the last 12 bytes of png files
    content = png_bytes[:-12]
    iend = png_bytes[-12:]

    metadata_chunks = b""

    if prompt is not None:
        payload = b'prompt\x00' + json.dumps(prompt).encode('utf-8')
        metadata_chunks += create_png_chunk(b'tEXt', payload)

    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            payload = k.encode('utf-8') + b'\x00' + json.dumps(v).encode('utf-8')
            metadata_chunks += create_png_chunk(b'tEXt', payload)

    return content + metadata_chunks + iend

def inject_comfy_metadata_exr(exr_bytes: bytes, prompt, extra_pnginfo) -> bytes:
    # skip magic and version
    idx = 8

    # parse through existing attributes to find the end of the header
    while True:
        name_start = idx
        while exr_bytes[idx] != 0:
            idx += 1
        name = exr_bytes[name_start:idx]
        idx += 1

        # empty name means we hit the header terminator
        if len(name) == 0:
            break

        # skip attribute type string
        while exr_bytes[idx] != 0:
            idx += 1
        idx += 1

        # read attribute size and skip the value
        attr_size = struct.unpack('<I', exr_bytes[idx:idx+4])[0]
        idx += 4 + attr_size

    # offset table starts right after the header terminator
    table_start = idx

    # build comfyui metadata payload
    payload = b""
    if prompt is not None:
        prompt_str = json.dumps(prompt).encode('utf-8')
        payload += b"prompt\x00string\x00" + struct.pack('<I', len(prompt_str)) + prompt_str
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            k_enc = k.encode('utf-8')[:254]
            v_enc = json.dumps(v).encode('utf-8')
            payload += k_enc + b"\x00string\x00" + struct.pack('<I', len(v_enc)) + v_enc

    # find the first pixel offset to calculate the table size
    min_offset = struct.unpack('<Q', exr_bytes[table_start:table_start+8])[0]
    num_entries = 1
    while table_start + num_entries * 8 < min_offset:
        offset = struct.unpack('<Q', exr_bytes[table_start + num_entries*8 : table_start + num_entries*8 + 8])[0]
        if offset < min_offset:
            min_offset = offset
        num_entries += 1

    # shift table pointers by the payload size
    shift_amount = len(payload)
    new_table = bytearray()
    for i in range(num_entries):
        offset = struct.unpack('<Q', exr_bytes[table_start + i*8 : table_start + i*8 + 8])[0]
        new_table.extend(struct.pack('<Q', offset + shift_amount))

    # stitch the file back together with the new header and updated table
    return exr_bytes[:table_start - 1] + payload + b'\x00' + new_table + exr_bytes[table_start + num_entries*8:]

def inject_comfy_metadata_avif(avif_bytes: bytes, prompt, extra_pnginfo) -> bytes:
    metadata = {}
    if prompt is not None:
        metadata["prompt"] = prompt
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            metadata[k] = v

    payload = json.dumps(metadata).encode('utf-8')

    # 16-byte uuid required by isobmff spec
    # 'comfyui_workflow' is exactly 16 bytes long!
    comfy_uuid = b'comfyui_workflow'

    # box size: 4 (size) + 4 (type) + 16 (uuid) + payload length
    box_size = 4 + 4 + 16 + len(payload)
    uuid_box = struct.pack('>I', box_size) + b'uuid' + comfy_uuid + payload

    # isobmff allows top-level boxes at the end of the file.
    return avif_bytes + uuid_box

class SaveImageAdvanced(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SaveImageAdvanced",
            category="image/advanced_io",
            output_node=True,
            inputs=[
                IO.Image.Input("images"),
                IO.String.Input("filename_prefix", default="ComfyUI"),
                IO.Combo.Input("file_format", options=["png", "exr", "avif"], default="png"),
                IO.Combo.Input("bit_depth", options=["8-bit", "16-bit", "32-bit"], default="8-bit"),
                IO.Boolean.Input("embed_workflow", default=True),
                IO.Hidden.Input("prompt", type="PROMPT"),
                IO.Hidden.Input("extra_pnginfo", type="EXTRA_PNGINFO"),
            ],
            outputs=[]
        )

    @classmethod
    def execute(cls, images, filename_prefix="ComfyUI", file_format="png", bit_depth="8-bit",
                embed_workflow=True, prompt=None, extra_pnginfo=None) -> IO.NodeOutput:

        output_dir = folder_paths.get_output_directory()

        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])

        results = list()

        for batch_number, image in enumerate(images):
            img_tensor = image.clone()

            height, width, num_channels = img_tensor.shape
            has_alpha = (num_channels == 4)

            # file pathing
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}.{file_format}"
            file_path = os.path.join(full_output_folder, file)

            if file_format in ["png", "exr", "avif"]:

                # safe bit downcasting
                if (file_format == "png" or file_format == "avif") and bit_depth == "32-bit":
                    bit_depth = "16-bit"
                if file_format == "exr" and bit_depth == "8-bit":
                    bit_depth = "16-bit"

                if bit_depth == "32-bit":
                    img_np = img_tensor.cpu().numpy()
                    # rgba128le handles 4x32f, gbrpf32le handles 3x32f planar
                    av_fmt = 'rgba128le' if has_alpha else 'gbrpf32le'
                elif bit_depth == "16-bit":
                    img_np = (img_tensor * 65535.0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                    if file_format == "png":
                        # png requires Big-Endian (be) for 16-bit
                        av_fmt = 'rgba64be' if has_alpha else 'rgb48be'
                        img_np = img_np.byteswap().view(img_np.dtype.newbyteorder('>'))
                    else:
                        av_fmt = 'rgba64le' if has_alpha else 'rgb48le'
                else:
                    img_np = (img_tensor * 255.0).clamp(0, 255).to(torch.int32).cpu().numpy().astype(np.uint8)
                    av_fmt = 'rgba' if has_alpha else 'rgb24'

                memory_buffer = io.BytesIO()
                container_format = "image2" if file_format in ["png", "exr"] else "avif"
                container = av.open(memory_buffer, mode='w', format=container_format)

                if file_format == "exr":
                    stream = container.add_stream('exr', rate=1)
                    stream.pix_fmt = av_fmt
                elif file_format == "avif":
                    stream = container.add_stream('av1', rate=1)
                    # YUV color spaces
                    stream.pix_fmt = 'yuv444p12le' if bit_depth in["16-bit", "32-bit"] else 'yuv444p'
                elif file_format == "png":
                    stream = container.add_stream('png', rate=1)
                    stream.pix_fmt = av_fmt

                stream.width = width
                stream.height = height

                # planar: all red, all blue, all green instead of r, g, b, r, g, b
                is_planar = av_fmt.startswith('gbrp') or 'p' in av_fmt.split('rgba')[-1]
                if is_planar:
                    img_np = img_np.transpose(2, 0, 1)

                try:
                    frame = av.VideoFrame.from_ndarray(img_np, format=av_fmt)
                except ValueError:
                    # FFMPEG Float32 Fallback: not all ffmpeg versions are able to handle float32 format for images
                    # float16 fallback conversion
                    logging.warning("[WARNING] Current FFMPEG Binary can't save float32 images. Fallbacking to float16")
                    img_np = (img_tensor * 65535.0).clamp(0, 65535).to(torch.int32).cpu().numpy().astype(np.uint16)
                    av_fmt = 'rgba64le' if has_alpha else 'rgb48le'
                    frame = av.VideoFrame.from_ndarray(img_np, format=av_fmt)
                    if file_format == "exr" or file_format == "png":
                        stream.pix_fmt = av_fmt

                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

                container.close()

                final_bytes = memory_buffer.getvalue()

                if embed_workflow and not args.disable_metadata:
                    if file_format == "png":
                        final_bytes = inject_comfy_metadata_png(final_bytes, prompt, extra_pnginfo)
                    elif file_format == "exr":
                        final_bytes = inject_comfy_metadata_exr(final_bytes, prompt, extra_pnginfo)
                    else:
                        final_bytes = inject_comfy_metadata_avif(final_bytes, prompt, extra_pnginfo)

                with open(file_path, "wb") as f:
                    f.write(final_bytes)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1

        return IO.NodeOutput(ui={"images": results})

# Rec.709 to Rec.2020 Gamut Conversion Matrix
M_709_to_2020 = torch.tensor([[0.6274, 0.3293, 0.0433],[0.0691, 0.9195, 0.0114],[0.0164, 0.0880, 0.8956]
])

# Rec.2020 to Rec.709 Gamut Conversion Matrix
M_2020_to_709 = torch.tensor([[ 1.6605, -0.5876, -0.0728],[-0.1246,  1.1329, -0.0083],[-0.0182, -0.1006,  1.1187]
])

def srgb_to_linear(tensor):
    mask = tensor <= 0.04045
    return torch.where(mask, tensor / 12.92, torch.pow((tensor + 0.055) / 1.055, 2.4))

def linear_to_srgb(tensor):
    mask = tensor <= 0.0031308
    return torch.where(mask, tensor * 12.92, 1.055 * torch.pow(tensor.clamp(min=1e-8), 1.0 / 2.4) - 0.055)

def linear_to_pq(linear_tensor):
    """SMPTE ST 2084 (PQ) encoding"""
    m1, m2 = (2610 / 4096 / 4), (2523 / 4096 * 128)
    c1, c2, c3 = (3424 / 4096), (2413 / 4096 * 32), (2392 / 4096 * 32)
    l_norm = torch.clamp(linear_tensor, 0.0, 1.0)
    l_m1 = torch.pow(l_norm, m1)
    return torch.pow((c1 + c2 * l_m1) / (1 + c3 * l_m1), m2)

def pq_to_linear(pq_tensor):
    """Inverse SMPTE ST 2084 (PQ) decoding"""
    m1, m2 = (2610 / 4096 / 4), (2523 / 4096 * 128)
    c1, c2, c3 = (3424 / 4096), (2413 / 4096 * 32), (2392 / 4096 * 32)
    n = torch.pow(torch.clamp(pq_tensor, 0.0, 1.0), 1/m2)
    return torch.pow(torch.clamp((n - c1) / (c2 - c3 * n), min=0.0), 1/m1)

class ConvertColorSpace(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Convert Color Space",
            category="image/color",
            inputs=[
                IO.Image.Input("images"),
                IO.Combo.Input("source_color_space", options=["sRGB", "Linear", "HDR (Rec.2020)", "Grayscale"], default="sRGB"),
                IO.Combo.Input("target_color_space", options=["sRGB", "Linear", "HDR (Rec.2020)", "Grayscale"], default="Linear"),
            ],
            outputs=[
                IO.Image.Output("images"),
            ]
        )

    @classmethod
    def execute(cls, images, source_color_space, target_color_space) -> IO.NodeOutput:
        img_tensor = images.clone()
        device = img_tensor.device

        has_alpha = img_tensor.shape[-1] == 4
        alpha = img_tensor[..., 3:4] if has_alpha else None
        rgb = img_tensor[..., :3]

        # turn source into linear
        if source_color_space == "sRGB":
            rgb = srgb_to_linear(rgb)

        elif source_color_space == "Grayscale":
            # assume Grayscale has sRGB gamma
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            rgb = luma.unsqueeze(-1).repeat(1, 1, 1, 3)
            rgb = linear_to_srgb(rgb)

        elif source_color_space == "HDR (Rec.2020)":
            # assuming Linear Rec.2020 input. Convert to Linear Rec.709
            matrix = M_2020_to_709.to(device)
            rgb = pq_to_linear(rgb)
            rgb = torch.matmul(rgb, matrix.T)


        # turn source into target space
        if target_color_space == "sRGB":
            rgb = linear_to_srgb(rgb)

        elif target_color_space == "Grayscale":
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            rgb = luma.unsqueeze(-1).repeat(1, 1, 1, 3)
            rgb = linear_to_srgb(rgb) # reapply srgb gamma

        elif target_color_space == "HDR (Rec.2020)":
            # convert Gamut from Linear Rec.709 to Linear Rec.2020
            rgb = torch.matmul(rgb, M_709_to_2020.to(device).T).clamp(min=0)
            rgb = linear_to_pq(rgb)

        img_tensor = torch.cat([rgb, alpha], dim=-1) if has_alpha else rgb

        return IO.NodeOutput(images=img_tensor)

class AdvancedImageSave(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [
            SaveImageAdvanced,
            ConvertColorSpace
        ]


async def comfy_entrypoint() -> AdvancedImageSave:
    return AdvancedImageSave()
